"""Two-stage SPD reconstruction classifier.

The pipeline:

1. **Stage A — reconstruction**: train an :class:`igl.IGLModule` to map a
   log-Eig-vectorized SPD input to itself (an autoencoder in log-Euclidean
   tangent space). The loss is AIRM² — i.e. the Riemannian distance between
   the input SPD and the SPD recovered from the predicted log-Eig vector.
2. **Stage B — readout**: with the encoder frozen, compute the Green-kernel
   design matrix Φ on the training data and fit a
   :class:`sklearn.linear_model.LogisticRegression` on ``(Φ, y_label)``.

This decoupling lets the encoder learn the SPD manifold geometry without
seeing class labels — a strong regulariser when labels are noisy or scarce —
and makes the readout a single convex problem.

Optional ``orthogonality_weight > 0`` plugs an
:class:`igl.spd.OrthogonalityPenalty` into stage A.
"""

# Stubs for sklearn (LogisticRegression.fit/predict/predict_proba) and torch
# (manual_seed) are partial; silence Unknown-member diagnostics module-wide
# instead of annotating each call. The wrapper class is thin, so this trades
# inline noise for a single header.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from igl.config import EncoderConfig, IGLConfig, KernelConfig, MatryoshkaConfig
from igl.core.normalization import normalize_phi
from igl.core.trainer import MatryoshkaTrainer
from igl.exceptions import IGLConfigError, IGLNotFittedError
from igl.matryoshka.dimension_curve import detect_elbow, eval_dimension_curve
from igl.nn.module import IGLModule
from igl.spd.airm import AIRMLoss
from igl.spd.orthogonality import OrthogonalityPenalty
from igl.types import NormalizeModeLike

if TYPE_CHECKING:
    from numpy.typing import NDArray

_MIN_CLASSES = 2


def _vec_dim_for(latent_dim: int) -> int:
    return latent_dim * (latent_dim + 1) // 2


class IGLReconSPDClassifier(BaseEstimator, ClassifierMixin):
    """Two-stage SPD classifier: AIRM-reconstruction encoder + sklearn readout.

    Inputs are flat log-Eig vectors (use :class:`igl.spd.LogEigVectorizer` to
    obtain them from SPD matrices). The encoder reconstructs the input via
    AIRM, then a logistic-regression head learns ``y_label`` from the
    Green-kernel design matrix Φ.

    Args:
        latent_dim: Side length ``d`` of the underlying SPD matrices —
            controls the dimensionality of the reconstruction target
            ``D = d * (d + 1) / 2``.
        max_dim: Matryoshka maximum latent dimension for the IGL encoder.
        n_anchors: Anchor count for the Green kernel.
        n_scales: Scale count for the Green kernel.
        orthogonality_weight: When ``> 0``, an
            :class:`igl.spd.OrthogonalityPenalty` is added to stage-A training.
        orthogonality_every: Frequency (in batches) of the orthogonality
            penalty.
        encoder_hidden: Encoder ``hidden`` shorthand.
        encoder_depth: Encoder depth shorthand.
        normalize_input: Whether the encoder wraps its input in a
            ``BatchNorm1d(affine=False)``. Default ``False`` — log-Eig
            tangent vectors have already been centred upstream, and a
            second input-BN would strip the per-feature scale the Green's
            kernel and the AIRM reconstruction depend on.
        normalize: Φ-normalisation mode. ``None`` (default) defers to
            ``config.kernel.normalize`` if a ``config`` is supplied, else
            the package default :data:`igl.NormalizeMode.NW`.
        validation_fraction: Fraction of the training tensor used for
            validation. The 80/20 split (the default) is performed via
            ``torch.randperm(N)`` so the RNG-consumption profile matches
            the EEG reference trainer's split bit-for-bit.
        fork_rng: When ``True`` (default), :meth:`fit` runs inside
            ``torch.random.fork_rng()`` so the caller's global torch / numpy
            RNG state is preserved. Set ``False`` to mutate the global RNG
            instead — required for bit-exact reproduction of EEG headline
            numbers that were produced with the reference trainer's bare
            ``torch.manual_seed`` discipline.
        config: Optional :class:`igl.IGLConfig`. Explicit kwargs above
            override its values.
        random_state: Optional integer seed.

    Attributes:
        classes_: Sorted unique training labels.
        n_features_in_: Length of the flat log-Eig input vector.
        module_: The IGLModule used by the encoder.
        history_: :class:`igl.TrainingHistory` from stage A.
        readout_: The fitted :class:`sklearn.linear_model.LogisticRegression`.
        dimension_curve_, effective_dimension_: Same as the Euclidean
            wrappers — measured in AIRM-curve-score space.
    """

    latent_dim: int
    max_dim: int
    n_anchors: int
    n_scales: int
    orthogonality_weight: float
    orthogonality_every: int
    encoder_hidden: int | tuple[int, ...] | None
    encoder_depth: int | None
    normalize_input: bool
    normalize: NormalizeModeLike | None
    validation_fraction: float
    fork_rng: bool
    config: IGLConfig | None
    random_state: int | None

    def __init__(  # noqa: PLR0913
        self,
        *,
        latent_dim: int,
        max_dim: int = 16,
        n_anchors: int = 64,
        n_scales: int = 4,
        orthogonality_weight: float = 0.0,
        orthogonality_every: int = 20,
        encoder_hidden: int | tuple[int, ...] | None = None,
        encoder_depth: int | None = None,
        normalize_input: bool = False,
        normalize: NormalizeModeLike | None = None,
        validation_fraction: float = 0.2,
        fork_rng: bool = True,
        config: IGLConfig | None = None,
        random_state: int | None = None,
    ) -> None:
        if latent_dim < 1:
            raise IGLConfigError(f"latent_dim must be >= 1, got {latent_dim}")
        if not 0.0 < validation_fraction < 1.0:
            raise IGLConfigError(
                f"validation_fraction must be in (0, 1), got {validation_fraction}",
            )
        self.latent_dim = latent_dim
        self.max_dim = max_dim
        self.n_anchors = n_anchors
        self.n_scales = n_scales
        self.orthogonality_weight = orthogonality_weight
        self.orthogonality_every = orthogonality_every
        self.encoder_hidden = encoder_hidden
        self.encoder_depth = encoder_depth
        self.normalize_input = normalize_input
        self.normalize = normalize
        self.validation_fraction = validation_fraction
        self.fork_rng = fork_rng
        self.config = config
        self.random_state = random_state

    def _resolve_encoder_config(self) -> EncoderConfig:
        base = self.config.encoder if self.config is not None else EncoderConfig()
        if self.encoder_hidden is None and self.encoder_depth is None:
            return base
        hidden = self.encoder_hidden if self.encoder_hidden is not None else base.hidden
        depth = self.encoder_depth if self.encoder_depth is not None else base.depth
        return EncoderConfig(
            kind=base.kind,
            hidden=hidden,
            depth=depth,
            norm=base.norm,
            activation=base.activation,
        )

    def _matryoshka_config(self) -> MatryoshkaConfig:
        """Resolve the trainer config.

        When the user passes ``config=``, that config's ``matryoshka`` block
        is honoured verbatim. When ``config`` is ``None``, a default block
        is built with:

        - ``sigma_max_diagnostic=True`` — runs the per-epoch encoder
          Jacobian diagnostic so the EEG reference trainer's RNG and BN
          buffer profile is matched.
        - ``skip_failing_batches=True`` — guards each batch's
          loss/backward/step against ``(RuntimeError, _LinAlgError)``,
          matching the reference trainer's discipline for ill-conditioned
          AIRM-on-SPD batches (Issue 3.4).
        """
        if self.config is not None:
            return self.config.matryoshka
        return MatryoshkaConfig(
            sigma_max_diagnostic=True,
            skip_failing_batches=True,
        )

    def _resolve_normalize(self) -> NormalizeModeLike:
        """Pick the Φ normaliser: explicit kwarg > config.kernel.normalize > package default."""
        if self.normalize is not None:
            return self.normalize
        if self.config is not None:
            return self.config.kernel.normalize
        return KernelConfig().normalize

    @contextmanager
    def _local_rng(self) -> Generator[None]:
        """Scope torch RNG mutations to ``.fit`` per ``self.fork_rng``.

        When ``self.fork_rng=True`` (default) the caller's global torch and
        numpy RNG state survives. When ``self.fork_rng=False`` the global
        state is mutated — required for bit-exact reproduction of EEG
        results produced with bare ``torch.manual_seed``.

        ``self.random_state=None`` short-circuits the seed call but still
        forks (or doesn't) as requested, since the caller may have already
        seeded.
        """
        if self.fork_rng:
            with torch.random.fork_rng():
                if self.random_state is not None:
                    torch.manual_seed(self.random_state)
                    np.random.seed(self.random_state)
                yield
        else:
            if self.random_state is not None:
                torch.manual_seed(self.random_state)
                np.random.seed(self.random_state)
            yield

    def fit(
        self,
        x: NDArray[np.floating],
        y: NDArray[np.generic],
        *,
        covs: torch.Tensor | None = None,
    ) -> IGLReconSPDClassifier:
        """Fit both stages.

        Args:
            x: Log-Eig vectors ``[N, d * (d + 1) / 2]``.
            y: Integer class labels ``[N]``.
            covs: Optional raw SPD targets ``[N, d, d]`` aligned with ``x``.
                When provided, AIRM uses ``covs[batch_indices] + jitter * I``
                directly as the reconstruction target, bypassing the
                log-Eig → matrix-exp round-trip and the ~1e-6 element-wise
                noise it introduces. Required for bit-exact reproduction of
                AIRM-recon results from the reference trainer.
        """
        expected_dim = _vec_dim_for(self.latent_dim)
        if x.shape[1] != expected_dim:
            raise IGLConfigError(
                f"x has {x.shape[1]} features; expected {expected_dim} for latent_dim={self.latent_dim}",
            )
        n_classes = len(np.unique(y))
        if n_classes < _MIN_CLASSES:
            raise IGLConfigError(f"need >= {_MIN_CLASSES} classes in y, got {n_classes}")
        if covs is not None and covs.shape[0] != x.shape[0]:
            raise IGLConfigError(
                f"covs has {covs.shape[0]} samples; expected {x.shape[0]} to match x",
            )

        self.n_features_in_: int = x.shape[1]
        self.classes_: NDArray[np.generic] = np.unique(y)

        x_tensor = torch.as_tensor(np.asarray(x, dtype=np.float32))
        n_samples = x_tensor.shape[0]
        extras: tuple[OrthogonalityPenalty, ...] = ()
        if self.orthogonality_weight > 0.0:
            extras = (
                OrthogonalityPenalty(
                    weight=self.orthogonality_weight,
                    every=self.orthogonality_every,
                ),
            )

        resolved_normalize = self._resolve_normalize()

        # Stage A: reconstruct the log-Eig vector via AIRM. All RNG mutations
        # (module construction, the 80/20 split, batch perms inside the
        # trainer) are scoped to this block per `self.fork_rng`.
        with self._local_rng():
            self.module_: IGLModule = IGLModule(
                input_dim=expected_dim,
                max_dim=self.max_dim,
                output_dim=expected_dim,
                n_anchors=self.n_anchors,
                n_scales=self.n_scales,
                encoder_config=self._resolve_encoder_config(),
                normalize_input=self.normalize_input,
                normalize=resolved_normalize,
            )

            # 80/20 split — `torch.randperm(N)` matches the reference
            # trainer's RNG-consumption profile (sklearn's `train_test_split`
            # uses its own RNG and would desync every downstream draw).
            n_val = max(1, int(round(n_samples * self.validation_fraction)))
            perm = torch.randperm(n_samples)
            val_idx, train_idx = perm[:n_val], perm[n_val:]
            x_train = x_tensor[train_idx]
            x_val = x_tensor[val_idx]
            covs_train = covs[train_idx] if covs is not None else None

            trainer_config = self._matryoshka_config()
            trainer = MatryoshkaTrainer(loss=AIRMLoss(latent_dim=self.latent_dim), config=trainer_config)
            # Wire the loss to the trainer so per-batch `current_batch_indices`
            # can be read by AIRMLoss(covs=...). The trainer's `loss` attribute
            # holds the strategy used during training.
            trainer.loss = AIRMLoss(
                latent_dim=self.latent_dim,
                covs=covs_train,
                trainer=trainer,
            )
            self.history_ = trainer.fit(
                self.module_,
                x_train,
                x_train,
                x_val=x_val,
                y_val=x_val,
                extra_losses=extras,
            )

        # Dimension-curve eval and Stage B operate on the FULL training
        # tensor — matching the reference's behaviour after the encoder is
        # trained.
        eval_loss = AIRMLoss(latent_dim=self.latent_dim)
        self.dimension_curve_ = eval_dimension_curve(
            self.module_,
            x_tensor,
            x_tensor,
            loss=eval_loss,
            source_l2=trainer_config.source_l2,
        )
        self.effective_dimension_: int = detect_elbow(self.dimension_curve_)

        # Stage B: fit LogisticRegression on the design matrix Φ over the
        # full input (not just the train split).
        with torch.no_grad():
            phi = self._design_matrix(x_tensor).cpu().numpy()
        self.readout_: LogisticRegression = LogisticRegression(max_iter=1000)
        self.readout_.fit(phi, np.asarray(y))
        return self

    def _design_matrix(self, x: torch.Tensor) -> torch.Tensor:
        z = self.module_.encoder(x)
        phi = self.module_.green(z)
        return normalize_phi(phi, self.module_.normalize)

    def _check_fitted(self) -> None:
        if not hasattr(self, "readout_"):
            raise IGLNotFittedError(
                "IGLReconSPDClassifier is not fitted yet; call .fit(X, y) first.",
            )

    def predict(self, x: NDArray[np.floating]) -> NDArray[np.generic]:
        """Predict class labels."""
        self._check_fitted()
        with torch.no_grad():
            phi = self._design_matrix(torch.as_tensor(np.asarray(x, dtype=np.float32))).cpu().numpy()
        return cast("NDArray[np.generic]", self.readout_.predict(phi))

    def predict_proba(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Class probabilities via the readout's :meth:`predict_proba`."""
        self._check_fitted()
        with torch.no_grad():
            phi = self._design_matrix(torch.as_tensor(np.asarray(x, dtype=np.float32))).cpu().numpy()
        return np.asarray(self.readout_.predict_proba(phi), dtype=np.float64)


__all__ = ["IGLReconSPDClassifier"]
