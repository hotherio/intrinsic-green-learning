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

import dataclasses
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from igl.config import EncoderConfig, IGLConfig, KernelConfig, MatryoshkaConfig
from igl.core.loss import MSELoss
from igl.core.normalization import normalize_phi
from igl.core.trainer import MatryoshkaTrainer
from igl.exceptions import IGLConfigError, IGLNotFittedError
from igl.matryoshka.dimension_curve import detect_elbow, eval_dimension_curve
from igl.nn.module import IGLModule
from igl.spd.airm import AIRMLoss
from igl.spd.log_eig import LogEigVectorizer
from igl.spd.orthogonality import OrthogonalityPenalty
from igl.spd.preconditioning import precondition
from igl.types import NormalizeModeLike, PreconditionMode, PreconditionModeLike

if TYPE_CHECKING:
    from numpy.typing import NDArray

_MIN_CLASSES = 2
_SPD_NDIM = 3


def _iglconfig_default_max_dim() -> int:
    """``IGLConfig.max_dim``'s declared default, read off the dataclass.

    Used to tell an untouched ``max_dim`` apart from one the caller set
    deliberately. Derived rather than hardcoded so it cannot drift out of
    sync with :class:`igl.IGLConfig`.
    """
    field = next(f for f in dataclasses.fields(IGLConfig) if f.name == "max_dim")
    return cast("int", field.default)


def _vec_dim_for(latent_dim: int) -> int:
    return latent_dim * (latent_dim + 1) // 2


def _looks_like_spd_batch(x: NDArray[np.floating]) -> bool:
    """``True`` iff ``x`` has shape ``[N, d, d]`` (square last two axes).

    Lets :meth:`IGLReconSPDClassifier.fit` / :meth:`.predict` accept either
    log-Eig vectors (the original 2-D API) or raw SPD matrices (so the
    sklearn pipeline built by :func:`igl.make_igl_airm` works without an
    explicit :class:`LogEigVectorizer` step). The square-shape check is
    sufficient: the input pipeline never returns ``[N, d1, d2]`` with
    ``d1 != d2``.
    """
    return x.ndim == _SPD_NDIM and x.shape[-1] == x.shape[-2]


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
            When a ``config`` is supplied, its ``max_dim`` must either match
            this value or be left at its default (which then inherits it).
        n_anchors: Anchor count for the Green kernel. ``None`` (default)
            defers to ``config.kernel.n_anchors`` (default ``64``).
        n_scales: Scale count for the Green kernel. ``None`` (default)
            defers to ``config.kernel.n_scales`` (default ``4``).
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
        precondition: SPD-side preconditioning applied to every input
            covariance before AIRM. Default ``"tikhonov"``: bit-identical
            to ``"none"`` at ``d ≤ 64`` (the encoder BatchNorm absorbs the
            constant offset) and rescues ``torch.linalg.eigh`` from LAPACK
            error 8481 at ``d ≥ 128``. See
            :class:`igl.PreconditionMode` for the full mode catalogue and
            ``alex-eeg-igl/MAINTAINER_MEMO_lwf_tikh_rules.md`` for the
            empirical justification.
        precondition_epsilon: Ridge magnitude for the Tikhonov branches
            of ``precondition``. Default ``1e-6`` is the value
            characterised in the memo as universally safe across the
            tested datasets.
        config: Optional :class:`igl.IGLConfig`. Its ``kernel`` block is
            forwarded to the Green kernel, so ``null_space``,
            ``polynomial_degree``, ``sigma_log_range`` and
            ``anchor_init_std`` all take effect here. Explicit kwargs above
            override its values. ``config.max_dim`` must either match
            ``max_dim`` or be left at its default.
        random_state: Optional integer seed.

    Attributes:
        classes_: Sorted unique training labels.
        n_features_in_: Length of the flat log-Eig input vector.
        module_: The IGLModule used by the encoder.
        history_: :class:`igl.TrainingHistory` from stage A.
        readout_: The fitted :class:`sklearn.linear_model.LogisticRegression`.
        dimension_curve_, effective_dimension_: Same as the Euclidean
            wrappers — measured in AIRM-curve-score space.
        precondition_mode_: The resolved :class:`PreconditionMode` actually
            applied at fit time (round-trips through pickle).
        precondition_epsilon_: The ridge value actually used at fit time.
    """

    latent_dim: int
    max_dim: int
    n_anchors: int | None
    n_scales: int | None
    orthogonality_weight: float
    orthogonality_every: int
    encoder_hidden: int | tuple[int, ...] | None
    encoder_depth: int | None
    normalize_input: bool
    normalize: NormalizeModeLike | None
    validation_fraction: float
    fork_rng: bool
    precondition: PreconditionModeLike
    precondition_epsilon: float
    config: IGLConfig | None
    random_state: int | None
    device: str
    recon_loss: str

    def __init__(  # noqa: PLR0913
        self,
        *,
        latent_dim: int,
        max_dim: int = 16,
        n_anchors: int | None = None,
        n_scales: int | None = None,
        orthogonality_weight: float = 0.0,
        orthogonality_every: int = 20,
        encoder_hidden: int | tuple[int, ...] | None = None,
        encoder_depth: int | None = None,
        normalize_input: bool = False,
        normalize: NormalizeModeLike | None = None,
        validation_fraction: float = 0.2,
        fork_rng: bool = True,
        precondition: PreconditionModeLike = PreconditionMode.TIKHONOV,
        precondition_epsilon: float = 1e-6,
        config: IGLConfig | None = None,
        random_state: int | None = None,
        device: str = "cpu",
        recon_loss: str = "airm",
    ) -> None:
        if latent_dim < 1:
            raise IGLConfigError(f"latent_dim must be >= 1, got {latent_dim}")
        if not 0.0 < validation_fraction < 1.0:
            raise IGLConfigError(
                f"validation_fraction must be in (0, 1), got {validation_fraction}",
            )
        if precondition_epsilon < 0:
            raise IGLConfigError(
                f"precondition_epsilon must be >= 0, got {precondition_epsilon}",
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
        self.precondition = precondition
        self.precondition_epsilon = precondition_epsilon
        self.config = config
        self.random_state = random_state
        self.device = device
        if recon_loss not in ("airm", "log_euclidean"):
            raise IGLConfigError(
                f"recon_loss must be 'airm' or 'log_euclidean', got {recon_loss!r}",
            )
        self.recon_loss = recon_loss

    def _precondition(self, c: torch.Tensor) -> torch.Tensor:
        """Apply the configured SPD preconditioning to a ``covs`` tensor."""
        return precondition(c, mode=self.precondition, epsilon=self.precondition_epsilon)

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

    def _resolve_module_config(self) -> IGLConfig | None:
        """Align ``config.max_dim`` with this estimator's ``max_dim``.

        :class:`IGLModule` cross-validates the two and raises when they
        disagree. Callers routinely leave ``IGLConfig.max_dim`` at its
        default while setting the estimator's ``max_dim``, so an untouched
        default is realigned silently. A *deliberately* set value that
        contradicts ``max_dim`` is a user error and raises.

        Returning the config (rather than ``None``) is what lets
        ``config.kernel``'s ``null_space``, ``polynomial_degree``,
        ``sigma_log_range`` and ``anchor_init_std`` reach the Green kernel.
        """
        if self.config is None:
            return None
        if self.config.max_dim not in (self.max_dim, _iglconfig_default_max_dim()):
            raise IGLConfigError(
                f"config.max_dim ({self.config.max_dim}) conflicts with max_dim ({self.max_dim}); "
                f"leave config.max_dim at its default to inherit max_dim",
            )
        return dataclasses.replace(self.config, max_dim=self.max_dim)

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
            x: Either log-Eig vectors ``[N, d * (d + 1) / 2]`` (the
                original API) or raw SPD matrices ``[N, d, d]``. When ``x``
                is SPD-shaped, it is vectorized internally via
                :class:`igl.spd.LogEigVectorizer` and *also* reused as the
                AIRM target (``covs`` defaults to ``x`` in that case).
                This makes the wrapper drop-in compatible with sklearn
                pipelines that emit SPDs upstream (e.g.
                :func:`igl.make_igl_airm`).
            y: Integer class labels ``[N]``.
            covs: Optional raw SPD targets ``[N, d, d]`` aligned with ``x``.
                When provided, AIRM uses ``covs[batch_indices] + jitter * I``
                directly as the reconstruction target, bypassing the
                log-Eig → matrix-exp round-trip and the ~1e-6 element-wise
                noise it introduces. Required for bit-exact reproduction of
                AIRM-recon results from the reference trainer. Ignored
                when ``x`` is SPD-shaped (then ``x`` itself is the target).
        """
        if _looks_like_spd_batch(x):
            # Sklearn-pipeline path: vectorize SPDs internally and reuse
            # them as the AIRM target. Cache the fitted vectorizer so
            # ``.predict`` can re-apply the exact same √2 scaling.
            self.log_eig_: LogEigVectorizer = LogEigVectorizer().fit(x)
            if covs is None:
                # float32 matches the encoder/AIRM contract; the existing
                # explicit-covs API uses float32 too (see test fixtures).
                covs = torch.as_tensor(np.asarray(x, dtype=np.float32))
            x = self.log_eig_.transform(x)

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

        # Resolve and freeze the precondition contract so it round-trips
        # through pickle (sklearn convention — fitted-state attributes end
        # in an underscore).
        self.precondition_mode_: PreconditionMode = PreconditionMode(self.precondition)
        self.precondition_epsilon_: float = self.precondition_epsilon

        x_tensor = torch.as_tensor(np.asarray(x, dtype=np.float32))
        n_samples = x_tensor.shape[0]

        # Apply SPD-side preconditioning once, up front. Affects the AIRM
        # target only — the predicted side still flows through the
        # encoder/Green-kernel/log-Eig pipeline unchanged.
        if covs is not None:
            covs = self._precondition(covs)
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
                config=self._resolve_module_config(),
            )
            # Move to the target device AFTER construction so the parameter
            # init (which consumes torch RNG inside ``IGLModule``) stays on CPU
            # and bit-identical across devices — only the storage moves. The
            # trainer reads ``next(module.parameters()).device`` and relocates
            # ``x_train``/``x_val`` accordingly; ``AIRMLoss`` relocates ``covs``
            # on first batch; ``eval_dimension_curve``/``_design_matrix`` move
            # their inputs to the module device.
            self.module_ = self.module_.to(torch.device(self.device))

            # 80/20 split — ``torch.randperm(N)`` matches the reference
            # trainer's RNG-consumption profile (sklearn's ``train_test_split``
            # uses its own RNG and would desync every downstream draw).
            # ``int(N * f)`` **truncates** (matching the reference at
            # ``igl_recon_spd_orth.py:494``); ``int(round(...))`` would
            # off-by-one whenever ``N * f`` is non-integer and flip every
            # subsequent RNG draw (Issue 4 in the EEG reproducibility chain).
            n_val = max(1, int(n_samples * self.validation_fraction))
            perm = torch.randperm(n_samples)
            val_idx, train_idx = perm[:n_val], perm[n_val:]
            x_train = x_tensor[train_idx]
            x_val = x_tensor[val_idx]
            covs_train = covs[train_idx] if covs is not None else None

            trainer_config = self._matryoshka_config()
            trainer = MatryoshkaTrainer(loss=AIRMLoss(latent_dim=self.latent_dim), config=trainer_config)
            if self.recon_loss == "log_euclidean":
                # Log-Euclidean reconstruction: ||pred - LogEig(C)||^2. Because the
                # √2-scaled LogEig map is a Frobenius isometry, MSE on the log-Eig
                # vectors *is* the squared Log-Euclidean distance — and it needs ZERO
                # eigendecompositions per step (no matrix_exp/log/pow). A faster but
                # non-affine-invariant surrogate for AIRM; quantify score parity
                # before using for headline results.
                trainer.loss = MSELoss()
            else:
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
        eval_loss = MSELoss() if self.recon_loss == "log_euclidean" else AIRMLoss(latent_dim=self.latent_dim)
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
        # Relocate the input to wherever the module lives (a no-op on CPU, so
        # the CPU path stays bit-identical). Callers `.cpu()` the result.
        x = x.to(next(self.module_.parameters()).device)
        z = self.module_.encoder(x)
        phi = self.module_.green(z)
        return normalize_phi(phi, self.module_.normalize)

    def _check_fitted(self) -> None:
        if not hasattr(self, "readout_"):
            raise IGLNotFittedError(
                "IGLReconSPDClassifier is not fitted yet; call .fit(X, y) first.",
            )

    def _vectorize_if_spd(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Mirror ``.fit``'s SPD-input handling on the predict path.

        Re-uses the :class:`LogEigVectorizer` fitted in ``.fit`` so the √2
        upper-triangle scaling matches bit-exactly.
        """
        if _looks_like_spd_batch(x):
            if not hasattr(self, "log_eig_"):
                raise IGLConfigError(
                    "x is SPD-shaped but the wrapper was fitted on log-Eig "
                    "vectors. Either pass log-Eig vectors here or refit with "
                    "SPD-shaped inputs.",
                )
            return self.log_eig_.transform(x)
        return x

    def predict(self, x: NDArray[np.floating]) -> NDArray[np.generic]:
        """Predict class labels. Accepts log-Eig vectors or raw SPDs."""
        self._check_fitted()
        x_vec = self._vectorize_if_spd(x)
        with torch.no_grad():
            phi = self._design_matrix(torch.as_tensor(np.asarray(x_vec, dtype=np.float32))).cpu().numpy()
        return cast("NDArray[np.generic]", self.readout_.predict(phi))

    def predict_proba(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Class probabilities via the readout's :meth:`predict_proba`."""
        self._check_fitted()
        x_vec = self._vectorize_if_spd(x)
        with torch.no_grad():
            phi = self._design_matrix(torch.as_tensor(np.asarray(x_vec, dtype=np.float32))).cpu().numpy()
        return np.asarray(self.readout_.predict_proba(phi), dtype=np.float64)


__all__ = ["IGLReconSPDClassifier"]
