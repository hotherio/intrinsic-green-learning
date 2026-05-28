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

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from igl.config import EncoderConfig, IGLConfig, MatryoshkaConfig
from igl.core.normalization import normalize_phi
from igl.core.trainer import MatryoshkaTrainer
from igl.exceptions import IGLConfigError, IGLNotFittedError
from igl.matryoshka.dimension_curve import detect_elbow, eval_dimension_curve
from igl.nn.module import IGLModule
from igl.spd.airm import AIRMLoss
from igl.spd.orthogonality import OrthogonalityPenalty

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
    config: IGLConfig | None
    random_state: int | None

    def __init__(
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
        config: IGLConfig | None = None,
        random_state: int | None = None,
    ) -> None:
        if latent_dim < 1:
            raise IGLConfigError(f"latent_dim must be >= 1, got {latent_dim}")
        self.latent_dim = latent_dim
        self.max_dim = max_dim
        self.n_anchors = n_anchors
        self.n_scales = n_scales
        self.orthogonality_weight = orthogonality_weight
        self.orthogonality_every = orthogonality_every
        self.encoder_hidden = encoder_hidden
        self.encoder_depth = encoder_depth
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
        return self.config.matryoshka if self.config is not None else MatryoshkaConfig()

    def fit(self, x: NDArray[np.floating], y: NDArray[np.generic]) -> IGLReconSPDClassifier:
        """Fit both stages.

        Args:
            x: Log-Eig vectors ``[N, d * (d + 1) / 2]``.
            y: Integer class labels ``[N]``.
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)  # pyright: ignore[reportUnknownMemberType]
            np.random.seed(self.random_state)

        expected_dim = _vec_dim_for(self.latent_dim)
        if x.shape[1] != expected_dim:
            raise IGLConfigError(
                f"x has {x.shape[1]} features; expected {expected_dim} for latent_dim={self.latent_dim}",
            )
        n_classes = len(np.unique(y))
        if n_classes < _MIN_CLASSES:
            raise IGLConfigError(f"need >= {_MIN_CLASSES} classes in y, got {n_classes}")

        self.n_features_in_: int = x.shape[1]
        self.classes_: NDArray[np.generic] = np.unique(y)

        x_tensor = torch.as_tensor(np.asarray(x, dtype=np.float32))
        # Stage A: reconstruct the log-Eig vector via AIRM.
        self.module_: IGLModule = IGLModule(
            input_dim=expected_dim,
            max_dim=self.max_dim,
            output_dim=expected_dim,
            n_anchors=self.n_anchors,
            n_scales=self.n_scales,
            encoder_config=self._resolve_encoder_config(),
        )
        loss = AIRMLoss(latent_dim=self.latent_dim)
        trainer = MatryoshkaTrainer(loss=loss, config=self._matryoshka_config())

        extras = ()
        if self.orthogonality_weight > 0.0:
            extras = (
                OrthogonalityPenalty(
                    weight=self.orthogonality_weight,
                    every=self.orthogonality_every,
                ),
            )

        self.history_ = trainer.fit(self.module_, x_tensor, x_tensor, extra_losses=extras)

        self.dimension_curve_ = eval_dimension_curve(
            self.module_,
            x_tensor,
            x_tensor,
            loss=loss,
            source_l2=self._matryoshka_config().source_l2,
        )
        self.effective_dimension_: int = detect_elbow(self.dimension_curve_)

        # Stage B: fit LogisticRegression on the design matrix Φ.
        with torch.no_grad():
            phi = self._design_matrix(x_tensor).cpu().numpy()
        self.readout_: LogisticRegression = LogisticRegression(max_iter=1000)
        self.readout_.fit(phi, np.asarray(y))  # pyright: ignore[reportUnknownMemberType]
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
        return cast("NDArray[np.generic]", self.readout_.predict(phi))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

    def predict_proba(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Class probabilities via the readout's :meth:`predict_proba`."""
        self._check_fitted()
        with torch.no_grad():
            phi = self._design_matrix(torch.as_tensor(np.asarray(x, dtype=np.float32))).cpu().numpy()
        return np.asarray(self.readout_.predict_proba(phi), dtype=np.float64)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


__all__ = ["IGLReconSPDClassifier"]
