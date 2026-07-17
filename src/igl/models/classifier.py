"""scikit-learn-compatible classifier wrapper around :class:`igl.IGLModule`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from sklearn.base import ClassifierMixin

from igl.config import IGLConfig
from igl.core.loss import CrossEntropyLoss
from igl.exceptions import IGLConfigError
from igl.models._base import _BaseIGLEstimator
from igl.types import NormalizeModeLike, OperatorNameLike

if TYPE_CHECKING:
    from numpy.typing import NDArray


_MIN_CLASSES = 2


class IGLClassifier(_BaseIGLEstimator[CrossEntropyLoss], ClassifierMixin):
    """scikit-learn-compatible classifier built on :class:`igl.IGLModule`.

    Args:
        max_dim: Latent dimension ``d_max`` for Matryoshka truncation.
        n_anchors: Anchor count. ``None`` defers to ``config`` or default 64.
        n_scales: Scale count. ``None`` defers to ``config`` or 4.
        operator: Kernel operator. ``None`` defers to ``config`` or
            ``OperatorName.GAUSSIAN``.
        normalize: Φ-normalization mode.
        encoder_hidden: Encoder ``hidden`` shorthand
            (``int`` for uniform width, tuple for per-layer widths).
        encoder_depth: Encoder depth shorthand.
        config: Optional :class:`igl.IGLConfig`. Explicit kwargs override it.
        random_state: Optional integer seed for reproducible training.
        validation_fraction: If set, hold out this fraction of training data
            for early-stopping and dimension-curve evaluation.

    Attributes:
        classes_: Sorted unique training labels.
        n_features_in_: Ambient input dimension seen during fit.
        module_: The underlying :class:`igl.IGLModule`.
        history_: :class:`igl.TrainingHistory` from the trainer.
        dimension_curve_: ``{k: error_rate}`` from
            :func:`igl.eval_dimension_curve` post-fit.
        effective_dimension_: Detected elbow ``k`` (the discovered ``d_eff``).
        scaler_: :class:`sklearn.preprocessing.StandardScaler` used on inputs.
    """

    validation_fraction: float | None

    def __init__(
        self,
        *,
        max_dim: int = 16,
        n_anchors: int | None = None,
        n_scales: int | None = None,
        operator: OperatorNameLike | None = None,
        normalize: NormalizeModeLike | None = None,
        encoder_hidden: int | tuple[int, ...] | None = None,
        encoder_depth: int | None = None,
        config: IGLConfig | None = None,
        random_state: int | None = None,
        device: str | torch.device | None = None,
        validation_fraction: float | None = 0.2,
    ) -> None:
        super().__init__(
            max_dim=max_dim,
            n_anchors=n_anchors,
            n_scales=n_scales,
            operator=operator,
            normalize=normalize,
            encoder_hidden=encoder_hidden,
            encoder_depth=encoder_depth,
            config=config,
            random_state=random_state,
            device=device,
        )
        self.validation_fraction = validation_fraction

    def _build_loss(self, y: NDArray[np.generic]) -> CrossEntropyLoss:
        n_classes = len(np.unique(y))
        if n_classes < _MIN_CLASSES:
            raise IGLConfigError(f"need >= {_MIN_CLASSES} classes in y, got {n_classes}")
        return CrossEntropyLoss(n_classes=n_classes)

    def _output_dim(self, y: NDArray[np.generic]) -> int:
        return len(np.unique(y))

    def _prepare_y(
        self,
        y: NDArray[np.generic],
        *,
        device: torch.device,
        x_scaled: NDArray[np.floating] | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        # Encode labels to a contiguous range [0, n_classes) for the trainer.
        self.classes_: NDArray[np.generic] = np.unique(y)
        encoded = np.searchsorted(self.classes_, y).astype(np.int64)
        return torch.as_tensor(encoded, dtype=torch.long, device=device)

    def fit(self, x: NDArray[np.floating], y: NDArray[np.generic]) -> IGLClassifier:
        """Fit the classifier on ``(x, y)``."""
        self._fit_core(np.asarray(x), np.asarray(y), validation_fraction=self.validation_fraction)
        return self

    def predict(self, x: NDArray[np.floating]) -> NDArray[np.generic]:
        """Predict class labels for ``x``."""
        logits = self._predict_phi(np.asarray(x))
        idx = logits.argmax(dim=-1).cpu().numpy()
        return self.classes_[idx]

    def predict_proba(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Predict class probabilities for ``x``."""
        logits = self._predict_phi(np.asarray(x))
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        return np.asarray(probs, dtype=np.float64)

    # `score` is inherited from ``sklearn.base.ClassifierMixin`` — returns
    # mean accuracy. We don't override it to keep sample_weight handling free.


__all__ = ["IGLClassifier"]
