"""Metric-weighted state distillation as a sklearn-style estimator."""

import numpy as np
import torch
from numpy.typing import NDArray

from igl.config import IGLConfig
from igl.exceptions import IGLConfigError
from igl.models._base import _BaseIGLEstimator, _check_is_fitted, _to_torch
from igl.whitening.loss import WhitenedMSELoss
from igl.whitening.whitener import TargetWhitener

__all__ = ["IGLDistiller"]

_EXPECTED_X_NDIM = 2


class IGLDistiller(_BaseIGLEstimator[WhitenedMSELoss]):
    """Distill states into a Matryoshka latent under a metric geometry.

    The pipeline mirrors metric-weighted distillation: inputs are centered
    per feature and scaled by a single scalar std; the reconstruction target
    is the raw state, whitened inside the loss by ``metric``'s square root
    so plain least squares optimizes the second-order expansion of the
    geometry the metric encodes. :meth:`reconstruct` undoes the whitening,
    returning states in the original space.

    Args:
        max_dim: Matryoshka latent width.
        metric: Symmetric PSD metric ``[C, C]`` over the state space (e.g.
            :func:`igl.whitening.fisher_pullback`), or ``None`` for the
            identity metric (variance-faithful distillation).
        clamp: Eigenvalue floor for the metric square root.
        config: Optional :class:`igl.IGLConfig`.
        device: Optional torch device for fitting.
        random_state: Seed scoping module construction and training.
        validation_fraction: Held-out fraction for early stopping.

    Attributes:
        module_: The fitted :class:`igl.IGLModule` (post-fit).
        history_: Training history (post-fit).
        dimension_curve_: Loss-versus-dimension curve in the whitened
            geometry (post-fit).
        effective_dimension_: Elbow of the dimension curve (post-fit).
        whitener_: The fitted :class:`igl.whitening.TargetWhitener`
            (post-fit).
        input_mean_: Per-feature input mean ``[C]`` (post-fit).
        input_std_: Scalar input std (post-fit).
    """

    metric: torch.Tensor | None
    clamp: float
    validation_fraction: float | None

    def __init__(
        self,
        *,
        max_dim: int = 16,
        metric: torch.Tensor | None = None,
        clamp: float = 1e-6,
        config: IGLConfig | None = None,
        device: str | torch.device | None = None,
        random_state: int | None = None,
        validation_fraction: float | None = 0.2,
    ) -> None:
        super().__init__(max_dim=max_dim, config=config, random_state=random_state, device=device)
        self.metric = metric
        self.clamp = clamp
        self.validation_fraction = validation_fraction

    def fit(self, x: NDArray[np.floating], y: None = None) -> "IGLDistiller":  # noqa: ARG002
        """Fit the distiller on states ``[N, C]`` (the states are the target).

        Args:
            x: States to distill.
            y: Ignored; present for sklearn API compatibility.

        Returns:
            ``self``, for chaining.
        """
        if x.ndim != _EXPECTED_X_NDIM:
            x = np.asarray(x).reshape(len(x), -1)
        n_features = x.shape[1]
        self.n_features_in_: int = n_features

        device = torch.device(self.device) if self.device is not None else torch.device("cpu")
        raw = _to_torch(np.asarray(x, dtype=np.float32), device=torch.device("cpu"))
        self.input_mean_: torch.Tensor = raw.mean(dim=0)
        self.input_std_: float = max(float(raw.std()), torch.finfo(torch.float32).tiny)

        self.whitener_: TargetWhitener = TargetWhitener(self.metric, clamp=self.clamp).fit(raw)
        loss = WhitenedMSELoss(self.whitener_)

        x_scaled = (raw - self.input_mean_) / self.input_std_
        self._fit_tensors(
            x_scaled.to(device),
            raw.to(device),
            loss=loss,
            output_dim=n_features,
            validation_fraction=self.validation_fraction,
            device=device,
        )
        return self

    def project(self, x: NDArray[np.floating], *, k: int | None = None) -> NDArray[np.floating]:
        """Map states to chart coordinates, optionally truncated to ``k``.

        Args:
            x: States ``[N, C]``.
            k: Keep only the first ``k`` Matryoshka coordinates; ``None``
                keeps all ``max_dim``.

        Returns:
            Coordinates ``[N, k or max_dim]``.
        """
        _check_is_fitted(self, "module_")
        k = self._check_k(k)
        device = next(self.module_.parameters()).device
        x_scaled = self._scale_inputs(x, device)
        with torch.no_grad():
            latent = self.module_.latent(x_scaled)
        return latent[:, :k].cpu().numpy()

    def reconstruct(self, x: NDArray[np.floating], *, k: int | None = None) -> NDArray[np.floating]:
        """Reconstruct states through the bottleneck, back in the original space.

        Args:
            x: States ``[N, C]``.
            k: Read the reconstruction from only the first ``k`` coordinates
                (a tight-budget read-out); ``None`` uses all ``max_dim``.

        Returns:
            Reconstructed states ``[N, C]``.
        """
        _check_is_fitted(self, "module_")
        k = self._check_k(k)
        device = next(self.module_.parameters()).device
        x_scaled = self._scale_inputs(x, device)
        gate_mask = None
        if k < self.module_.max_dim:
            gate_mask = torch.zeros(self.module_.max_dim, device=device)
            gate_mask[:k] = 1.0
        with torch.no_grad():
            whitened = self.module_(x_scaled, gate_mask=gate_mask)
        return self.whitener_.inverse_transform(whitened.cpu()).numpy()

    def _scale_inputs(self, x: NDArray[np.floating], device: torch.device) -> torch.Tensor:
        raw = _to_torch(np.asarray(x, dtype=np.float32).reshape(len(x), -1), device=torch.device("cpu"))
        return _to_torch(((raw - self.input_mean_) / self.input_std_).numpy(), device=device)

    def _check_k(self, k: int | None) -> int:
        if k is None:
            return self.module_.max_dim
        if not 1 <= k <= self.module_.max_dim:
            raise IGLConfigError(f"k must be in [1, {self.module_.max_dim}], got {k}")
        return k
