"""Target whitening by a metric square root."""

from collections.abc import Mapping
from typing import Self

import torch

from igl.exceptions import IGLConfigError, IGLNotFittedError
from igl.whitening.linalg import psd_sqrt_inv

__all__ = ["TargetWhitener"]

_MATRIX_NDIM = 2
_STATE_KEYS = frozenset({"a", "a_inv", "mu", "y_scale", "clamp"})


class TargetWhitener:
    """Whiten targets by a metric square root: ``y_w = ((y - mu) @ a) / y_scale``.

    Fitting computes the mean of ``y``, the symmetric square-root pair of the
    metric, and a scalar RMS scale so whitened targets are unit-scale for the
    trainer. ``inverse_transform`` undoes all three. With ``metric=None`` the
    whitener degrades to centering plus unit scaling.

    Args:
        metric: Symmetric PSD metric ``[C, C]`` over the target space, or
            ``None`` for the identity metric.
        clamp: Eigenvalue floor forwarded to
            :func:`igl.whitening.psd_sqrt_inv`.

    Attributes:
        mu_: Target mean ``[C]`` (post-fit).
        a_: Metric square root ``[C, C]`` (post-fit).
        a_inv_: Inverse square root ``[C, C]`` (post-fit).
        y_scale_: Scalar RMS of the centered, whitened targets (post-fit).
    """

    metric: torch.Tensor | None
    clamp: float

    def __init__(self, metric: torch.Tensor | None = None, *, clamp: float = 1e-6) -> None:
        self.metric = metric
        self.clamp = clamp

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has run."""
        return hasattr(self, "y_scale_")

    def fit(self, y: torch.Tensor) -> Self:
        """Fit the whitening constants on targets ``[N, C]``.

        Args:
            y: Targets ``[N, C]``.

        Returns:
            ``self``, for chaining.
        """
        if y.dim() != _MATRIX_NDIM:
            raise IGLConfigError(f"y must be 2-D [N, C], got shape {tuple(y.shape)}")
        y = y.detach().float()
        n_cols = y.shape[1]
        metric = self.metric if self.metric is not None else torch.eye(n_cols)
        if metric.shape != (n_cols, n_cols):
            raise IGLConfigError(f"metric shape {tuple(metric.shape)} does not match target width {n_cols}")
        self.mu_: torch.Tensor = y.mean(dim=0)
        self.a_: torch.Tensor
        self.a_inv_: torch.Tensor
        self.a_, self.a_inv_ = psd_sqrt_inv(metric, clamp=self.clamp)
        whitened = (y - self.mu_) @ self.a_
        self.y_scale_: float = max(float(whitened.std()), torch.finfo(torch.float32).tiny)
        return self

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        """Whiten targets: center, rotate-and-scale by ``a_``, unit-scale."""
        self._check_fitted()
        return ((y.float() - self.mu_) @ self.a_) / self.y_scale_

    def inverse_transform(self, y_w: torch.Tensor) -> torch.Tensor:
        """Undo :meth:`transform`."""
        self._check_fitted()
        return (y_w.float() * self.y_scale_) @ self.a_inv_ + self.mu_

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Serialize the fitted constants as a flat tensor dict."""
        self._check_fitted()
        return {
            "a": self.a_.clone(),
            "a_inv": self.a_inv_.clone(),
            "mu": self.mu_.clone(),
            "y_scale": torch.tensor(self.y_scale_),
            "clamp": torch.tensor(self.clamp),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, torch.Tensor]) -> Self:
        """Rebuild a fitted whitener from :meth:`state_dict` output."""
        missing = _STATE_KEYS - state.keys()
        if missing:
            raise IGLConfigError(f"whitener state is missing keys: {sorted(missing)}")
        whitener = cls(clamp=float(state["clamp"].item()))
        whitener.a_ = state["a"].clone()  # type: ignore[reportUninitializedInstanceVariable]
        whitener.a_inv_ = state["a_inv"].clone()
        whitener.mu_ = state["mu"].clone()
        whitener.y_scale_ = float(state["y_scale"].item())
        return whitener

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise IGLNotFittedError("TargetWhitener is not fitted; call fit() first")
