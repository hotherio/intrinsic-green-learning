"""Gaussian Green's-function kernel: ``exp(-d² / (2σ²))``."""

import torch

from igl.kernels._constants import KERNEL_EPS
from igl.kernels._registry import register_operator


class _Gaussian:
    is_oscillatory: bool = False

    def __call__(self, d: torch.Tensor, sigma: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor]:
        log_abs = -(d**2) / (2 * sigma**2 + KERNEL_EPS)
        return log_abs, torch.ones_like(d)


def _squared_distance(d: torch.Tensor) -> torch.Tensor:
    return d * d


def _inverse_two_sigma_squared(sigma: torch.Tensor) -> torch.Tensor:
    return 1.0 / (2 * sigma**2 + KERNEL_EPS)


gaussian = _Gaussian()
register_operator(
    "gaussian",
    gaussian,
    is_oscillatory=gaussian.is_oscillatory,
    separable=(_squared_distance, _inverse_two_sigma_squared),
)

__all__ = ["gaussian"]
