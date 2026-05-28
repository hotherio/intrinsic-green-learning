"""Laplacian Green's-function kernel: ``exp(-|d| / σ)``."""

import torch

from igl.kernels._constants import KERNEL_EPS
from igl.kernels._registry import register_operator


class _Laplacian:
    is_oscillatory: bool = False

    def __call__(self, d: torch.Tensor, sigma: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor]:
        log_abs = -torch.abs(d) / (sigma + KERNEL_EPS)
        return log_abs, torch.ones_like(d)


laplacian = _Laplacian()
register_operator("laplacian", laplacian, is_oscillatory=laplacian.is_oscillatory)

__all__ = ["laplacian"]
