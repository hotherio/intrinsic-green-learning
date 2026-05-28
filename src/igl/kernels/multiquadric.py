"""Inverse multiquadric kernel: ``1 / sqrt(1 + d²/σ²)``."""

import torch

from igl.kernels._constants import KERNEL_EPS
from igl.kernels._registry import register_operator


class _Multiquadric:
    is_oscillatory: bool = False

    def __call__(self, d: torch.Tensor, sigma: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor]:
        log_abs = -0.5 * torch.log1p(d**2 / (sigma**2 + KERNEL_EPS))
        return log_abs, torch.ones_like(d)


multiquadric = _Multiquadric()
register_operator("multiquadric", multiquadric, is_oscillatory=multiquadric.is_oscillatory)

__all__ = ["multiquadric"]
