"""Cauchy Green's-function kernel: ``1 / (1 + d²/σ²)``."""

import torch

from igl.kernels._constants import KERNEL_EPS
from igl.kernels._registry import register_operator


class _Cauchy:
    is_oscillatory: bool = False

    def __call__(self, d: torch.Tensor, sigma: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor]:
        log_abs = -torch.log1p(d**2 / (sigma**2 + KERNEL_EPS))
        return log_abs, torch.ones_like(d)


cauchy = _Cauchy()
register_operator("cauchy", cauchy, is_oscillatory=cauchy.is_oscillatory)

__all__ = ["cauchy"]
