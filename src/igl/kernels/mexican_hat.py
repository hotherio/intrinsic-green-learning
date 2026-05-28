"""Mexican-hat (Ricker) kernel: ``(1 - d²/σ²) · exp(-d² / (2σ²))`` (oscillatory)."""

import torch

from igl.kernels._constants import KERNEL_EPS
from igl.kernels._registry import register_operator


class _MexicanHat:
    is_oscillatory: bool = True

    def __call__(self, d: torch.Tensor, sigma: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor]:
        ratio = d**2 / (sigma**2 + KERNEL_EPS)
        factor = 1 - ratio
        log_abs = torch.log(factor.abs().clamp(min=KERNEL_EPS)) + (-ratio / 2)
        sign = torch.where(factor >= 0, torch.ones_like(d), -torch.ones_like(d))
        return log_abs, sign


mexican_hat = _MexicanHat()
register_operator("mexican_hat", mexican_hat, is_oscillatory=mexican_hat.is_oscillatory)

__all__ = ["mexican_hat"]
