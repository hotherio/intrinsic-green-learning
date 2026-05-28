"""Gabor kernel: ``exp(-d² / (2σ²)) · cos(π · d / σ)`` (oscillatory Gaussian envelope)."""

import math

import torch

from igl.kernels._constants import KERNEL_EPS
from igl.kernels._registry import register_operator


class _Gabor:
    is_oscillatory: bool = True

    def __call__(self, d: torch.Tensor, sigma: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor]:
        cos_val = torch.cos(math.pi * d / (sigma + KERNEL_EPS))
        log_abs = -(d**2) / (2 * sigma**2 + KERNEL_EPS) + torch.log(cos_val.abs().clamp(min=KERNEL_EPS))
        sign = torch.where(cos_val >= 0, torch.ones_like(d), -torch.ones_like(d))
        return log_abs, sign


gabor = _Gabor()
register_operator("gabor", gabor, is_oscillatory=gabor.is_oscillatory)

__all__ = ["gabor"]
