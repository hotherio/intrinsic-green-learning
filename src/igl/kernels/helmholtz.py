"""Helmholtz Green's-function kernel: ``exp(-|d|/σ) · cos(π · d / σ)`` (oscillatory)."""

import math

import torch

from igl.kernels._registry import register_operator

_EPS = 1e-8


class _Helmholtz:
    is_oscillatory: bool = True

    def __call__(self, d: torch.Tensor, sigma: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor]:
        cos_val = torch.cos(math.pi * d / (sigma + _EPS))
        log_abs = -torch.abs(d) / (sigma + _EPS) + torch.log(cos_val.abs().clamp(min=_EPS))
        sign = torch.where(cos_val >= 0, torch.ones_like(d), -torch.ones_like(d))
        return log_abs, sign


helmholtz = _Helmholtz()
register_operator("helmholtz", helmholtz, is_oscillatory=helmholtz.is_oscillatory)

__all__ = ["helmholtz"]
