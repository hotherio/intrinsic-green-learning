"""Gaussian Green's-function kernel: ``exp(-d² / (2σ²))``."""

import torch

from igl.kernels._registry import register_operator

_EPS = 1e-8


class _Gaussian:
    is_oscillatory: bool = False

    def __call__(self, d: torch.Tensor, sigma: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor]:
        log_abs = -(d**2) / (2 * sigma**2 + _EPS)
        return log_abs, torch.ones_like(d)


gaussian = _Gaussian()
register_operator("gaussian", gaussian, is_oscillatory=gaussian.is_oscillatory)

__all__ = ["gaussian"]
