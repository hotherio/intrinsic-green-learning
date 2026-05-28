"""Yukawa Green's-function kernel: ``exp(-|d| / σ)`` (Laplacian shape, separate identity).

Mathematically identical to :mod:`igl.kernels.laplacian`; kept as a distinct
registered operator so users can mix the same shape under different names in
multi-operator configurations (the per-name learnable γ and σ make the two
diverge during training).
"""

import torch

from igl.kernels._registry import register_operator

_EPS = 1e-8


class _Yukawa:
    is_oscillatory: bool = False

    def __call__(self, d: torch.Tensor, sigma: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor]:
        log_abs = -torch.abs(d) / (sigma + _EPS)
        return log_abs, torch.ones_like(d)


yukawa = _Yukawa()
register_operator("yukawa", yukawa, is_oscillatory=yukawa.is_oscillatory)

__all__ = ["yukawa"]
