"""Soft-box kernel: smooth indicator of ``[-σ, σ]`` via sigmoid difference.

``soft_box(d, σ) = sigmoid((d + σ)/τ) - sigmoid((d - σ)/τ)`` with ``τ = 0.1``.

The result is always non-negative, so the kernel is non-oscillatory.
"""

import torch

from igl.kernels._constants import KERNEL_EPS
from igl.kernels._registry import register_operator

_TAU = 0.1


class _SoftBox:
    is_oscillatory: bool = False

    def __call__(self, d: torch.Tensor, sigma: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor]:
        left = torch.sigmoid((d + sigma) / _TAU)
        right = torch.sigmoid((d - sigma) / _TAU)
        log_abs = torch.log((left - right).clamp(min=KERNEL_EPS))
        return log_abs, torch.ones_like(d)


soft_box = _SoftBox()
register_operator("soft_box", soft_box, is_oscillatory=soft_box.is_oscillatory)

__all__ = ["soft_box"]
