"""Numerical constants shared across the built-in kernel operators.

Every built-in operator independently defined ``_EPS = 1e-8`` for two related
but distinct numerical purposes:

1. **Safe division** in ``sigma + EPS`` / ``sigma**2 + EPS`` denominators —
   keeps the kernel finite when ``sigma`` underflows to zero during training.
2. **Logarithm clamp** in ``log(x).clamp(min=EPS)`` calls — keeps the
   log-space accumulation finite when an operator factor approaches zero
   (oscillatory kernels, soft-box indicator).

Both purposes use the same value, so a single constant suffices today;
keeping it here makes the choice grep-able if a future change wants to
diverge the two floors.
"""

KERNEL_EPS: float = 1e-8
"""Numerical floor shared by built-in kernel operators (see module docstring)."""


__all__ = ["KERNEL_EPS"]
