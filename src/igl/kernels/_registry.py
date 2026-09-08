"""Operator registry for the multi-scale Green's-function kernel.

Each operator is an ``OperatorFn`` (see :class:`igl.types.OperatorFn`) returning
``(log_abs, sign)`` so the multi-scale product can be accumulated in log-space
while sign parity is tracked across dimensions for oscillatory kernels.

External users can :func:`register_operator` to plug in new kernels at runtime;
:func:`get_operator` and :func:`list_operators` enumerate the current contents.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from igl.exceptions import IGLConfigError
from igl.types import OperatorFn

SeparableParts = tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]
"""``(f, g)`` with ``log|G(d, σ)| = -f(d) · g(σ)``: the operator is separable in distance and width."""


@dataclass(frozen=True, slots=True)
class Operator:
    """A registered kernel operator.

    Attributes:
        name: Stable identifier used in configs and serialization.
        fn: Callable implementing :class:`igl.types.OperatorFn`.
        is_oscillatory: ``True`` if the kernel can take negative values
            (e.g. ``helmholtz``, ``gabor``, ``mexican_hat``); the kernel layer
            uses this to decide whether sign tracking is needed.
        separable: Optional ``(f, g)`` pair with ``log|G(d, σ)| = -f(d) · g(σ)``
            (e.g. Gaussian: ``f = d²``, ``g = 1 / (2σ² + ε)``). When present the
            kernel layer can contract the product over latent dimensions as one
            matrix product instead of materialising an ``[N, R, K, d]`` tensor.
    """

    name: str
    fn: OperatorFn
    is_oscillatory: bool
    separable: SeparableParts | None = None


_OPERATORS: dict[str, Operator] = {}


def register_operator(
    name: str,
    fn: OperatorFn,
    *,
    is_oscillatory: bool = False,
    separable: SeparableParts | None = None,
) -> None:
    """Register a kernel operator under ``name``.

    Args:
        name: Stable identifier (must be unique).
        fn: Callable matching :class:`igl.types.OperatorFn`.
        is_oscillatory: Whether the kernel can take negative values.
        separable: Optional ``(f, g)`` pair with ``log|G(d, σ)| = -f(d) · g(σ)``
            enabling the kernel layer's contracted fast path.

    Raises:
        IGLConfigError: If ``name`` is already registered.
    """
    if name in _OPERATORS:
        raise IGLConfigError(f"operator {name!r} is already registered")
    _OPERATORS[name] = Operator(name=name, fn=fn, is_oscillatory=is_oscillatory, separable=separable)


def get_operator(name: str) -> Operator:
    """Return the operator registered under ``name``.

    Raises:
        IGLConfigError: If ``name`` is not registered.
    """
    try:
        return _OPERATORS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_OPERATORS))
        raise IGLConfigError(f"unknown operator {name!r}; registered: [{available}]") from exc


def list_operators() -> list[str]:
    """Return the sorted list of registered operator names."""
    return sorted(_OPERATORS)


__all__ = [
    "Operator",
    "SeparableParts",
    "get_operator",
    "list_operators",
    "register_operator",
]
