"""Null-space basis augmentations for the design matrix.

Operators $L$ commonly invoked in IGL (Laplacian with Neumann BCs,
Helmholtz at a resonance, Schrödinger with bound states, …) typically
have non-trivial kernels: functions $\\xi$ with $L\\xi = 0$. The Green's
kernel cannot reach those modes — its expansion divides by
eigenvalues, and the zero eigenvalues' contributions vanish in the
limit. The fix the paper adopts is to **augment** the design matrix
$\\Phi$ with explicit columns evaluating each null-mode basis function,
so the null-space component is carried directly by the design matrix
rather than reached through the kernel expansion. Their coefficients are
then fitted alongside the anchor coefficients by
:func:`igl.direct_solve_weights`, which applies its ridge uniformly to
every column of $\\Phi$ — null-space columns are not exempted.

This module ships three null-space bases:

- :class:`ConstantNullSpace` — one column of ones. Standard for the
  Neumann Laplacian (its kernel is the constants).
- :class:`PolynomialNullSpace` — constants + per-dimension monomials up
  to a fixed degree. Includes the harmonics of the Euclidean Laplacian
  up to that degree.
- :class:`CustomNullSpace` — user-supplied callable for domain-specific
  null modes.

The :func:`build_null_space` factory builds one from an
:class:`igl.types.NullSpaceKind` value (used by configs).
"""

from collections.abc import Callable

import torch

from igl.exceptions import IGLConfigError
from igl.types import NullSpaceKind, NullSpaceKindLike


class ConstantNullSpace:
    """One column of ones — the DC mode.

    Standard null space for the Neumann Laplacian (constants are
    harmonic with the Neumann boundary condition). Useful in general as
    a cheap way to give a local :class:`igl.GreenKernel` a DC mode it
    otherwise lacks.
    """

    n_columns: int = 1

    def evaluate(self, z: torch.Tensor, /) -> torch.Tensor:
        return torch.ones(z.shape[0], 1, dtype=z.dtype, device=z.device)


class PolynomialNullSpace:
    """Constants + per-dimension monomials up to ``degree``.

    For ``degree = 1``: one column of ones plus ``d`` columns ``z_j``.
    For ``degree = 2``: also includes ``z_j^2``. Up to ``degree = k``:
    the constant plus all univariate monomials per dimension —
    contains the harmonics of the Euclidean Laplacian up to that
    degree (mixed monomials like ``z_j z_k`` are not included; users
    needing those can subclass :class:`CustomNullSpace`).

    Args:
        latent_dim: Number of input dimensions ``d``.
        degree: Maximum monomial degree (inclusive). Default ``1``.

    Raises:
        IGLConfigError: For ``degree < 0`` or ``latent_dim < 1``.
    """

    n_columns: int

    def __init__(self, *, latent_dim: int, degree: int = 1) -> None:
        if latent_dim < 1:
            raise IGLConfigError(f"latent_dim must be >= 1, got {latent_dim}")
        if degree < 0:
            raise IGLConfigError(f"degree must be >= 0, got {degree}")
        self.latent_dim = latent_dim
        self.degree = degree
        # Constant + degree×latent_dim monomials.
        self.n_columns = 1 + degree * latent_dim

    def evaluate(self, z: torch.Tensor, /) -> torch.Tensor:
        if z.shape[-1] != self.latent_dim:
            raise IGLConfigError(
                f"z has {z.shape[-1]} features; expected {self.latent_dim}",
            )
        columns: list[torch.Tensor] = [torch.ones(z.shape[0], 1, dtype=z.dtype, device=z.device)]
        for p in range(1, self.degree + 1):
            columns.append(z**p)
        return torch.cat(columns, dim=-1)


class CustomNullSpace:
    """Wraps a user-supplied callable as a :class:`NullSpaceBasis`.

    Args:
        fn: Callable ``z -> [N, n_columns]``.
        n_columns: Number of columns the callable returns.

    Raises:
        IGLConfigError: When ``n_columns < 1``.
    """

    n_columns: int

    def __init__(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        n_columns: int,
    ) -> None:
        if n_columns < 1:
            raise IGLConfigError(f"n_columns must be >= 1, got {n_columns}")
        self._fn = fn
        self.n_columns = n_columns

    def evaluate(self, z: torch.Tensor, /) -> torch.Tensor:
        out = self._fn(z)
        if out.shape[0] != z.shape[0] or out.shape[-1] != self.n_columns:
            raise IGLConfigError(
                f"custom null-space fn must return [N={z.shape[0]}, n_columns={self.n_columns}], got {tuple(out.shape)}",
            )
        return out


def build_null_space(
    kind: NullSpaceKindLike,
    *,
    latent_dim: int,
    degree: int = 1,
) -> ConstantNullSpace | PolynomialNullSpace | None:
    """Build a built-in null space from a :class:`NullSpaceKind`.

    Args:
        kind: One of :data:`NullSpaceKind`. ``NONE`` returns ``None``.
        latent_dim: Required for ``POLYNOMIAL``.
        degree: Forwarded to :class:`PolynomialNullSpace`.

    Returns:
        An instance of :class:`ConstantNullSpace` /
        :class:`PolynomialNullSpace`, or ``None`` if ``kind == NONE``.
    """
    enum = NullSpaceKind(kind)
    if enum is NullSpaceKind.NONE:
        return None
    if enum is NullSpaceKind.CONSTANT:
        return ConstantNullSpace()
    return PolynomialNullSpace(latent_dim=latent_dim, degree=degree)


__all__ = [
    "ConstantNullSpace",
    "CustomNullSpace",
    "PolynomialNullSpace",
    "build_null_space",
]
