"""Row-wise normalization of the design matrix Φ before the lstsq solve."""

from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812

from igl.exceptions import IGLConfigError
from igl.types import NormalizeMode, NormalizeModeLike

_EPS = 1e-8


def normalize_phi(phi: torch.Tensor, mode: NormalizeModeLike) -> torch.Tensor:
    """Normalize the design matrix per row.

    Args:
        phi: Design matrix of shape ``[N, R]``.
        mode: A :class:`NormalizeMode` member or matching string.

    Returns:
        The normalised matrix, same shape as ``phi``.

    Raises:
        IGLConfigError: If ``mode`` is not a recognised value.
    """
    try:
        mode_enum = NormalizeMode(mode)
    except ValueError as exc:
        raise IGLConfigError(f"unknown normalize mode: {mode!r}") from exc
    if mode_enum is NormalizeMode.NONE:
        return phi
    if mode_enum is NormalizeMode.SOFTMAX:
        return F.softmax(phi, dim=-1)
    if mode_enum is NormalizeMode.L2:
        # torch's .norm has partial stubs; cast to recover the known return type.
        norm = cast(torch.Tensor, phi.norm(dim=-1, keepdim=True))  # pyright: ignore[reportUnknownMemberType]
        return phi / (norm + _EPS)
    return phi / (phi.sum(dim=-1, keepdim=True) + _EPS)


__all__ = ["normalize_phi"]
