"""Row-wise normalization of the design matrix Φ before the lstsq solve."""

from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812

from igl.exceptions import IGLConfigError
from igl.types import NormalizeMode

_EPS = 1e-8


def normalize_phi(phi: torch.Tensor, mode: NormalizeMode) -> torch.Tensor:
    """Normalize the design matrix per row.

    Args:
        phi: Design matrix of shape ``[N, R]``.
        mode: One of ``"none"`` / ``"softmax"`` / ``"l2"`` / ``"nw"``.

    Returns:
        The normalised matrix, same shape as ``phi``.

    Raises:
        IGLConfigError: If ``mode`` is not a recognised value.
    """
    if mode == "none":
        return phi
    if mode == "softmax":
        return F.softmax(phi, dim=-1)
    if mode == "l2":
        # torch's .norm has partial stubs; cast to recover the known return type.
        norm = cast(torch.Tensor, phi.norm(dim=-1, keepdim=True))  # pyright: ignore[reportUnknownMemberType]
        return phi / (norm + _EPS)
    if mode == "nw":
        return phi / (phi.sum(dim=-1, keepdim=True) + _EPS)
    raise IGLConfigError(f"unknown normalize mode: {mode!r}")


__all__ = ["normalize_phi"]
