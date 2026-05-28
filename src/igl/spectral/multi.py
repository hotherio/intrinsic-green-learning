# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportAttributeAccessIssue=false
"""Mixture wrapper combining multiple 1-D spectral bases on the same dimension."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from igl.exceptions import IGLConfigError
from igl.types import SpectralBasis


class MultiSpectralBasis(nn.Module):
    """A weighted concatenation of several :class:`SpectralBasis` instances.

    Implements the :class:`SpectralBasis` Protocol: ``n_modes`` is the
    sum of the sub-bases' mode counts, ``eigenvalues`` is the concatenated
    vector, and ``null_indices`` is the union of the sub-bases' null
    indices (re-numbered to account for offsets).

    The mixing weights are softmax-normalised mode-block weights —
    every basis is multiplied by a learned (or fixed) coefficient so the
    optimiser can up- or down-weight a contribution.

    Args:
        bases: Sequence of two or more :class:`SpectralBasis` instances.
        learnable: When ``True`` (default), the mixing weights are
            ``nn.Parameter``; otherwise they are equal and fixed.

    Raises:
        IGLConfigError: For empty or singleton sequences.
    """

    n_modes: int
    null_indices: tuple[int, ...]
    domain: tuple[float, float]

    def __init__(
        self,
        bases: Sequence[SpectralBasis],
        *,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        if len(bases) < 2:  # noqa: PLR2004
            raise IGLConfigError(
                f"MultiSpectralBasis needs >= 2 bases, got {len(bases)}",
            )
        self._bases = nn.ModuleList([b for b in bases if isinstance(b, nn.Module)])
        if len(self._bases) != len(bases):
            raise IGLConfigError("all bases must be nn.Modules")

        self.n_modes = sum(b.n_modes for b in bases)
        # Concatenate eigenvalues + re-number null indices.
        offsets: list[int] = []
        offset = 0
        for b in bases:
            offsets.append(offset)
            offset += b.n_modes
        self.register_buffer(
            "eigenvalues",
            torch.cat([b.eigenvalues for b in bases], dim=0),
        )
        null: list[int] = []
        for off, b in zip(offsets, bases, strict=True):
            null.extend(off + j for j in b.null_indices)
        self.null_indices = tuple(null)
        # All sub-bases share the same input domain, but in general the
        # union doesn't have a tight bounding box — use the widest range.
        lows = [b.domain[0] for b in bases]
        highs = [b.domain[1] for b in bases]
        self.domain = (min(lows), max(highs))

        log_weights = torch.zeros(len(bases))
        if learnable:
            self.log_weights = nn.Parameter(log_weights)
        else:
            self.register_buffer("log_weights", log_weights)

    def evaluate(self, z: torch.Tensor, /) -> torch.Tensor:
        """Evaluate every sub-basis and weight by the softmax-mixed coefficients."""
        outputs: list[torch.Tensor] = []
        weights = F.softmax(self.log_weights, dim=0)
        for i, basis in enumerate(self._bases):
            outputs.append(basis(z) * weights[i])
        return torch.cat(outputs, dim=-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.evaluate(z)


__all__ = ["MultiSpectralBasis"]
