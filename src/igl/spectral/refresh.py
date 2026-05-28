"""`ExtraLoss` hook that periodically refreshes a learned LB basis."""

from typing import cast

import torch
from torch import nn

from igl.exceptions import IGLConfigError
from igl.spectral.bases.learned_lb import LearnedLaplacianBasis


class LearnedLBRefresh:
    """Refresh a :class:`LearnedLaplacianBasis` every ``every`` batches.

    Implements :class:`igl.types.ExtraLoss` purely for its scheduling
    semantics — it returns ``None`` (no loss contribution) and uses the
    callback to invoke :meth:`LearnedLaplacianBasis.refresh` with the
    current batch's encoded latents. The first refresh happens on the
    very first call.

    Args:
        basis: The learned-LB basis to keep up to date.
        every: Refresh frequency in batches.
        weight: Ignored (kept for protocol compatibility); defaults to 0.

    Raises:
        IGLConfigError: When ``every < 1``.
    """

    weight: float
    every: int

    def __init__(
        self,
        basis: LearnedLaplacianBasis,
        *,
        every: int = 200,
        weight: float = 0.0,
    ) -> None:
        if every < 1:
            raise IGLConfigError(f"every must be >= 1, got {every}")
        self._basis = basis
        self.every = every
        self.weight = weight

    def __call__(
        self,
        *,
        encoder: nn.Module,
        x_batch: torch.Tensor,
        gate_mask: torch.Tensor,  # noqa: ARG002
        k: int,  # noqa: ARG002
        epoch: int,  # noqa: ARG002
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor | None:
        with torch.no_grad():
            z = cast(torch.Tensor, encoder(x_batch))
            self._basis.refresh(z)
        return None


__all__ = ["LearnedLBRefresh"]
