"""Pullback-metric orthogonality penalty + Stiefel projection helpers.

The pullback metric of an encoder ``Ψ`` at input ``x`` is the symmetric
positive-semidefinite matrix ``g(x) = J(x) J(x)^T`` where ``J = ∂Ψ/∂x``.
When ``g(x)`` is diagonal, the latent coordinates are orthogonal to first
order around ``x`` — the Stäckel condition; geodesics separate cleanly per
coordinate. The :class:`OrthogonalityPenalty` regularizer drives the
off-diagonal mass toward zero so the encoder learns
geometrically-disentangled coordinates.

Two complementary tools:

- :func:`init_encoder_orthogonal_` initialises each ``nn.Linear`` weight in
  the encoder to a Stiefel matrix via PyTorch's QR-orthogonal init. Cheap
  and applies before training.
- :class:`OrthogonalityPenalty` implements :class:`igl.types.ExtraLoss` and
  plugs into :meth:`igl.MatryoshkaTrainer.fit` via the ``extra_losses``
  parameter. Active during training; skipped on ``k < 2`` (the metric is
  trivially diagonal for a single dimension).
"""

import torch
from torch import nn

from igl.exceptions import IGLConfigError

_MIN_K_FOR_ORTH = 2


def jacobian(encoder: nn.Module, x: torch.Tensor, *, output_dim: int) -> torch.Tensor:
    """Compute ``J[b, j, i] = ∂encoder(x_b)_j / ∂x_b_i`` keeping the autograd graph.

    Used to drive the orthogonality penalty: the resulting Jacobian gradient
    flows back to the encoder parameters so the penalty actually updates them.

    Args:
        encoder: The encoder module.
        x: Input batch ``[B, D]``.
        output_dim: Number of output dimensions ``d`` of the encoder.

    Returns:
        Jacobian tensor of shape ``[B, d, D]``.
    """
    x = x.detach().requires_grad_(True)
    z = encoder(x)
    rows: list[torch.Tensor] = []
    for j in range(output_dim):
        grad = torch.autograd.grad(z[:, j].sum(), x, create_graph=True, retain_graph=True)[0]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        rows.append(grad)  # pyright: ignore[reportUnknownArgumentType]
    return torch.stack(rows, dim=1)


def pullback_metric(j: torch.Tensor) -> torch.Tensor:
    """Pullback metric ``g = J J^T`` of shape ``[B, d, d]``.

    Args:
        j: Jacobian tensor ``[B, d, D]``.
    """
    return torch.bmm(j, j.transpose(1, 2))


def orthogonality_loss(g: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Sum of squared off-diagonals divided by sum of squared diagonals.

    Args:
        g: Pullback metric ``[B, d, d]``.
        eps: Numerical floor on the diagonal-squared denominator.

    Returns:
        Scalar penalty (mean across the batch).
    """
    diag = torch.diagonal(g, dim1=-2, dim2=-1)
    off = g - torch.diag_embed(diag)
    off_sq = (off**2).sum(dim=(-2, -1))
    diag_sq = (diag**2).sum(dim=-1)
    return (off_sq / (diag_sq + eps)).mean()


@torch.no_grad()
def init_encoder_orthogonal_(encoder: nn.Module) -> int:
    """Initialize every ``nn.Linear`` weight in ``encoder`` with QR-orthogonal init.

    Operates in-place. Returns the number of linear layers initialised.
    """
    count = 0
    for module in encoder.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)  # pyright: ignore[reportUnknownMemberType]
            count += 1
    return count


class OrthogonalityPenalty:
    """:class:`igl.types.ExtraLoss` driving the pullback metric toward diagonality.

    The contribution is

        weight × ‖off-diag(g)‖² / (‖diag(g)‖² + ε)

    where ``g = J J^T`` and ``J`` is the encoder Jacobian at the truncated
    output. Skipped for ``k < 2`` (the metric is trivially diagonal for a
    single dimension).

    Args:
        weight: Multiplier applied by the trainer. ``0.1`` is the value
            validated on EEG data.
        every: Call frequency in *batches*. ``20`` is a reasonable default —
            the Jacobian computation is O(d · D) so applying it on every
            batch is expensive.
        eps: Diagonal-squared floor for numerical stability.
    """

    weight: float
    every: int
    eps: float

    def __init__(self, *, weight: float = 0.1, every: int = 20, eps: float = 1e-6) -> None:
        if weight < 0:
            raise IGLConfigError(f"weight must be non-negative, got {weight}")
        if every < 1:
            raise IGLConfigError(f"every must be >= 1, got {every}")
        self.weight = weight
        self.every = every
        self.eps = eps

    def __call__(
        self,
        *,
        encoder: nn.Module,
        x_batch: torch.Tensor,
        gate_mask: torch.Tensor,
        k: int,
        epoch: int,  # noqa: ARG002
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor | None:
        if k < _MIN_K_FOR_ORTH:
            return None
        # Compute the Jacobian only for the active latent dims.
        j_full = jacobian(encoder, x_batch, output_dim=gate_mask.shape[0])
        j_active = j_full[:, : int(gate_mask.sum().item()), :]
        g = pullback_metric(j_active)
        return orthogonality_loss(g, eps=self.eps)


__all__ = [
    "OrthogonalityPenalty",
    "init_encoder_orthogonal_",
    "jacobian",
    "orthogonality_loss",
    "pullback_metric",
]
