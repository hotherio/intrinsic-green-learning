"""Latent-space encoders for IGL.

Two implementations are provided:

- :class:`LinearEncoder` — a single ``nn.Linear`` projection. Useful for
  ablations and for cases where the ambient features are already a good
  representation.
- :class:`MLPEncoder` — a configurable depth-``d`` MLP with LayerNorm/BatchNorm/
  Identity normalization and SiLU/GELU/ReLU/Tanh activations between blocks.
  The output is unbounded (no final non-linearity), which keeps the latent
  scale free for the optimizer to set.

Both satisfy :class:`igl.types.EncoderProtocol`.
"""

from typing import Literal

import torch
from torch import nn

from igl.exceptions import IGLConfigError

NormType = Literal["layer", "batch", "none"]
ActivationType = Literal["silu", "tanh", "relu", "gelu"]


def _make_norm(kind: NormType, width: int) -> nn.Module:
    if kind == "layer":
        return nn.LayerNorm(width)
    if kind == "batch":
        return nn.BatchNorm1d(width, affine=True)
    return nn.Identity()


def _make_activation(kind: ActivationType) -> nn.Module:
    if kind == "silu":
        return nn.SiLU()
    if kind == "tanh":
        return nn.Tanh()
    if kind == "relu":
        return nn.ReLU()
    return nn.GELU()


class LinearEncoder(nn.Module):
    """Single ``nn.Linear`` projection from ambient to latent space.

    Args:
        input_dim: Ambient dimension ``D``.
        max_dim: Latent dimension ``d_max``.
    """

    input_dim: int
    max_dim: int

    def __init__(self, input_dim: int, max_dim: int) -> None:
        super().__init__()
        if input_dim < 1:
            raise IGLConfigError(f"input_dim must be >= 1, got {input_dim}")
        if max_dim < 1:
            raise IGLConfigError(f"max_dim must be >= 1, got {max_dim}")
        self.input_dim = input_dim
        self.max_dim = max_dim
        self.linear = nn.Linear(input_dim, max_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPEncoder(nn.Module):
    """MLP encoder: ``depth × (Linear → Norm → Activation) → Linear``.

    Args:
        input_dim: Ambient dimension ``D``.
        max_dim: Latent dimension ``d_max``.
        hidden: Width of each hidden layer (default ``256``).
        depth: Number of intermediate blocks before the final projection
            (default ``2`` — matches the reference Matryoshka implementation).
        norm: Type of normalization in each block (``"layer"`` by default).
        activation: Activation function name (``"silu"`` by default).

    Raises:
        IGLConfigError: If ``depth`` is not at least 1, or any dimension is
            non-positive.
    """

    input_dim: int
    max_dim: int

    def __init__(
        self,
        input_dim: int,
        max_dim: int,
        *,
        hidden: int = 256,
        depth: int = 2,
        norm: NormType = "layer",
        activation: ActivationType = "silu",
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise IGLConfigError(f"input_dim must be >= 1, got {input_dim}")
        if max_dim < 1:
            raise IGLConfigError(f"max_dim must be >= 1, got {max_dim}")
        if depth < 1:
            raise IGLConfigError(f"depth must be >= 1, got {depth}")
        if hidden < 1:
            raise IGLConfigError(f"hidden must be >= 1, got {hidden}")

        self.input_dim = input_dim
        self.max_dim = max_dim

        layers: list[nn.Module] = []
        d_in = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(d_in, hidden))
            layers.append(_make_norm(norm, hidden))
            layers.append(_make_activation(activation))
            d_in = hidden
        layers.append(nn.Linear(hidden, max_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


__all__ = [
    "ActivationType",
    "LinearEncoder",
    "MLPEncoder",
    "NormType",
]
