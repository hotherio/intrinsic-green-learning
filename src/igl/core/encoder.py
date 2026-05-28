"""Latent-space encoders for IGL.

Two implementations are provided:

- :class:`LinearEncoder` — a single ``nn.Linear`` projection. Useful for
  ablations and for cases where the ambient features are already a good
  representation.
- :class:`MLPEncoder` — a configurable MLP with LayerNorm/BatchNorm/Identity
  normalization and SiLU/GELU/ReLU/Tanh activations between blocks. The
  output is unbounded (no final non-linearity), which keeps the latent scale
  free for the optimizer to set. ``hidden`` accepts either a single ``int``
  (uniform width across all blocks) or a sequence of ints (per-layer widths,
  e.g. ``(256, 128, 64)`` for a pyramidal encoder).

Both satisfy :class:`igl.types.EncoderProtocol`.
"""

from collections.abc import Sequence

import torch
from torch import nn

from igl.config import EncoderConfig
from igl.exceptions import IGLConfigError
from igl.types import (
    ActivationType,
    ActivationTypeLike,
    EncoderKind,
    NormType,
    NormTypeLike,
)


def _make_norm(kind: NormType, width: int) -> nn.Module:
    if kind is NormType.LAYER:
        return nn.LayerNorm(width)
    if kind is NormType.BATCH:
        return nn.BatchNorm1d(width, affine=True)
    return nn.Identity()


def _make_activation(kind: ActivationType) -> nn.Module:
    if kind is ActivationType.SILU:
        return nn.SiLU()
    if kind is ActivationType.TANH:
        return nn.Tanh()
    if kind is ActivationType.RELU:
        return nn.ReLU()
    return nn.GELU()


def _resolve_widths(hidden: int | Sequence[int], depth: int | None) -> list[int]:
    """Turn ``(hidden, depth)`` into an explicit list of per-layer widths.

    Args:
        hidden: A single width (uniform) or a sequence of per-layer widths.
        depth: Explicit depth override. When ``hidden`` is an ``int`` and
            ``depth`` is ``None``, depth defaults to 2. When ``hidden`` is a
            sequence, ``depth`` must either be ``None`` or match
            ``len(hidden)``.

    Raises:
        IGLConfigError: For empty sequences, non-positive widths, or a depth
            that contradicts the hidden sequence length.
    """
    if isinstance(hidden, int):
        if hidden < 1:
            raise IGLConfigError(f"hidden must be >= 1, got {hidden}")
        effective_depth = 2 if depth is None else depth
        if effective_depth < 1:
            raise IGLConfigError(f"depth must be >= 1, got {effective_depth}")
        return [hidden] * effective_depth

    widths = list(hidden)
    if not widths:
        raise IGLConfigError("hidden sequence must be non-empty")
    for index, width in enumerate(widths):
        if width < 1:
            raise IGLConfigError(f"hidden[{index}] must be >= 1, got {width}")
    if depth is not None and depth != len(widths):
        raise IGLConfigError(
            f"depth ({depth}) does not match hidden sequence length ({len(widths)})",
        )
    return widths


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
        hidden: Either a single width applied uniformly to every hidden layer
            (default ``256``), or a sequence of per-layer widths
            (e.g. ``(256, 128, 64)`` for a pyramidal encoder).
        depth: Optional explicit depth. When ``hidden`` is an ``int``,
            defaults to ``2``. When ``hidden`` is a sequence and ``depth`` is
            provided, the two must agree.
        norm: Type of normalization in each block (default
            :data:`igl.NormType.LAYER`). Accepts the enum or a matching
            string.
        activation: Activation function (default
            :data:`igl.ActivationType.SILU`). Accepts the enum or a matching
            string.

    Raises:
        IGLConfigError: If any dimension is non-positive, ``hidden`` is an
            empty sequence, or ``depth`` contradicts ``len(hidden)``.
    """

    input_dim: int
    max_dim: int
    hidden_widths: tuple[int, ...]

    def __init__(
        self,
        input_dim: int,
        max_dim: int,
        *,
        hidden: int | Sequence[int] = 256,
        depth: int | None = None,
        norm: NormTypeLike = NormType.LAYER,
        activation: ActivationTypeLike = ActivationType.SILU,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise IGLConfigError(f"input_dim must be >= 1, got {input_dim}")
        if max_dim < 1:
            raise IGLConfigError(f"max_dim must be >= 1, got {max_dim}")

        widths = _resolve_widths(hidden, depth)
        norm_enum = NormType(norm)
        activation_enum = ActivationType(activation)

        self.input_dim = input_dim
        self.max_dim = max_dim
        self.hidden_widths = tuple(widths)

        layers: list[nn.Module] = []
        d_in = input_dim
        for width in widths:
            layers.append(nn.Linear(d_in, width))
            layers.append(_make_norm(norm_enum, width))
            layers.append(_make_activation(activation_enum))
            d_in = width
        layers.append(nn.Linear(d_in, max_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_mlp_encoder(
    input_dim: int,
    max_dim: int,
    *,
    config: EncoderConfig,
) -> MLPEncoder:
    """Factory: construct an :class:`MLPEncoder` from an :class:`igl.EncoderConfig`.

    Args:
        input_dim: Ambient dimension ``D``.
        max_dim: Latent dimension ``d_max``.
        config: Encoder configuration. ``config.kind`` must be
            :data:`igl.EncoderKind.MLP`.

    Raises:
        IGLConfigError: If ``config.kind`` is not :data:`igl.EncoderKind.MLP`.
    """
    if config.kind is not EncoderKind.MLP:
        raise IGLConfigError(f"build_mlp_encoder supports kind=MLP, got {config.kind!r}")
    return MLPEncoder(
        input_dim=input_dim,
        max_dim=max_dim,
        hidden=config.hidden,
        depth=config.depth,
        norm=config.norm,
        activation=config.activation,
    )


__all__ = [
    "LinearEncoder",
    "MLPEncoder",
    "build_mlp_encoder",
]
