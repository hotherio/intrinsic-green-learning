"""Public Protocols, enums, and lightweight type aliases used across :mod:`igl`.

String-valued choices (kernel operator names, sampling modes, normalization
modes) are :class:`enum.StrEnum` classes — the **canonical reference** for
their value sets. Each enum is paired with a companion ``Literal`` alias
listing the same values so callers can pass either an enum member (recommended
for IDE autocomplete and rename safety) or a raw literal string (basedpyright
narrows the allowed values via the Literal). The Literal must stay synced
with the StrEnum; both live side-by-side in this module so drift is hard to
miss.

A ``…Like`` union alias is also provided per concept so function signatures
can advertise the friendly union directly:

::

    def fn(operator: OperatorNameLike) -> None: ...

    fn(OperatorName.GAUSSIAN)   # OK (member)
    fn("gaussian")              # OK (Literal-narrowed)
    fn("not-a-kernel")          # static type error

These types are re-exported by ``igl/__init__.py`` so downstream consumers
can implement custom encoders, loss strategies, and samplers without
depending on the internal modules where they're consumed.
"""

from collections.abc import Mapping
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

import torch


class OperatorName(StrEnum):
    """Built-in kernel operator names. Canonical reference for kernel choices."""

    GAUSSIAN = "gaussian"
    LAPLACIAN = "laplacian"
    CAUCHY = "cauchy"
    YUKAWA = "yukawa"
    MULTIQUADRIC = "multiquadric"
    HELMHOLTZ = "helmholtz"
    GABOR = "gabor"
    MEXICAN_HAT = "mexican_hat"
    SOFT_BOX = "soft_box"


OperatorNameLiteral = Literal[
    "gaussian",
    "laplacian",
    "cauchy",
    "yukawa",
    "multiquadric",
    "helmholtz",
    "gabor",
    "mexican_hat",
    "soft_box",
]
"""Literal companion of :class:`OperatorName`. Mirrors the enum's values."""

type OperatorNameLike = OperatorName | OperatorNameLiteral
"""Type accepting either an :class:`OperatorName` member or a matching string."""


class SamplingMode(StrEnum):
    """Truncation-level sampling strategies for Matryoshka training."""

    UNIFORM = "uniform"
    POWER_LAW = "power_law"


SamplingModeLiteral = Literal["uniform", "power_law"]
"""Literal companion of :class:`SamplingMode`."""

type SamplingModeLike = SamplingMode | SamplingModeLiteral


class NormalizeMode(StrEnum):
    """Row-wise normalization applied to the design matrix Φ before lstsq.

    - ``NONE``: identity.
    - ``SOFTMAX``: row-softmax.
    - ``L2``: L2-normalise each row.
    - ``NW``: Nadaraya–Watson (divide each row by its sum, with a small epsilon).
    """

    NONE = "none"
    SOFTMAX = "softmax"
    L2 = "l2"
    NW = "nw"


NormalizeModeLiteral = Literal["none", "softmax", "l2", "nw"]
"""Literal companion of :class:`NormalizeMode`."""

type NormalizeModeLike = NormalizeMode | NormalizeModeLiteral


class NormType(StrEnum):
    """Normalization layer inserted after each hidden ``Linear`` of an MLP encoder."""

    LAYER = "layer"
    BATCH = "batch"
    NONE = "none"


NormTypeLiteral = Literal["layer", "batch", "none"]
"""Literal companion of :class:`NormType`."""

type NormTypeLike = NormType | NormTypeLiteral


class ActivationType(StrEnum):
    """Activation function applied after each ``(Linear → Norm)`` block."""

    SILU = "silu"
    TANH = "tanh"
    RELU = "relu"
    GELU = "gelu"


ActivationTypeLiteral = Literal["silu", "tanh", "relu", "gelu"]
"""Literal companion of :class:`ActivationType`."""

type ActivationTypeLike = ActivationType | ActivationTypeLiteral


class EncoderKind(StrEnum):
    """Kind of encoder built by :func:`igl.build_mlp_encoder` / future factories."""

    MLP = "mlp"
    LINEAR = "linear"


EncoderKindLiteral = Literal["mlp", "linear"]
"""Literal companion of :class:`EncoderKind`."""

type EncoderKindLike = EncoderKind | EncoderKindLiteral


class SchedulerType(StrEnum):
    """Learning-rate scheduler choices for :class:`igl.MatryoshkaTrainer`."""

    COSINE_WARM_RESTARTS = "cosine_warm_restarts"
    NONE = "none"


SchedulerTypeLiteral = Literal["cosine_warm_restarts", "none"]
"""Literal companion of :class:`SchedulerType`."""

type SchedulerTypeLike = SchedulerType | SchedulerTypeLiteral


DimensionCurve = Mapping[int, float]
"""Post-training mapping from truncation level ``k`` to validation loss at ``k``."""


@runtime_checkable
class EncoderProtocol(Protocol):
    """A callable mapping ambient inputs to an ``max_dim``-dimensional latent."""

    input_dim: int
    max_dim: int

    def __call__(self, x: torch.Tensor, /) -> torch.Tensor: ...


@runtime_checkable
class OperatorFn(Protocol):
    """Log-space kernel operator.

    Given a distance tensor ``d`` and a width tensor ``sigma`` of broadcastable
    shapes, returns ``(log_abs, sign)`` so the multi-scale product kernel can be
    accumulated in log-space while tracking signs for oscillatory operators.
    """

    is_oscillatory: bool

    def __call__(
        self,
        d: torch.Tensor,
        sigma: torch.Tensor,
        /,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


@runtime_checkable
class LossStrategy(Protocol):
    """Pluggable loss for :class:`igl.MatryoshkaTrainer`.

    Implementations supply task-specific targets (one-hot encoding,
    pass-through, log-Euclidean SPD targets, …), the loss to minimise, and a
    scalar metric for early-stopping decisions.

    Attributes:
        higher_is_better: ``True`` when ``metric()`` should be maximised
            (e.g. accuracy), ``False`` when minimised (e.g. MSE, AIRM).
    """

    higher_is_better: bool

    def target(self, y: torch.Tensor) -> torch.Tensor: ...

    def loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor: ...

    def metric(self, pred: torch.Tensor, target: torch.Tensor) -> float: ...


@runtime_checkable
class MatryoshkaSampler(Protocol):
    """Samples a truncation level ``k ∈ {1, …, d_max}`` per training step."""

    def __call__(self, d_max: int, /) -> int: ...


__all__ = [
    "ActivationType",
    "ActivationTypeLike",
    "ActivationTypeLiteral",
    "DimensionCurve",
    "EncoderKind",
    "EncoderKindLike",
    "EncoderKindLiteral",
    "EncoderProtocol",
    "LossStrategy",
    "MatryoshkaSampler",
    "NormType",
    "NormTypeLike",
    "NormTypeLiteral",
    "NormalizeMode",
    "NormalizeModeLike",
    "NormalizeModeLiteral",
    "OperatorFn",
    "OperatorName",
    "OperatorNameLike",
    "OperatorNameLiteral",
    "SamplingMode",
    "SamplingModeLike",
    "SamplingModeLiteral",
    "SchedulerType",
    "SchedulerTypeLike",
    "SchedulerTypeLiteral",
]
