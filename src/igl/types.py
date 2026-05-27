"""Public Protocols, Literal aliases, and lightweight type aliases used across :mod:`igl`.

These types are imported by `igl/__init__.py` and re-exported at the top level so
that downstream consumers can implement custom encoders, loss strategies, and
samplers without depending on the internal modules where they're consumed.
"""

from collections.abc import Mapping
from typing import Literal, Protocol, runtime_checkable

import torch

OperatorName = Literal[
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
"""Built-in kernel operators. Custom operators may be registered at runtime."""

SamplingMode = Literal["uniform", "power_law"]
"""Truncation-level sampling strategies for Matryoshka training."""

NormalizeMode = Literal["none", "softmax", "l2", "nw"]
"""Row-wise normalization applied to the design matrix Φ before the lstsq solve.

- ``"none"``: identity.
- ``"softmax"``: row-softmax.
- ``"l2"``: L2-normalise each row.
- ``"nw"``: Nadaraya–Watson (divide each row by its sum, with a small epsilon).
"""

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
    "DimensionCurve",
    "EncoderProtocol",
    "LossStrategy",
    "MatryoshkaSampler",
    "NormalizeMode",
    "OperatorFn",
    "OperatorName",
    "SamplingMode",
]
