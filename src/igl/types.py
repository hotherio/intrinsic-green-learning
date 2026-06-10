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
from typing import Literal, Protocol

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


class PreconditionMode(StrEnum):
    """SPD-side preconditioning applied to input covariances before AIRM.

    - ``NONE``: passthrough (legacy behaviour).
    - ``TIKHONOV``: ``C + epsilon * I``. Default. Bit-identical to ``NONE``
      at ``d ≤ 64`` (BatchNorm absorbs the constant offset) and rescues
      ``torch.linalg.eigh`` from LAPACK error 8481 at ``d ≥ 128``. See
      ``alex-eeg-igl/report/third_summary_report.tex`` §3.
    - ``TRACE``: per-matrix trace-normalisation ``C / trace(C) * d``.
      Exposed for completeness; *not* recommended as a global default —
      drops AUC by up to 0.05 on small-d datasets.
    - ``TIKHONOV_TRACE``: trace then tikhonov. Exposed; not recommended.
    """

    NONE = "none"
    TIKHONOV = "tikhonov"
    TRACE = "trace"
    TIKHONOV_TRACE = "tikhonov+trace"


PreconditionModeLiteral = Literal["none", "tikhonov", "trace", "tikhonov+trace"]
"""Literal companion of :class:`PreconditionMode`."""

type PreconditionModeLike = PreconditionMode | PreconditionModeLiteral


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


class SpectralKind(StrEnum):
    """Built-in 1-D spectral-basis identifiers.

    Each value names an orthonormal basis with known (or estimated)
    eigenvalues — the spectral decomposition of a self-adjoint operator
    on its domain. The first six are closed-form; the last two are
    data-driven (learned LB / user-supplied graph).
    """

    FOURIER_SINE = "fourier_sine"
    FOURIER_COSINE = "fourier_cosine"
    CHEBYSHEV = "chebyshev"
    LEGENDRE = "legendre"
    HERMITE = "hermite"
    LAGUERRE = "laguerre"
    LEARNED_LB = "learned_lb"
    GRAPH_LAPLACIAN = "graph_laplacian"


SpectralKindLiteral = Literal[
    "fourier_sine",
    "fourier_cosine",
    "chebyshev",
    "legendre",
    "hermite",
    "laguerre",
    "learned_lb",
    "graph_laplacian",
]
"""Literal companion of :class:`SpectralKind`."""

type SpectralKindLike = SpectralKind | SpectralKindLiteral


class NullSpaceKind(StrEnum):
    """Built-in null-space augmentation strategies."""

    NONE = "none"
    CONSTANT = "constant"
    POLYNOMIAL = "polynomial"


NullSpaceKindLiteral = Literal["none", "constant", "polynomial"]
"""Literal companion of :class:`NullSpaceKind`."""

type NullSpaceKindLike = NullSpaceKind | NullSpaceKindLiteral


class GraphLaplacianNorm(StrEnum):
    """Normalisation modes for the graph Laplacian."""

    SYMMETRIC = "symmetric"
    RANDOM_WALK = "rw"
    UNNORMALIZED = "unnormalized"


GraphLaplacianNormLiteral = Literal["symmetric", "rw", "unnormalized"]
"""Literal companion of :class:`GraphLaplacianNorm`."""

type GraphLaplacianNormLike = GraphLaplacianNorm | GraphLaplacianNormLiteral


DimensionCurve = Mapping[int, float]
"""Post-training mapping from truncation level ``k`` to validation loss at ``k``."""


class EncoderProtocol(Protocol):
    """A callable mapping ambient inputs to an ``max_dim``-dimensional latent."""

    input_dim: int
    max_dim: int

    def __call__(self, x: torch.Tensor, /) -> torch.Tensor: ...


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


class LossStrategy(Protocol):
    """Pluggable loss for :class:`igl.MatryoshkaTrainer`.

    Implementations supply task-specific targets (one-hot encoding,
    pass-through, log-Euclidean SPD targets, …), the loss to minimise, a
    scalar metric for early-stopping decisions, and — separately — a
    *curve score* used by :func:`igl.eval_dimension_curve` to make
    dimension curves informative (e.g. error rate for classifiers, which
    doesn't saturate the way cross-entropy does).

    Attributes:
        higher_is_better: ``True`` when ``metric()`` should be maximised
            (e.g. accuracy), ``False`` when minimised (e.g. MSE, AIRM).
            ``curve_score()`` is always lower-is-better regardless of this
            flag.
    """

    higher_is_better: bool

    def target(self, y: torch.Tensor) -> torch.Tensor: ...

    def loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor: ...

    def metric(self, pred: torch.Tensor, target: torch.Tensor) -> float: ...

    def curve_score(self, pred: torch.Tensor, target: torch.Tensor) -> float: ...


class MatryoshkaSampler(Protocol):
    """Samples a truncation level ``k ∈ {1, …, d_max}`` per training step."""

    def __call__(self, d_max: int, /) -> int: ...


class SpectralBasis(Protocol):
    """A 1-D orthonormal basis with known (or estimated) eigenvalues.

    Implementations cover both closed-form bases (Fourier sine/cosine,
    Chebyshev, Legendre, Hermite, Laguerre) and data-driven bases
    (learned Laplace-Beltrami, graph Laplacian).

    Attributes:
        n_modes: Number of modes ``K`` exposed by the basis.
        eigenvalues: ``[K]`` tensor, sorted ascending.
        null_indices: Indices of modes with ``λ ≈ 0`` — the kernel's
            null space.
    """

    n_modes: int
    eigenvalues: torch.Tensor
    null_indices: tuple[int, ...]

    def evaluate(self, z: torch.Tensor, /) -> torch.Tensor:
        """Evaluate every mode at ``z``. Returns ``[..., K]``."""
        ...  # pragma: no cover  # Protocol method body


class NullSpaceBasis(Protocol):
    """Extra design-matrix columns capturing the operator's null space.

    Concretely: a set of functions ``ξ_i(x)`` such that ``L ξ_i = 0`` for
    the physical operator ``L`` being inverted. Their coefficients are
    learned by the closed-form lstsq solve with no Tikhonov shrinkage,
    so the null component is data-driven rather than shrunk to zero.

    Attributes:
        n_columns: Number of basis columns this null space contributes.
    """

    n_columns: int

    def evaluate(self, z: torch.Tensor, /) -> torch.Tensor:
        """Evaluate every null-space basis function. Returns ``[N, n_columns]``."""
        ...  # pragma: no cover  # Protocol method body


class ExtraLoss(Protocol):
    """Optional regularizer that contributes a scalar term to the training loss.

    Used by :class:`igl.MatryoshkaTrainer` to fold in geometry-specific
    penalties (e.g. the SPD orthogonality penalty from
    :class:`igl.spd.OrthogonalityPenalty`) without coupling the trainer to
    that geometry.

    The trainer calls the extra loss once per batch (subject to
    :attr:`every`), adds ``weight * contribution`` to the task loss, and
    backpropagates through everything in one go. Implementations may return
    ``None`` to skip the contribution for this step (useful when the
    regularizer is only defined for ``k ≥ 2``, for example).

    Attributes:
        weight: Multiplier applied to the returned tensor before adding to
            the task loss.
        every: Call frequency in *batches* (1 = every batch).
    """

    weight: float
    every: int

    def __call__(
        self,
        *,
        encoder: torch.nn.Module,
        x_batch: torch.Tensor,
        gate_mask: torch.Tensor,
        k: int,
        epoch: int,
        batch_idx: int,
    ) -> torch.Tensor | None: ...


__all__ = [
    "ActivationType",
    "ActivationTypeLike",
    "ActivationTypeLiteral",
    "DimensionCurve",
    "EncoderKind",
    "EncoderKindLike",
    "EncoderKindLiteral",
    "EncoderProtocol",
    "ExtraLoss",
    "GraphLaplacianNorm",
    "GraphLaplacianNormLike",
    "GraphLaplacianNormLiteral",
    "LossStrategy",
    "MatryoshkaSampler",
    "NormType",
    "NormTypeLike",
    "NormTypeLiteral",
    "NormalizeMode",
    "NormalizeModeLike",
    "NormalizeModeLiteral",
    "NullSpaceBasis",
    "NullSpaceKind",
    "NullSpaceKindLike",
    "NullSpaceKindLiteral",
    "OperatorFn",
    "OperatorName",
    "OperatorNameLike",
    "OperatorNameLiteral",
    "PreconditionMode",
    "PreconditionModeLike",
    "PreconditionModeLiteral",
    "SamplingMode",
    "SamplingModeLike",
    "SamplingModeLiteral",
    "SchedulerType",
    "SchedulerTypeLike",
    "SchedulerTypeLiteral",
    "SpectralBasis",
    "SpectralKind",
    "SpectralKindLike",
    "SpectralKindLiteral",
]
