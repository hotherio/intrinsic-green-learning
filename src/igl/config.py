"""Configuration dataclasses for IGL components.

All configs are stdlib :func:`dataclasses.dataclass` instances with
``frozen=True``, ``slots=True``, ``kw_only=True``: hashable, slotted (no
accidental attribute assignment), and forced keyword-args in constructors so
new fields can be added without breaking call sites.

String-valued fields accept either a :class:`enum.StrEnum` member (the
canonical reference) or the matching string literal — ``__post_init__``
coerces strings to enums so internal code can rely on identity comparisons.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Self, TypeVar, cast

from igl.exceptions import IGLConfigError
from igl.types import (
    ActivationType,
    ActivationTypeLike,
    EncoderKind,
    EncoderKindLike,
    NormalizeMode,
    NormalizeModeLike,
    NormType,
    NormTypeLike,
    NullSpaceKind,
    NullSpaceKindLike,
    OperatorName,
    OperatorNameLike,
    SamplingMode,
    SamplingModeLike,
    SchedulerType,
    SchedulerTypeLike,
    SpectralKind,
    SpectralKindLike,
)

_T = TypeVar("_T", int, float, bool)


def _typed_get(data: Mapping[str, object], key: str, default: _T, /) -> _T:
    """Return ``data[key]`` if present and the same concrete type as ``default``.

    A focused helper for leaf-typed config fields: int, float, bool. The
    return type is inferred from ``default``, so callers don't need to
    annotate or :func:`cast`. For union / nullable / enum fields, keep
    :func:`cast` explicit — those don't have a single ``type`` to match
    against at runtime.

    Raises:
        IGLConfigError: If ``key`` is present with a wrongly-typed value.
            Falling back to ``default`` silently on a type mismatch would
            hide config typos that "happen to parse"; raising surfaces them.
    """
    if key not in data:
        return default
    val = data[key]
    if isinstance(val, type(default)):
        return val
    raise IGLConfigError(
        f"{key!r} must be {type(default).__name__}, got {type(val).__name__}",
    )


def _coerce_operator(
    operator: OperatorNameLike | Sequence[OperatorNameLike],
) -> OperatorName | tuple[OperatorName, ...]:
    if isinstance(operator, OperatorName):
        return operator
    if isinstance(operator, str):
        return OperatorName(operator)
    return tuple(OperatorName(op) for op in operator)


@dataclass(frozen=True, slots=True, kw_only=True)
class EncoderConfig:
    """MLP encoder configuration. Fields mirror :class:`igl.MLPEncoder`."""

    kind: EncoderKindLike = EncoderKind.MLP
    hidden: int | tuple[int, ...] = 256
    depth: int = 2
    norm: NormTypeLike = NormType.LAYER
    activation: ActivationTypeLike = ActivationType.SILU

    def __post_init__(self) -> None:
        # Coerce string forms to enums so downstream code can use identity checks.
        object.__setattr__(self, "kind", EncoderKind(self.kind))
        object.__setattr__(self, "norm", NormType(self.norm))
        object.__setattr__(self, "activation", ActivationType(self.activation))
        # Lists slipping in (e.g. from JSON) become tuples for hashability.
        if not isinstance(self.hidden, int):
            object.__setattr__(self, "hidden", tuple(self.hidden))


@dataclass(frozen=True, slots=True, kw_only=True)
class KernelConfig:
    """Green-kernel configuration. Fields mirror :class:`igl.GreenKernel`."""

    n_anchors: int = 64
    n_scales: int = 4
    operator: OperatorName | tuple[OperatorName, ...] = OperatorName.GAUSSIAN
    sigma_log_range: tuple[float, float] = (-1.5, 1.5)
    anchor_init_std: float = 0.5
    normalize: NormalizeModeLike = NormalizeMode.NW
    null_space: NullSpaceKindLike = NullSpaceKind.NONE
    polynomial_degree: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "operator", _coerce_operator(self.operator))
        object.__setattr__(self, "normalize", NormalizeMode(self.normalize))
        object.__setattr__(self, "null_space", NullSpaceKind(self.null_space))


@dataclass(frozen=True, slots=True, kw_only=True)
class SpectralConfig:
    """Spectral-kernel configuration. Mirrors :class:`igl.spectral.SpectralKernel`.

    When set on :class:`IGLConfig.spectral`, the sklearn wrappers build a
    :class:`igl.spectral.SpectralKernel` (with the listed basis(es)) in
    place of the local :class:`igl.GreenKernel`.
    """

    kind: SpectralKindLike | tuple[SpectralKindLike, ...] = SpectralKind.FOURIER_SINE
    n_modes: int = 16
    n_anchors: int = 64
    null_space: NullSpaceKindLike = NullSpaceKind.NONE
    polynomial_degree: int = 1
    epsilon: float = 1e-4
    anchor_init_std: float = 0.25
    # Learned-LB only:
    k_nn: int = 10
    refresh_every: int = 200

    def __post_init__(self) -> None:
        if isinstance(self.kind, str | SpectralKind):
            object.__setattr__(self, "kind", SpectralKind(self.kind))
        else:
            object.__setattr__(
                self,
                "kind",
                tuple(SpectralKind(k) for k in self.kind),
            )
        object.__setattr__(self, "null_space", NullSpaceKind(self.null_space))


@dataclass(frozen=True, slots=True, kw_only=True)
class MatryoshkaConfig:
    """Matryoshka VP training configuration."""

    epochs: int = 1500
    batch_size: int = 256
    inner_batch_size: int = 4096
    encoder_lr: float = 1e-3
    weight_decay: float | None = None
    source_l2: float = 1e-3
    grad_clip: float = 1.0
    sampling: SamplingModeLike = SamplingMode.UNIFORM
    alpha: float = 1.0
    scheduler: SchedulerTypeLike = SchedulerType.NONE
    early_stop_patience: int | None = 100
    early_stop_min_epochs: int = 200
    noise_std: float = 0.0
    log_every: int = 100
    verbose: bool = False
    sigma_max_diagnostic: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "sampling", SamplingMode(self.sampling))
        object.__setattr__(self, "scheduler", SchedulerType(self.scheduler))


def _operator_to_serial(value: OperatorName | tuple[OperatorName, ...]) -> str | list[str]:
    if isinstance(value, OperatorName):
        return str(value)
    return [str(op) for op in value]


@dataclass(frozen=True, slots=True, kw_only=True)
class IGLConfig:
    """Top-level IGL configuration composing the per-component configs.

    When ``spectral`` is set, the spectral kernel replaces the local Green
    kernel; ``kernel`` is then ignored. When ``spectral`` is ``None``
    (default), the local kernel is used.
    """

    max_dim: int = 16
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    kernel: KernelConfig = field(default_factory=KernelConfig)
    matryoshka: MatryoshkaConfig = field(default_factory=MatryoshkaConfig)
    spectral: SpectralConfig | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dict (suitable for JSON / TOML round-trips).

        StrEnum members serialise to their underlying string; tuples serialise
        to lists for JSON friendliness.
        """
        # The dataclasses are frozen + coerced via __post_init__, so the enum
        # fields are guaranteed to be StrEnum instances at this point. We
        # cast to placate basedpyright (which sees the declared `…Like` type).
        encoder_norm = cast(NormType, self.encoder.norm)
        encoder_activation = cast(ActivationType, self.encoder.activation)
        encoder_kind = cast(EncoderKind, self.encoder.kind)
        kernel_normalize = cast(NormalizeMode, self.kernel.normalize)
        kernel_null = cast(NullSpaceKind, self.kernel.null_space)
        matryoshka_sampling = cast(SamplingMode, self.matryoshka.sampling)
        matryoshka_scheduler = cast(SchedulerType, self.matryoshka.scheduler)

        hidden_serial: int | list[int]
        hidden_serial = self.encoder.hidden if isinstance(self.encoder.hidden, int) else list(self.encoder.hidden)

        return {
            "max_dim": self.max_dim,
            "encoder": {
                "kind": str(encoder_kind),
                "hidden": hidden_serial,
                "depth": self.encoder.depth,
                "norm": str(encoder_norm),
                "activation": str(encoder_activation),
            },
            "kernel": {
                "n_anchors": self.kernel.n_anchors,
                "n_scales": self.kernel.n_scales,
                "operator": _operator_to_serial(self.kernel.operator),
                "sigma_log_range": list(self.kernel.sigma_log_range),
                "anchor_init_std": self.kernel.anchor_init_std,
                "normalize": str(kernel_normalize),
                "null_space": str(kernel_null),
                "polynomial_degree": self.kernel.polynomial_degree,
            },
            "spectral": _spectral_to_dict(self.spectral),
            "matryoshka": {
                "epochs": self.matryoshka.epochs,
                "batch_size": self.matryoshka.batch_size,
                "inner_batch_size": self.matryoshka.inner_batch_size,
                "encoder_lr": self.matryoshka.encoder_lr,
                "weight_decay": self.matryoshka.weight_decay,
                "source_l2": self.matryoshka.source_l2,
                "grad_clip": self.matryoshka.grad_clip,
                "sampling": str(matryoshka_sampling),
                "alpha": self.matryoshka.alpha,
                "scheduler": str(matryoshka_scheduler),
                "early_stop_patience": self.matryoshka.early_stop_patience,
                "early_stop_min_epochs": self.matryoshka.early_stop_min_epochs,
                "noise_std": self.matryoshka.noise_std,
                "log_every": self.matryoshka.log_every,
                "verbose": self.matryoshka.verbose,
                "sigma_max_diagnostic": self.matryoshka.sigma_max_diagnostic,
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object], /) -> Self:
        """Construct from a previously-serialised dict.

        Coerces ``hidden`` (list → tuple), ``sigma_log_range`` (list →
        tuple), and string-valued enum fields back to their canonical types
        via :meth:`EncoderConfig.__post_init__` and friends.
        """
        enc_raw = data.get("encoder", {})
        ker_raw = data.get("kernel", {})
        mat_raw = data.get("matryoshka", {})
        if not isinstance(enc_raw, Mapping):
            raise IGLConfigError("encoder must be a mapping")
        if not isinstance(ker_raw, Mapping):
            raise IGLConfigError("kernel must be a mapping")
        if not isinstance(mat_raw, Mapping):
            raise IGLConfigError("matryoshka must be a mapping")
        max_dim_raw = data.get("max_dim", 16)
        if not isinstance(max_dim_raw, int):
            raise IGLConfigError("max_dim must be an int")

        # `enc_raw` etc. are typed `Mapping[str, object]` after the isinstance
        # narrowing above; pyright loses the str-key narrowing on `.get()` calls
        # inside the helpers, so we cast once at the boundary.
        enc = _make_encoder_config(cast(Mapping[str, object], enc_raw))
        ker = _make_kernel_config(cast(Mapping[str, object], ker_raw))
        mat = _make_matryoshka_config(cast(Mapping[str, object], mat_raw))

        spectral_raw = data.get("spectral", None)
        spectral: SpectralConfig | None
        if spectral_raw is None:
            spectral = None
        elif isinstance(spectral_raw, Mapping):
            spectral = _make_spectral_config(cast(Mapping[str, object], spectral_raw))
        else:
            raise IGLConfigError("spectral must be a mapping or null")

        return cls(max_dim=max_dim_raw, encoder=enc, kernel=ker, matryoshka=mat, spectral=spectral)


def _spectral_to_dict(spectral: SpectralConfig | None) -> dict[str, object] | None:
    if spectral is None:
        return None
    kind = spectral.kind
    kind_serial: str | list[str] = (
        str(cast(SpectralKind, kind)) if isinstance(kind, str | SpectralKind) else [str(SpectralKind(k)) for k in kind]
    )
    null_space = cast(NullSpaceKind, spectral.null_space)
    return {
        "kind": kind_serial,
        "n_modes": spectral.n_modes,
        "n_anchors": spectral.n_anchors,
        "null_space": str(null_space),
        "polynomial_degree": spectral.polynomial_degree,
        "epsilon": spectral.epsilon,
        "anchor_init_std": spectral.anchor_init_std,
        "k_nn": spectral.k_nn,
        "refresh_every": spectral.refresh_every,
    }


def _make_encoder_config(data: Mapping[str, object]) -> EncoderConfig:
    hidden = data.get("hidden", 256)
    if isinstance(hidden, list):
        hidden = tuple(cast(list[int], hidden))
    elif not isinstance(hidden, int | tuple):
        raise IGLConfigError("encoder.hidden must be an int, tuple, or list of ints")
    return EncoderConfig(
        kind=cast(EncoderKindLike, data.get("kind", EncoderKind.MLP)),
        hidden=cast(int | tuple[int, ...], hidden),
        depth=_typed_get(data, "depth", 2),
        norm=cast(NormTypeLike, data.get("norm", NormType.LAYER)),
        activation=cast(ActivationTypeLike, data.get("activation", ActivationType.SILU)),
    )


def _make_kernel_config(data: Mapping[str, object]) -> KernelConfig:
    operator_raw = data.get("operator", OperatorName.GAUSSIAN)
    if isinstance(operator_raw, list):
        operator: OperatorName | tuple[OperatorName, ...] = tuple(OperatorName(op) for op in cast(list[str], operator_raw))
    elif isinstance(operator_raw, OperatorName):
        operator = operator_raw
    elif isinstance(operator_raw, str):
        operator = OperatorName(operator_raw)
    elif isinstance(operator_raw, tuple):
        operator = tuple(OperatorName(op) for op in cast(tuple[str, ...], operator_raw))
    else:
        raise IGLConfigError("kernel.operator must be a string, list, or tuple of strings")

    sigma_raw = data.get("sigma_log_range", (-1.5, 1.5))
    if isinstance(sigma_raw, list):
        sigma_raw = tuple(cast(list[float], sigma_raw))
    if not isinstance(sigma_raw, tuple):
        raise IGLConfigError("kernel.sigma_log_range must be a 2-tuple of floats")
    sigma_pair = cast(tuple[float, float], sigma_raw)

    return KernelConfig(
        n_anchors=_typed_get(data, "n_anchors", 64),
        n_scales=_typed_get(data, "n_scales", 4),
        operator=operator,
        sigma_log_range=sigma_pair,
        anchor_init_std=_typed_get(data, "anchor_init_std", 0.5),
        normalize=cast(NormalizeModeLike, data.get("normalize", NormalizeMode.SOFTMAX)),
        null_space=cast(NullSpaceKindLike, data.get("null_space", NullSpaceKind.NONE)),
        polynomial_degree=_typed_get(data, "polynomial_degree", 1),
    )


def _make_spectral_config(data: Mapping[str, object]) -> SpectralConfig:
    kind_raw = data.get("kind", SpectralKind.FOURIER_SINE)
    kind: SpectralKindLike | tuple[SpectralKindLike, ...]
    if isinstance(kind_raw, list):
        kind = tuple(SpectralKind(k) for k in cast(list[str], kind_raw))
    elif isinstance(kind_raw, tuple):
        kind = tuple(SpectralKind(k) for k in cast(tuple[str, ...], kind_raw))
    elif isinstance(kind_raw, str | SpectralKind):
        kind = SpectralKind(kind_raw)
    else:
        raise IGLConfigError("spectral.kind must be a string, list, or tuple of strings")

    return SpectralConfig(
        kind=kind,
        n_modes=_typed_get(data, "n_modes", 16),
        n_anchors=_typed_get(data, "n_anchors", 64),
        null_space=cast(NullSpaceKindLike, data.get("null_space", NullSpaceKind.NONE)),
        polynomial_degree=_typed_get(data, "polynomial_degree", 1),
        epsilon=_typed_get(data, "epsilon", 1e-4),
        anchor_init_std=_typed_get(data, "anchor_init_std", 0.25),
        k_nn=_typed_get(data, "k_nn", 10),
        refresh_every=_typed_get(data, "refresh_every", 200),
    )


def _make_matryoshka_config(data: Mapping[str, object]) -> MatryoshkaConfig:
    return MatryoshkaConfig(
        epochs=_typed_get(data, "epochs", 1500),
        batch_size=_typed_get(data, "batch_size", 256),
        inner_batch_size=_typed_get(data, "inner_batch_size", 4096),
        encoder_lr=_typed_get(data, "encoder_lr", 1e-3),
        weight_decay=cast("float | None", data.get("weight_decay", None)),
        source_l2=_typed_get(data, "source_l2", 1e-3),
        grad_clip=_typed_get(data, "grad_clip", 1.0),
        sampling=cast(SamplingModeLike, data.get("sampling", SamplingMode.UNIFORM)),
        alpha=_typed_get(data, "alpha", 1.0),
        scheduler=cast(SchedulerTypeLike, data.get("scheduler", SchedulerType.NONE)),
        early_stop_patience=cast("int | None", data.get("early_stop_patience", 100)),
        early_stop_min_epochs=_typed_get(data, "early_stop_min_epochs", 200),
        noise_std=_typed_get(data, "noise_std", 0.0),
        log_every=_typed_get(data, "log_every", 100),
        verbose=_typed_get(data, "verbose", False),
        sigma_max_diagnostic=_typed_get(data, "sigma_max_diagnostic", False),
    )


__all__ = [
    "EncoderConfig",
    "IGLConfig",
    "KernelConfig",
    "MatryoshkaConfig",
    "SpectralConfig",
]
