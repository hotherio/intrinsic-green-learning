"""Configuration dataclasses for IGL components.

All configs are stdlib :func:`dataclasses.dataclass` instances with
``frozen=True``, ``slots=True``, ``kw_only=True``: hashable, slotted (no
accidental attribute assignment), and forced keyword-args in constructors so
new fields can be added without breaking call sites.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Self

from igl.types import NormalizeMode, OperatorName, SamplingMode


@dataclass(frozen=True, slots=True, kw_only=True)
class EncoderConfig:
    """MLP encoder configuration. Fields mirror :class:`igl.MLPEncoder`."""

    kind: str = "mlp"
    hidden: int = 256
    depth: int = 2
    norm: str = "layer"
    activation: str = "silu"


@dataclass(frozen=True, slots=True, kw_only=True)
class KernelConfig:
    """Green-kernel configuration. Fields mirror :class:`igl.GreenKernel`."""

    n_anchors: int = 64
    n_scales: int = 4
    operator: OperatorName | tuple[OperatorName, ...] = "gaussian"
    sigma_log_range: tuple[float, float] = (-1.5, 1.5)
    anchor_init_std: float = 0.5
    normalize: NormalizeMode = "softmax"


@dataclass(frozen=True, slots=True, kw_only=True)
class MatryoshkaConfig:
    """Matryoshka VP training configuration."""

    epochs: int = 1500
    batch_size: int = 256
    inner_batch_size: int = 4096
    encoder_lr: float = 1e-3
    weight_decay: float | None = 1e-4
    source_l2: float = 1e-3
    grad_clip: float = 1.0
    sampling: SamplingMode = "uniform"
    alpha: float = 1.0
    scheduler: str = "cosine_warm_restarts"
    early_stop_patience: int | None = 100
    early_stop_min_epochs: int = 200
    noise_std: float = 0.0
    log_every: int = 100
    verbose: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class IGLConfig:
    """Top-level IGL configuration composing the per-component configs."""

    max_dim: int = 16
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    kernel: KernelConfig = field(default_factory=KernelConfig)
    matryoshka: MatryoshkaConfig = field(default_factory=MatryoshkaConfig)

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dict (suitable for JSON / TOML round-trips)."""
        return {
            "max_dim": self.max_dim,
            "encoder": {
                "kind": self.encoder.kind,
                "hidden": self.encoder.hidden,
                "depth": self.encoder.depth,
                "norm": self.encoder.norm,
                "activation": self.encoder.activation,
            },
            "kernel": {
                "n_anchors": self.kernel.n_anchors,
                "n_scales": self.kernel.n_scales,
                "operator": self.kernel.operator,
                "sigma_log_range": self.kernel.sigma_log_range,
                "anchor_init_std": self.kernel.anchor_init_std,
                "normalize": self.kernel.normalize,
            },
            "matryoshka": {
                "epochs": self.matryoshka.epochs,
                "batch_size": self.matryoshka.batch_size,
                "inner_batch_size": self.matryoshka.inner_batch_size,
                "encoder_lr": self.matryoshka.encoder_lr,
                "weight_decay": self.matryoshka.weight_decay,
                "source_l2": self.matryoshka.source_l2,
                "grad_clip": self.matryoshka.grad_clip,
                "sampling": self.matryoshka.sampling,
                "alpha": self.matryoshka.alpha,
                "scheduler": self.matryoshka.scheduler,
                "early_stop_patience": self.matryoshka.early_stop_patience,
                "early_stop_min_epochs": self.matryoshka.early_stop_min_epochs,
                "noise_std": self.matryoshka.noise_std,
                "log_every": self.matryoshka.log_every,
                "verbose": self.matryoshka.verbose,
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object], /) -> Self:
        """Construct from a previously-serialised dict."""
        enc = data.get("encoder", {})
        ker = data.get("kernel", {})
        mat = data.get("matryoshka", {})
        assert isinstance(enc, Mapping), "encoder must be a mapping"
        assert isinstance(ker, Mapping), "kernel must be a mapping"
        assert isinstance(mat, Mapping), "matryoshka must be a mapping"
        max_dim = data.get("max_dim", 16)
        assert isinstance(max_dim, int), "max_dim must be an int"
        # Unpacking Mapping[str, object] into dataclass constructors: we trust
        # the to_dict / from_dict round-trip, so pyright's "argument is Unknown"
        # noise here is suppressed at the call sites.
        return cls(
            max_dim=max_dim,
            encoder=EncoderConfig(**enc),  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
            kernel=KernelConfig(**ker),  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
            matryoshka=MatryoshkaConfig(**mat),  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
        )


__all__ = [
    "EncoderConfig",
    "IGLConfig",
    "KernelConfig",
    "MatryoshkaConfig",
]
