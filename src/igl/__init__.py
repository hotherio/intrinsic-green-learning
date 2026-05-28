"""Intrinsic Green Learning: task-conditioned intrinsic-dimensionality discovery.

The public surface is intentionally flat: import everything from ``igl`` directly.
Subpackages (``igl.spd``, ``igl.contrib``, ``igl.viz``, ``igl.data.eeg``) stay
namespaced because they either require optional extras or carry weaker stability
promises.
"""

from importlib.metadata import PackageNotFoundError, version

from igl.config import (
    EncoderConfig,
    IGLConfig,
    KernelConfig,
    MatryoshkaConfig,
)
from igl.core.encoder import LinearEncoder, MLPEncoder
from igl.core.kernel import GreenKernel
from igl.core.loss import CrossEntropyLoss, MSELoss
from igl.core.normalization import normalize_phi
from igl.core.solver import direct_solve_weights
from igl.core.trainer import MatryoshkaTrainer, TrainingHistory
from igl.exceptions import (
    IGLConfigError,
    IGLConvergenceError,
    IGLDependencyError,
    IGLError,
    IGLNotFittedError,
)
from igl.kernels._registry import (
    Operator,
    get_operator,
    list_operators,
    register_operator,
)
from igl.matryoshka.dimension_curve import detect_elbow, eval_dimension_curve
from igl.matryoshka.sampler import PowerLawSampler, UniformSampler
from igl.metrics.dimension import DimensionComparison, compare_d_eff, d_eff_from_curve
from igl.models.autoencoder import IGLAutoencoder
from igl.models.classifier import IGLClassifier
from igl.models.regressor import IGLRegressor
from igl.nn.module import IGLModule
from igl.types import (
    ActivationType,
    ActivationTypeLike,
    ActivationTypeLiteral,
    DimensionCurve,
    EncoderKind,
    EncoderKindLike,
    EncoderKindLiteral,
    EncoderProtocol,
    LossStrategy,
    MatryoshkaSampler,
    NormalizeMode,
    NormalizeModeLike,
    NormalizeModeLiteral,
    NormType,
    NormTypeLike,
    NormTypeLiteral,
    OperatorFn,
    OperatorName,
    OperatorNameLike,
    OperatorNameLiteral,
    SamplingMode,
    SamplingModeLike,
    SamplingModeLiteral,
    SchedulerType,
    SchedulerTypeLike,
    SchedulerTypeLiteral,
)

try:
    __version__ = version("intrinsic-green-learning")
except PackageNotFoundError:  # pragma: no cover  # only triggers in non-installed dev tree
    __version__ = "0.0.0"


__all__ = [
    # Version
    "__version__",
    # Configs (frozen dataclasses)
    "EncoderConfig",
    "IGLConfig",
    "KernelConfig",
    "MatryoshkaConfig",
    # Core building blocks
    "GreenKernel",
    "IGLModule",
    "LinearEncoder",
    "MLPEncoder",
    "normalize_phi",
    # sklearn-compatible models
    "IGLAutoencoder",
    "IGLClassifier",
    "IGLRegressor",
    # Metrics
    "DimensionComparison",
    "compare_d_eff",
    # Training
    "CrossEntropyLoss",
    "MSELoss",
    "MatryoshkaTrainer",
    "TrainingHistory",
    "direct_solve_weights",
    # Matryoshka / dimension discovery
    "PowerLawSampler",
    "UniformSampler",
    "d_eff_from_curve",
    "detect_elbow",
    "eval_dimension_curve",
    # Kernel registry
    "Operator",
    "get_operator",
    "list_operators",
    "register_operator",
    # Types / Protocols / Enums
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
    # Exceptions
    "IGLConfigError",
    "IGLConvergenceError",
    "IGLDependencyError",
    "IGLError",
    "IGLNotFittedError",
]
