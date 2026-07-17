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
    SpectralConfig,
)
from igl.core.encoder import LinearEncoder, MLPEncoder
from igl.core.kernel import GreenKernel
from igl.core.loss import CrossEntropyLoss, MSELoss
from igl.core.normalization import normalize_phi
from igl.core.solver import direct_solve_weights
from igl.core.trainer import EpochStats, MatryoshkaTrainer, TrainingHistory
from igl.device import get_device
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
    GraphLaplacianNorm,
    GraphLaplacianNormLike,
    GraphLaplacianNormLiteral,
    LossStrategy,
    MatryoshkaSampler,
    NormalizeMode,
    NormalizeModeLike,
    NormalizeModeLiteral,
    NormType,
    NormTypeLike,
    NormTypeLiteral,
    NullSpaceBasis,
    NullSpaceKind,
    NullSpaceKindLike,
    NullSpaceKindLiteral,
    OperatorFn,
    OperatorName,
    OperatorNameLike,
    OperatorNameLiteral,
    PreconditionMode,
    PreconditionModeLike,
    PreconditionModeLiteral,
    SamplingMode,
    SamplingModeLike,
    SamplingModeLiteral,
    SchedulerType,
    SchedulerTypeLike,
    SchedulerTypeLiteral,
    SpectralBasis,
    SpectralKind,
    SpectralKindLike,
    SpectralKindLiteral,
)

try:
    __version__ = version("intrinsic-green-learning")
except PackageNotFoundError:  # pragma: no cover  # only triggers in non-installed dev tree
    __version__ = "0.0.0"


def make_igl_airm(latent_dim: int, **kwargs: object) -> object:
    """Recommended IGL-AIRM pipeline factory with the maintainer-memo defaults.

    Composes :class:`igl.preprocessing.AutoCovariances` and
    :class:`igl.spd.IGLReconSPDClassifier` into a single sklearn
    pipeline:

    1. ``AutoCovariances`` — picks Ledoit-Wolf or sample covariance at
       fit time based on the trial length ``T = X.shape[-1]``
       (threshold 500 — see ``alex-eeg-igl/report``).
    2. ``IGLReconSPDClassifier`` — Tikhonov ε=1e-6 preconditioning on
       every input SPD, then the standard two-stage AIRM-reconstruction
       + sklearn-readout pipeline.

    Args:
        latent_dim: Side length ``d`` of the SPD covariances — controls
            the reconstruction target dimensionality.
        **kwargs: Forwarded to :class:`IGLReconSPDClassifier`. Override
            the Tikhonov default with ``precondition="none"`` or any
            other :class:`PreconditionMode` value.

    Returns:
        A fitted-pipeline-shaped object (``sklearn.pipeline.Pipeline``)
        ready for ``.fit(X_raw, y)``. The pipeline expects ``X_raw`` of
        shape ``[N, d, T]`` (raw signals, the standard MOABB layout).

    Raises:
        IGLDependencyError: When the ``[eeg]`` extra (which ships
            ``pyriemann``) is not installed.
    """
    # Lazy imports keep ``import igl`` light for users who never touch
    # the eeg-extra path.
    from sklearn.pipeline import make_pipeline  # noqa: PLC0415  # pyright: ignore[reportUnknownVariableType]

    from igl.preprocessing import AutoCovariances  # noqa: PLC0415
    from igl.spd import IGLReconSPDClassifier  # noqa: PLC0415

    return make_pipeline(
        AutoCovariances(),
        IGLReconSPDClassifier(latent_dim=latent_dim, **kwargs),  # type: ignore[arg-type]
    )


__all__ = [
    # Version
    "__version__",
    # Configs (frozen dataclasses)
    "EncoderConfig",
    "IGLConfig",
    "KernelConfig",
    "MatryoshkaConfig",
    "SpectralConfig",
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
    "EpochStats",
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
    # Device
    "get_device",
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
    # Exceptions
    "IGLConfigError",
    "IGLConvergenceError",
    "IGLDependencyError",
    "IGLError",
    "IGLNotFittedError",
    # Factory
    "make_igl_airm",
]
