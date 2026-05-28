"""Intrinsic Green Learning: task-conditioned intrinsic-dimensionality discovery.

The public surface is intentionally flat: import everything from ``igl`` directly.
Subpackages (``igl.spd``, ``igl.contrib``, ``igl.viz``, ``igl.data.eeg``) stay
namespaced because they either require optional extras or carry weaker stability
promises.
"""

from importlib.metadata import PackageNotFoundError, version

from igl.exceptions import (
    IGLConfigError,
    IGLConvergenceError,
    IGLDependencyError,
    IGLError,
    IGLNotFittedError,
)

try:
    __version__ = version("intrinsic-green-learning")
except PackageNotFoundError:  # pragma: no cover  # only triggers in non-installed dev tree
    __version__ = "0.0.0"


__all__ = [
    # Version
    "__version__",
    # Exceptions
    "IGLConfigError",
    "IGLConvergenceError",
    "IGLDependencyError",
    "IGLError",
    "IGLNotFittedError",
]
