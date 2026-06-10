"""Optional preprocessing helpers (raw signal → SPD covariances).

Submodules require the ``[eeg]`` extra (``pyriemann``). Importing them
without that extra installed raises :class:`igl.IGLDependencyError` with
an actionable install hint.

Currently:

- :class:`AutoCovariances` — sklearn-compatible Ledoit-Wolf / sample
  covariance switcher, chosen at fit time based on the trial length
  ``T = X.shape[-1]``.
"""

from igl.preprocessing._pyriemann import require_pyriemann
from igl.preprocessing.covariances import (
    AutoCovariances,
    CovarianceEstimator,
    CovarianceEstimatorLike,
    CovarianceEstimatorLiteral,
)

__all__ = [
    "AutoCovariances",
    "CovarianceEstimator",
    "CovarianceEstimatorLike",
    "CovarianceEstimatorLiteral",
    "require_pyriemann",
]
