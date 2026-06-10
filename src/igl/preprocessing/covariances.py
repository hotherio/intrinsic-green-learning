"""Sklearn-compatible covariance-estimator switcher (LWF vs sample).

``AutoCovariances`` wraps :class:`pyriemann.estimation.Covariances` with a
trial-length-based heuristic for switching between Ledoit-Wolf and
sample covariance. The threshold ``T = 500`` is empirically derived on
MOABB motor-imagery datasets at 128 Hz resampling — see
``alex-eeg-igl/report/third_summary_report.tex`` §6.5 and
``alex-eeg-igl/MAINTAINER_MEMO_lwf_tikh_rules.md``.
"""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUntypedFunctionDecorator=false, reportUnknownArgumentType=false
# (pyriemann and sklearn ship without stubs; gate the noise at file level.)

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Literal

# Fail fast with an actionable hint when pyriemann is missing.
from igl.preprocessing._pyriemann import require_pyriemann

require_pyriemann()

from pyriemann.estimation import Covariances  # noqa: E402
from sklearn.base import BaseEstimator, TransformerMixin  # noqa: E402

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class CovarianceEstimator(StrEnum):
    """Pyriemann estimator identifier accepted by :class:`AutoCovariances`."""

    AUTO = "auto"
    LWF = "lwf"
    COV = "cov"


CovarianceEstimatorLiteral = Literal["auto", "lwf", "cov"]
"""Literal companion of :class:`CovarianceEstimator`."""

type CovarianceEstimatorLike = CovarianceEstimator | CovarianceEstimatorLiteral

_DEFAULT_T_THRESHOLD: int = 500
_EXPECTED_NDIM: int = 3


class AutoCovariances(BaseEstimator, TransformerMixin):
    """Choose Ledoit-Wolf or sample covariance automatically.

    Heuristic derived empirically on MOABB motor-imagery datasets at 128 Hz
    (see ``alex-eeg-igl`` §6.5): per-trial sample covariance is statistically
    rank-marginal when the trial has fewer than ~500 time samples, so LWF
    shrinkage helps; above that threshold sample covariance is well-
    estimated and unshrunk covariance preserves discriminative spectrum
    structure that LWF would flatten.

    Decision is made once at :meth:`fit` time based on the training
    batch's last dimension, and applied consistently to every subsequent
    :meth:`transform` call. Mixing trial lengths within a pipeline is
    unsupported (standard sklearn convention).

    Args:
        T_threshold: Trial length below which LWF is selected. The
            empirical 500 default was tuned at 128 Hz on EEG motor
            imagery; modalities at other sampling rates may need to
            override. See the memo's "Out of scope" section.
        force: Override the heuristic. ``"auto"`` (default) applies the
            heuristic; ``"lwf"`` / ``"cov"`` act as plain
            :class:`pyriemann.estimation.Covariances` passthrough.

    Attributes:
        estimator_: The pyriemann estimator name (``"lwf"`` or ``"cov"``)
            actually selected at fit time.
    """

    T_threshold: int
    force: CovarianceEstimatorLike

    def __init__(
        self,
        T_threshold: int = _DEFAULT_T_THRESHOLD,  # noqa: N803
        force: CovarianceEstimatorLike = CovarianceEstimator.AUTO,
    ) -> None:
        self.T_threshold = T_threshold
        self.force = force

    def _pick_estimator(self, x: NDArray[np.floating]) -> str:
        forced = CovarianceEstimator(self.force)
        if forced is not CovarianceEstimator.AUTO:
            return forced.value
        if x.ndim != _EXPECTED_NDIM:
            msg = f"Expected [N, d, T] raw signals, got shape {x.shape}"
            raise ValueError(msg)
        t = x.shape[-1]
        return CovarianceEstimator.LWF.value if t < self.T_threshold else CovarianceEstimator.COV.value

    def fit(self, x: NDArray[np.floating], y: NDArray[np.generic] | None = None) -> AutoCovariances:
        """Pick the estimator from ``x.shape[-1]`` and fit the inner ``Covariances``."""
        self.estimator_: str = self._pick_estimator(x)
        self._cov: Covariances = Covariances(estimator=self.estimator_).fit(x, y)
        return self

    def transform(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute the SPD batch from raw signals via the fitted estimator."""
        return self._cov.transform(x)

    def fit_transform(  # type: ignore[override]
        self,
        x: NDArray[np.floating],
        y: NDArray[np.generic] | None = None,
        **fit_params: object,  # noqa: ARG002
    ) -> NDArray[np.floating]:
        return self.fit(x, y).transform(x)


__all__ = [
    "AutoCovariances",
    "CovarianceEstimator",
    "CovarianceEstimatorLike",
    "CovarianceEstimatorLiteral",
]
