"""Unit tests for ``igl.preprocessing.AutoCovariances``.

Skipped if ``pyriemann`` is not installed (the ``[eeg]`` extra is optional).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyriemann")

from pyriemann.estimation import Covariances  # noqa: E402

from igl.preprocessing import AutoCovariances, CovarianceEstimator  # noqa: E402


def _raw_signals(n: int = 6, d: int = 4, t: int = 384, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d, t)).astype(np.float64)


def test_autocov_picks_lwf_when_T_short() -> None:  # noqa: N802
    """Trial length T=384 < 500 → LWF selected."""
    x = _raw_signals(t=384)
    ac = AutoCovariances().fit(x)
    assert ac.estimator_ == CovarianceEstimator.LWF.value


def test_autocov_picks_cov_when_T_long() -> None:  # noqa: N802
    """Trial length T=600 ≥ 500 → sample covariance selected."""
    x = _raw_signals(t=600)
    ac = AutoCovariances().fit(x)
    assert ac.estimator_ == CovarianceEstimator.COV.value


def test_autocov_force_lwf_overrides_heuristic() -> None:
    """force='lwf' picks LWF regardless of T."""
    x = _raw_signals(t=600)
    ac = AutoCovariances(force="lwf").fit(x)
    assert ac.estimator_ == CovarianceEstimator.LWF.value


def test_autocov_force_cov_overrides_heuristic() -> None:
    """force='cov' picks sample-cov regardless of T."""
    x = _raw_signals(t=200)
    ac = AutoCovariances(force="cov").fit(x)
    assert ac.estimator_ == CovarianceEstimator.COV.value


def test_autocov_force_matches_pyriemann_covariances_element_wise() -> None:
    """With force='lwf', the output equals pyriemann's Covariances('lwf') exactly."""
    x = _raw_signals(t=300)
    ours = AutoCovariances(force="lwf").fit_transform(x)
    theirs = Covariances(estimator="lwf").fit_transform(x)
    np.testing.assert_array_equal(ours, theirs)


def test_autocov_rejects_wrong_ndim() -> None:
    """An [N, T] input (missing the d axis) raises ValueError."""
    rng = np.random.default_rng(0)
    x_2d = rng.standard_normal((6, 384))
    with pytest.raises(ValueError, match=r"\[N, d, T\]"):
        AutoCovariances().fit(x_2d)


def test_autocov_custom_T_threshold() -> None:  # noqa: N802
    """A custom T_threshold flips the heuristic at the requested boundary."""
    x = _raw_signals(t=384)
    # threshold=300 → 384 ≥ 300 → cov
    ac = AutoCovariances(T_threshold=300).fit(x)
    assert ac.estimator_ == CovarianceEstimator.COV.value


def test_autocov_transform_output_shape() -> None:
    """Output is ``[N, d, d]`` SPD matrices."""
    x = _raw_signals(n=5, d=4, t=200)
    out = AutoCovariances().fit_transform(x)
    assert out.shape == (5, 4, 4)
    np.testing.assert_allclose(out, out.transpose(0, 2, 1), atol=1e-8)
