"""Conditional Riemannian dimensionality reduction for SPD covariances.

High-channel "whale" datasets (PhysionetMI/Cho2017 d=64, Schirrmeister2017
d=128) make the AIRM reconstruction both numerically unstable (κ≈1e6 → the
``C^{-1/2}`` sandwich overflows to NaN, IGL diverges ~epoch 22) and expensive
(d³ eigh, ~5.5h/fold at d=128). The Matryoshka effective-dimension theory
reports k_eff = 8–24 on these datasets, so the full covariance is not needed
for accuracy: reducing to ``target_dim`` (default 32 >> k_eff) tames both the
divergence and the d³ cost while preserving the Riemannian geometry (a
congruence transform keeps SPD-ness).
"""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUntypedFunctionDecorator=false, reportUnknownArgumentType=false, reportUnknownParameterType=false
# (numpy/pyriemann/sklearn ship without complete stubs; gate the noise at file level.)

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from igl.preprocessing._pyriemann import require_pyriemann

require_pyriemann()

from pyriemann.spatialfilters import Whitening  # noqa: E402

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ConditionalDimReduce(BaseEstimator, TransformerMixin):
    """Whiten + reduce SPD covariances to ``target_dim``, only when d exceeds it.

    A no-op for ``d <= target_dim`` (low-d datasets stay bit-identical); for
    high-d whales it fits an unsupervised Riemannian whitening filter on the
    training covariances (leakage-free under LOSO — ``fit`` sees the train fold
    only) and projects to ``target_dim`` components. ``target_dim <= 0``
    disables reduction entirely (full-d passthrough).

    Input:  [n_trials, d, d]   SPD covariances
    Output: [n_trials, k, k]   k = min(d, target_dim)
    """

    def __init__(self, target_dim: int = 32, metric: str = "euclid") -> None:
        self.target_dim = target_dim
        self.metric = metric

    def fit(self, x: NDArray[np.floating], y: object = None) -> ConditionalDimReduce:  # noqa: ARG002
        d = np.asarray(x).shape[-1]
        if self.target_dim > 0 and d > self.target_dim:
            self.reducer_ = Whitening(metric=self.metric, dim_red={"n_components": self.target_dim}).fit(x)
        else:
            self.reducer_ = None  # identity passthrough
        return self

    def transform(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.reducer_ is None:
            return x
        return self.reducer_.transform(np.asarray(x))
