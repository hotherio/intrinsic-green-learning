"""Conditional SPD regularization (per-matrix eigenvalue floor).

Some MOABB covariance batches (notably Zhou2016) contain a small fraction of
near-singular trials whose smallest eigenvalue sits at the numerical noise
floor. Pyriemann's iterative Riemannian (Fréchet) mean — used by
``UnsupervisedRiemannianCenter`` — then fails with ``"Matrices must be
positive definite"``. The failure mode is *a noise-floor eigenvalue*, not high
condition number per se (BNCI2014_001 has a higher max condition number yet
never fails), so the fix targets exactly that and nothing else.
"""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUntypedFunctionDecorator=false, reportUnknownArgumentType=false, reportUnknownParameterType=false
# (numpy/sklearn ship without complete stubs; gate the noise at file level.)

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ConditionalTikhonov(BaseEstimator, TransformerMixin):
    """Trace-relative eigenvalue floor, applied per matrix only when needed.

    For each SPD matrix ``C`` with mean eigenvalue ``tr(C)/d``, raise its
    smallest eigenvalue to at least ``eps · tr(C)/d`` by adding a scalar
    diagonal load. A matrix whose minimum eigenvalue already exceeds the floor
    is returned untouched, so well-conditioned datasets (AlexMI, BNCI) are
    **bit-identical** while pathological near-singular trials (Zhou2016) are
    minimally lifted back to numerical positive-definiteness.

    Validated: at ``eps=1e-6`` this fixes Zhou2016's LOSO (≈0.1% of covs
    touched) while touching 0.0% of AlexMI/BNCI2014_001 covs. ``eps<=0``
    disables the floor entirely (exact passthrough).

    Input:  [n_trials, d, d]   SPD covariances
    Output: [n_trials, d, d]   same shape, dtype preserved
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def fit(self, x: NDArray[np.floating], y: object = None) -> ConditionalTikhonov:  # noqa: ARG002
        return self

    def transform(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        arr = np.asarray(x)
        if self.eps <= 0:
            return arr
        cov = arr.astype(np.float64, copy=False)
        ev_min = np.linalg.eigvalsh(cov)[..., 0]
        floor = self.eps * np.trace(cov, axis1=1, axis2=2) / cov.shape[-1]
        bump = np.clip(floor - ev_min, 0.0, None)
        if not np.any(bump > 0.0):
            return arr  # every matrix already above the floor -> exact passthrough
        out = cov + bump[:, None, None] * np.eye(cov.shape[-1])
        return out.astype(arr.dtype, copy=False)
