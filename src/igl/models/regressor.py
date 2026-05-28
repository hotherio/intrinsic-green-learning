"""scikit-learn-compatible regressor wrapper around :class:`igl.IGLModule`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.base import RegressorMixin

from igl.config import IGLConfig
from igl.core.loss import MSELoss
from igl.models._base import _BaseIGLEstimator
from igl.types import NormalizeModeLike, OperatorNameLike

if TYPE_CHECKING:
    from numpy.typing import NDArray

_PRED_NDIM_MULTIOUTPUT = 2


class IGLRegressor(_BaseIGLEstimator[MSELoss], RegressorMixin):
    """scikit-learn-compatible regressor.

    Supports scalar and multi-output regression. Targets ``y`` may be a
    1-D array (``[n_samples]``) for scalar regression, or 2-D
    (``[n_samples, n_outputs]``) for multi-output. The output dimension is
    inferred from ``y`` in :meth:`fit`.

    Args mirror :class:`IGLClassifier` except for ``validation_fraction``.

    Attributes:
        n_features_in_: Ambient input dimension seen during fit.
        n_outputs_: Number of output dimensions inferred from ``y``.
        module_: The underlying :class:`igl.IGLModule`.
        history_: :class:`igl.TrainingHistory` from the trainer.
        dimension_curve_: ``{k: mse}`` from :func:`igl.eval_dimension_curve`.
        effective_dimension_: Discovered ``d_eff``.
        scaler_: :class:`sklearn.preprocessing.StandardScaler`.
    """

    validation_fraction: float | None

    def __init__(
        self,
        *,
        max_dim: int = 16,
        n_anchors: int | None = None,
        n_scales: int | None = None,
        operator: OperatorNameLike | None = None,
        normalize: NormalizeModeLike | None = None,
        encoder_hidden: int | tuple[int, ...] | None = None,
        encoder_depth: int | None = None,
        config: IGLConfig | None = None,
        random_state: int | None = None,
        validation_fraction: float | None = 0.2,
    ) -> None:
        super().__init__(
            max_dim=max_dim,
            n_anchors=n_anchors,
            n_scales=n_scales,
            operator=operator,
            normalize=normalize,
            encoder_hidden=encoder_hidden,
            encoder_depth=encoder_depth,
            config=config,
            random_state=random_state,
        )
        self.validation_fraction = validation_fraction

    def _build_loss(self, y: NDArray[np.generic]) -> MSELoss:  # noqa: ARG002
        return MSELoss()

    def _output_dim(self, y: NDArray[np.generic]) -> int:
        return 1 if y.ndim == 1 else int(y.shape[1])

    def _prepare_y(
        self,
        y: NDArray[np.generic],
        *,
        device: torch.device,
        x_scaled: NDArray[np.floating] | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        arr = np.asarray(y, dtype=np.float32)
        self.n_outputs_: int = 1 if arr.ndim == 1 else int(arr.shape[1])
        return torch.as_tensor(arr, dtype=torch.float32, device=device)

    def fit(self, x: NDArray[np.floating], y: NDArray[np.floating]) -> IGLRegressor:
        """Fit the regressor on ``(x, y)``."""
        self._fit_core(np.asarray(x), np.asarray(y), validation_fraction=self.validation_fraction)
        return self

    def predict(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Predict targets for ``x``. Returns 1-D if fit with scalar targets."""
        out = self._predict_phi(np.asarray(x)).cpu().numpy()
        if self.n_outputs_ == 1 and out.ndim == _PRED_NDIM_MULTIOUTPUT and out.shape[1] == 1:
            out = out.reshape(-1)
        return np.asarray(out, dtype=np.float64)

    # `score` is inherited from ``sklearn.base.RegressorMixin`` — returns the
    # coefficient of determination R². We don't override it.


__all__ = ["IGLRegressor"]
