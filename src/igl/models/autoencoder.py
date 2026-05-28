"""IGL autoencoder — reconstructs the input through the Matryoshka bottleneck.

This is the third leg of the canonical task triple
``d_eff(cls) <= d_eff(reg) <= d_eff(recon)``. The autoencoder trains the
encoder + Green kernel to reproduce the (scaled) ambient input from its
truncated latent, so the discovered effective dimension is the smallest
``k`` at which reconstruction error saturates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.base import TransformerMixin

from igl.config import IGLConfig
from igl.core.loss import MSELoss
from igl.exceptions import IGLConfigError
from igl.models._base import _BaseIGLEstimator
from igl.types import NormalizeModeLike, OperatorNameLike

if TYPE_CHECKING:
    from numpy.typing import NDArray


class IGLAutoencoder(_BaseIGLEstimator[MSELoss], TransformerMixin):
    """Train an IGL model with ``y = x`` (reconstruction).

    The :meth:`fit` method ignores its ``y`` argument and uses ``x`` as both
    input and target.

    Attributes:
        n_features_in_: Ambient input dimension.
        n_outputs_: Equal to ``n_features_in_``.
        module_: The underlying :class:`igl.IGLModule`.
        history_, dimension_curve_, effective_dimension_, scaler_: same as
            :class:`IGLRegressor`.
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
        return int(y.shape[1])

    def _prepare_y(
        self,
        y: NDArray[np.generic],  # noqa: ARG002 — ignored; the target is x_scaled
        *,
        device: torch.device,
        x_scaled: NDArray[np.floating] | None = None,
    ) -> torch.Tensor:
        if x_scaled is None:
            raise IGLConfigError(
                "IGLAutoencoder._prepare_y requires x_scaled (the StandardScaler-transformed input) "
                "to be passed through; this is an internal contract from _BaseIGLEstimator._fit_core.",
            )
        self.n_outputs_: int = int(x_scaled.shape[1])
        return torch.as_tensor(x_scaled, dtype=torch.float32, device=device)

    def fit(self, x: NDArray[np.floating], y: NDArray[np.floating] | None = None) -> IGLAutoencoder:  # noqa: ARG002
        """Fit the autoencoder. ``y`` is ignored; the *scaled* ``x`` is the target.

        Reconstruction is therefore measured in the StandardScaler-scaled
        feature space. :meth:`reconstruct` inverse-transforms back to the
        original feature space for users who need real-space outputs.
        """
        x_arr = np.asarray(x)
        self._fit_core(x_arr, x_arr, validation_fraction=self.validation_fraction)
        return self

    def transform(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return the reconstructed input (in the scaled feature space)."""
        out = self._predict_phi(np.asarray(x)).cpu().numpy()
        return np.asarray(out, dtype=np.float64)

    def reconstruct(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Inverse-transform the reconstruction back to the original feature space."""
        scaled = self.transform(np.asarray(x))
        # sklearn's `inverse_transform` partial stubs.
        unscaled = self.scaler_.inverse_transform(scaled)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        return np.asarray(unscaled, dtype=np.float64)


__all__ = ["IGLAutoencoder"]
