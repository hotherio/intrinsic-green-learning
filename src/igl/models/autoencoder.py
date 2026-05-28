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

from igl.core.loss import MSELoss
from igl.models._base import _BaseIGLEstimator

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
        operator: object = None,
        normalize: object = None,
        encoder_hidden: int | tuple[int, ...] | None = None,
        encoder_depth: int | None = None,
        config: object = None,
        random_state: int | None = None,
        validation_fraction: float | None = 0.2,
    ) -> None:
        super().__init__(
            max_dim=max_dim,
            n_anchors=n_anchors,
            n_scales=n_scales,
            operator=operator,  # type: ignore[arg-type]
            normalize=normalize,  # type: ignore[arg-type]
            encoder_hidden=encoder_hidden,
            encoder_depth=encoder_depth,
            config=config,  # type: ignore[arg-type]
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
        assert x_scaled is not None, "IGLAutoencoder requires x_scaled to be passed through"
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
