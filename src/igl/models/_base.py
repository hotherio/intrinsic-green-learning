"""Shared infrastructure for scikit-learn-compatible IGL estimators.

The three concrete estimators (:class:`igl.IGLClassifier`,
:class:`igl.IGLRegressor`, :class:`igl.IGLAutoencoder`) all wrap the same
:class:`igl.IGLModule` + :class:`igl.MatryoshkaTrainer` pair. This module
factors out the shared scaffolding so each estimator stays small:

- :class:`_BaseIGLEstimator`: stores hyperparameters, builds the IGL config,
  runs the trainer, and exposes the ``dimension_curve_`` and
  ``effective_dimension_`` post-fit properties.
- :func:`_to_torch`: numpy → torch conversion with consistent device + dtype.
- :func:`_check_is_fitted`: emits :class:`igl.IGLNotFittedError` (instead of
  sklearn's generic NotFittedError) when an unfitted estimator is queried.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Generic, TypeVar

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from igl.config import EncoderConfig, IGLConfig, MatryoshkaConfig
from igl.core.trainer import MatryoshkaTrainer, TrainingHistory
from igl.exceptions import IGLNotFittedError
from igl.matryoshka.dimension_curve import detect_elbow, eval_dimension_curve
from igl.nn.module import IGLModule
from igl.spectral._build import build_kernel_null_space, build_spectral_kernel
from igl.types import DimensionCurve, LossStrategy, NormalizeModeLike, OperatorNameLike

_LossT = TypeVar("_LossT", bound=LossStrategy)

_EXPECTED_X_NDIM = 2


def _to_torch(x: NDArray[np.floating], *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Wrap a numpy array as a torch tensor on the requested device."""
    return torch.as_tensor(x, dtype=dtype, device=device)


def _check_is_fitted(estimator: BaseEstimator, attribute: str) -> None:
    """Raise :class:`IGLNotFittedError` if ``attribute`` is absent on ``estimator``."""
    if not hasattr(estimator, attribute):
        raise IGLNotFittedError(
            f"{type(estimator).__name__} is not fitted yet; call .fit(X, y) first.",
        )


class _BaseIGLEstimator(BaseEstimator, Generic[_LossT]):
    """Common base for the sklearn-compatible IGL estimators.

    Subclasses must:

    - Implement :meth:`_build_loss` returning a :class:`LossStrategy`.
    - Implement :meth:`_prepare_y` returning a 1-D or 2-D torch tensor used as
      the training target.
    - Set :attr:`_output_dim` after preprocessing (e.g., number of classes).

    Args:
        max_dim: Latent dimension ``d_max``.
        n_anchors: Anchor count override (``None`` → use config or default 64).
        n_scales: Kernel scale count override (``None`` → use config or 4).
        operator: Kernel operator override (``None`` → use config or
            ``OperatorName.GAUSSIAN``).
        normalize: Φ normalization override (``None`` → use config or
            ``NormalizeMode.SOFTMAX``).
        encoder_hidden: Encoder ``hidden`` shorthand (``int`` or tuple of
            per-layer widths). ``None`` defers to the encoder config.
        encoder_depth: Encoder depth shorthand.
        config: Optional top-level :class:`IGLConfig`; explicit kwargs above
            override its values.
        random_state: Optional integer seed. When set, the PyTorch RNG is
            seeded inside a :func:`torch.random.fork_rng` scope (so the
            caller's global RNG state is preserved across the fit); NumPy
            sampling uses a local :class:`np.random.RandomState` derived from
            the same seed.
    """

    # Declared on the class so `clone()` finds them via get_params.
    max_dim: int
    n_anchors: int | None
    n_scales: int | None
    operator: OperatorNameLike | None
    normalize: NormalizeModeLike | None
    encoder_hidden: int | tuple[int, ...] | None
    encoder_depth: int | None
    config: IGLConfig | None
    random_state: int | None

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
    ) -> None:
        # sklearn's contract: __init__ must ONLY store the kwargs. No logic.
        self.max_dim = max_dim
        self.n_anchors = n_anchors
        self.n_scales = n_scales
        self.operator = operator
        self.normalize = normalize
        self.encoder_hidden = encoder_hidden
        self.encoder_depth = encoder_depth
        self.config = config
        self.random_state = random_state

    # ---- internal helpers (subclass extension points) ----

    def _build_loss(self, y: NDArray[np.generic]) -> _LossT:  # pragma: no cover  # abstract
        raise NotImplementedError

    def _prepare_y(
        self,
        y: NDArray[np.generic],
        *,
        device: torch.device,
        x_scaled: NDArray[np.floating] | None = None,
    ) -> torch.Tensor:  # pragma: no cover  # abstract
        # ``x_scaled`` is the StandardScaler-transformed input; the autoencoder
        # uses it as its target so reconstruction is measured in the scaled
        # space. Other estimators ignore it.
        raise NotImplementedError

    def _resolve_encoder_config(self) -> EncoderConfig:
        base_encoder = self.config.encoder if self.config is not None else EncoderConfig()
        if self.encoder_hidden is None and self.encoder_depth is None:
            return base_encoder
        # Build a new EncoderConfig with overrides applied.
        hidden = self.encoder_hidden if self.encoder_hidden is not None else base_encoder.hidden
        depth = self.encoder_depth if self.encoder_depth is not None else base_encoder.depth
        return EncoderConfig(
            kind=base_encoder.kind,
            hidden=hidden,
            depth=depth,
            norm=base_encoder.norm,
            activation=base_encoder.activation,
        )

    @contextmanager
    def _local_rng(self) -> Generator[None]:
        """Scope torch RNG mutations to this block so callers' RNG state survives the fit.

        When :attr:`random_state` is set, fork torch's global RNG, seed the
        fork, and restore the parent on exit. Numpy needs no equivalent: the
        only direct ``np.random`` use in :meth:`_fit_core` already creates a
        local :class:`np.random.RandomState`.
        """
        if self.random_state is None:
            yield
            return
        with torch.random.fork_rng():  # pyright: ignore[reportUnknownMemberType]
            torch.manual_seed(self.random_state)  # pyright: ignore[reportUnknownMemberType]
            yield

    def _build_module(self, *, input_dim: int, output_dim: int) -> IGLModule:
        # Spectral path: build a SpectralKernel from the config and hand it
        # to IGLModule as a pre-built kernel. Local-kernel path: IGLModule
        # builds GreenKernel from per-field kwargs as before.
        kernel: object = None
        if self.config is not None and self.config.spectral is not None:
            kernel = build_spectral_kernel(
                latent_dim=self.max_dim,
                config=self.config.spectral,
            )
        elif self.config is not None:
            kernel = build_kernel_null_space(
                latent_dim=self.max_dim,
                config=self.config.kernel,
            )

        return IGLModule(
            input_dim=input_dim,
            max_dim=self.max_dim,
            output_dim=output_dim,
            n_anchors=self.n_anchors,
            n_scales=self.n_scales,
            operator=self.operator,
            encoder_config=self._resolve_encoder_config(),
            normalize=self.normalize,
            config=None,  # encoder_config already encodes the overrides
            kernel=kernel,  # type: ignore[arg-type]
        )

    def _matryoshka_config(self) -> MatryoshkaConfig:
        return self.config.matryoshka if self.config is not None else MatryoshkaConfig()

    # ---- public sklearn-compatible surface ----

    def _fit_core(
        self,
        x: NDArray[np.floating],
        y: NDArray[np.generic],
        *,
        validation_fraction: float | None = None,
    ) -> _LossT:
        """Shared fit machinery used by the concrete estimators.

        Stores ``self.scaler_``, ``self.module_``, ``self.history_``,
        ``self.dimension_curve_``, ``self.effective_dimension_``,
        ``self.n_features_in_``, and returns the loss strategy used for
        future prediction.
        """
        if x.ndim != _EXPECTED_X_NDIM:
            x = np.asarray(x).reshape(len(x), -1)
        n_features = x.shape[1]
        self.n_features_in_: int = n_features

        self.scaler_: StandardScaler = StandardScaler()
        # sklearn's transform methods have partial stubs; the return is an
        # ndarray we re-cast on the next line.
        x_scaled = self.scaler_.fit_transform(x)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

        device = torch.device("cpu")
        x_scaled_arr = np.asarray(x_scaled, dtype=np.float32)
        x_tensor = _to_torch(x_scaled_arr, device=device)
        y_tensor = self._prepare_y(y, device=device, x_scaled=x_scaled_arr)
        loss = self._build_loss(y)

        if validation_fraction is not None and 0.0 < validation_fraction < 1.0:
            n_val = max(1, int(len(x_tensor) * validation_fraction))
            perm = np.random.RandomState(self.random_state or 0).permutation(len(x_tensor))
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]
            x_train, x_val = x_tensor[train_idx], x_tensor[val_idx]
            y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]
        else:
            x_train, y_train = x_tensor, y_tensor
            x_val = y_val = None

        # Determine the output_dim per task.
        output_dim = self._output_dim(y)
        # Scope torch RNG mutations to module construction + training so the
        # caller's global RNG state is preserved.
        with self._local_rng():
            self.module_: IGLModule = self._build_module(input_dim=n_features, output_dim=output_dim)
            trainer = MatryoshkaTrainer(loss=loss, config=self._matryoshka_config())
            self.history_: TrainingHistory = trainer.fit(self.module_, x_train, y_train, x_val=x_val, y_val=y_val)

        # Compute dimension curve on whatever data we have for evaluation.
        x_curve = x_val if x_val is not None else x_train
        y_curve = y_val if y_val is not None else y_train
        self.dimension_curve_: DimensionCurve = eval_dimension_curve(
            self.module_,
            x_curve,
            y_curve,
            loss=loss,
            source_l2=self._matryoshka_config().source_l2,
        )
        self.effective_dimension_: int = detect_elbow(self.dimension_curve_)
        return loss

    def _output_dim(self, y: NDArray[np.generic]) -> int:  # pragma: no cover  # abstract
        raise NotImplementedError

    def _predict_phi(self, x: NDArray[np.floating]) -> torch.Tensor:
        _check_is_fitted(self, "module_")
        device = next(self.module_.parameters()).device
        # sklearn's `.transform` partial stubs: the runtime return is an ndarray.
        x_scaled = self.scaler_.transform(np.asarray(x, dtype=np.float32))  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        x_tensor = _to_torch(np.asarray(x_scaled, dtype=np.float32), device=device)
        self.module_.eval()
        with torch.no_grad():
            return self.module_(x_tensor)


__all__ = ["_BaseIGLEstimator", "_check_is_fitted", "_to_torch"]
