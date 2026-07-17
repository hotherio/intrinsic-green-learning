"""Versioned checkpoint save/load for modules and estimators.

One file, one schema: :func:`save` writes a ``torch.save`` payload holding
only tensors and JSON-safe primitives, so :func:`load` reads it with
``torch.load(..., weights_only=True)``. The payload carries the resolved
:class:`igl.IGLConfig`, the module ``state_dict``, the preprocessing
constants (scaler statistics, scalar input stats, or a fitted
:class:`igl.whitening.TargetWhitener`), estimator extras (params, classes,
history, dimension curve), and provenance (package and torch versions,
seed, epochs, quick/full profile).
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch

from igl.config import IGLConfig
from igl.exceptions import IGLConfigError, IGLSerializationError
from igl.io._module import build_module_from_config
from igl.io._schema import SCHEMA_VERSION, validate_payload
from igl.models.autoencoder import IGLAutoencoder
from igl.models.classifier import IGLClassifier
from igl.models.distiller import IGLDistiller
from igl.models.regressor import IGLRegressor
from igl.nn.module import IGLModule
from igl.whitening.whitener import TargetWhitener

__all__ = ["PreprocessingState", "Profile", "Provenance", "load", "read_provenance", "save"]

_ESTIMATOR_KINDS: dict[type, str] = {
    IGLClassifier: "classifier",
    IGLRegressor: "regressor",
    IGLAutoencoder: "autoencoder",
    IGLDistiller: "distiller",
}
_KIND_CLASSES = {kind: cls for cls, kind in _ESTIMATOR_KINDS.items()}

type Saveable = IGLModule | IGLClassifier | IGLRegressor | IGLAutoencoder | IGLDistiller
type _Estimator = IGLClassifier | IGLRegressor | IGLAutoencoder | IGLDistiller


class Profile(StrEnum):
    """Fit profile recorded in provenance: quick fits must not be silently reused."""

    QUICK = "quick"
    FULL = "full"


type ProfileLike = Profile | Literal["quick", "full"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Provenance:
    """User-supplied provenance stamped into the checkpoint.

    Attributes:
        seed: Seed used for the fit, if any.
        epochs: Epoch budget of the fit, if recorded.
        profile: ``Profile.FULL`` (default) or ``Profile.QUICK``; quick
            checkpoints are refused by :func:`load` unless ``allow_quick``.
    """

    seed: int | None = None
    epochs: int | None = None
    profile: ProfileLike = Profile.FULL


@dataclass(frozen=True, slots=True, kw_only=True)
class PreprocessingState:
    """Preprocessing constants for bare-module checkpoints.

    Estimator checkpoints derive these from fitted attributes; bare
    :class:`igl.IGLModule` saves carry them explicitly.
    """

    mu: torch.Tensor | None = None
    sd: float | None = None
    y_scale: float | None = None
    whitener: TargetWhitener | None = None


def save(
    obj: Saveable,
    path: str | Path,
    *,
    config: IGLConfig | None = None,
    preprocessing: PreprocessingState | None = None,
    provenance: Provenance | None = None,
) -> None:
    """Write a schema-v1 checkpoint for a module or fitted estimator.

    Args:
        obj: A bare :class:`igl.IGLModule` or a fitted estimator.
        path: Destination file.
        config: Required for bare modules (a module does not retain its
            config); forbidden for estimators (theirs is derived).
        preprocessing: Optional constants for bare modules; forbidden for
            estimators.
        provenance: Optional provenance; versions and timestamp are stamped
            automatically.

    Raises:
        IGLSerializationError: For unfitted estimators or a bare module
            without ``config``.
        IGLConfigError: When estimator-derived fields are passed explicitly.
    """
    provenance = provenance or Provenance()
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "package_version": _package_version(),
            "torch_version": str(torch.__version__),
            "created": datetime.now(tz=UTC).isoformat(),
            "seed": provenance.seed,
            "epochs": provenance.epochs,
            "profile": str(Profile(provenance.profile)),
        },
    }
    if isinstance(obj, IGLModule):
        if config is None:
            raise IGLSerializationError(
                "saving a bare IGLModule requires config=: the module does not retain its architecture config"
            )
        pre = preprocessing or PreprocessingState()
        payload.update(
            kind="igl_module",
            dims=_dims(obj),
            config=config.to_dict(),
            state_dict={key: value.cpu() for key, value in obj.state_dict().items()},
            preprocessing={
                "mu": pre.mu.cpu() if pre.mu is not None else None,
                "sd": pre.sd,
                "y_scale": pre.y_scale,
                "whitener": pre.whitener.state_dict() if pre.whitener is not None else None,
            },
            estimator=None,
        )
    else:
        if config is not None or preprocessing is not None:
            raise IGLConfigError("config= and preprocessing= are derived from fitted estimators; do not pass them")
        if not hasattr(obj, "module_"):
            raise IGLSerializationError("estimator is not fitted; fit() before save()")
        payload.update(
            kind=_ESTIMATOR_KINDS[type(obj)],
            dims=_dims(obj.module_),
            config=obj._resolved_config().to_dict(),  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            state_dict={key: value.cpu() for key, value in obj.module_.state_dict().items()},
            preprocessing=_estimator_preprocessing(obj),
            estimator=_estimator_extras(obj),
        )
    torch.save(payload, Path(path))


def load(path: str | Path, *, allow_quick: bool = False, map_location: str | torch.device = "cpu") -> Saveable:
    """Read a checkpoint written by :func:`save`.

    Args:
        path: Checkpoint file.
        allow_quick: Load ``profile="quick"`` checkpoints; refused by
            default so a quick fit is never silently mistaken for a full one.
        map_location: Device for the loaded tensors.

    Returns:
        The reconstructed module or estimator, ready for inference.

    Raises:
        IGLSerializationError: On schema mismatch or a refused quick profile.
    """
    payload = validate_payload(torch.load(Path(path), map_location=map_location, weights_only=True))
    provenance = cast(dict[str, object], payload["provenance"])
    if provenance.get("profile") == str(Profile.QUICK) and not allow_quick:
        raise IGLSerializationError("checkpoint has profile='quick'; pass allow_quick=True to load it anyway")

    dims = cast(dict[str, object], payload["dims"])
    config = IGLConfig.from_dict(cast(dict[str, object], payload["config"]))
    module = build_module_from_config(
        config,
        input_dim=cast(int, dims["input_dim"]),
        output_dim=cast(int, dims["output_dim"]),
    )
    module.load_state_dict(cast(dict[str, torch.Tensor], payload["state_dict"]), strict=True)
    module.eval()

    if payload["kind"] == "igl_module":
        return module
    return _rebuild_estimator(
        kind=cast(str, payload["kind"]),
        module=module,
        config=config,
        preprocessing=cast(dict[str, object], payload["preprocessing"]),
        extras=cast(dict[str, object], payload["estimator"]),
    )


def read_provenance(path: str | Path) -> dict[str, object]:
    """Read only the provenance block of a checkpoint."""
    payload = validate_payload(torch.load(Path(path), map_location="cpu", weights_only=True))
    return cast(dict[str, object], payload["provenance"])


def _package_version() -> str:
    from importlib.metadata import version

    return version("intrinsic-green-learning")


def _dims(module: IGLModule) -> dict[str, object]:
    return {
        "input_dim": module.input_dim,
        "max_dim": module.max_dim,
        "output_dim": module.output_dim,
        "normalize_input": module.normalize_input,
    }


def _estimator_preprocessing(obj: _Estimator) -> dict[str, object]:
    out: dict[str, object] = {
        "scaler_mean": None,
        "scaler_scale": None,
        "input_mean": None,
        "input_std": None,
        "whitener": None,
    }
    scaler = getattr(obj, "scaler_", None)
    if scaler is not None:
        out["scaler_mean"] = torch.as_tensor(scaler.mean_, dtype=torch.float64)
        out["scaler_scale"] = torch.as_tensor(scaler.scale_, dtype=torch.float64)
    if isinstance(obj, IGLDistiller):
        out["input_mean"] = obj.input_mean_.cpu()
        out["input_std"] = obj.input_std_
        out["whitener"] = obj.whitener_.state_dict()
    return out


def _estimator_extras(obj: _Estimator) -> dict[str, object]:
    params: dict[str, object] = {}
    for key, value in obj.get_params(deep=False).items():  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if key in {"config", "metric"}:
            continue  # config is stored resolved; the metric tensor is stored below
        params[cast(str, key)] = str(value) if isinstance(value, torch.device) else value
    history = getattr(obj, "history_", None)
    classes = getattr(obj, "classes_", None)
    return {
        "params": params,
        "metric": getattr(obj, "metric", None),
        "classes": None if classes is None else [int(c) for c in classes],
        "n_outputs": getattr(obj, "n_outputs_", None),
        "history": None
        if history is None
        else {
            "train_loss": list(history.train_loss),
            "val_loss": list(history.val_loss),
            "val_metric": list(history.val_metric),
            "truncation_k": list(history.truncation_k),
            "best_epoch": history.best_epoch,
            "best_metric": history.best_metric,
            "stopped_epoch": history.stopped_epoch,
            "early_stopped": history.early_stopped,
        },
        "dimension_curve": dict(getattr(obj, "dimension_curve_", {})),
        "effective_dimension": getattr(obj, "effective_dimension_", None),
        "n_features_in": getattr(obj, "n_features_in_", None),
    }


def _rebuild_estimator(
    *,
    kind: str,
    module: IGLModule,
    config: IGLConfig,
    preprocessing: dict[str, object],
    extras: dict[str, object],
) -> Saveable:
    from sklearn.preprocessing import StandardScaler

    from igl.core.trainer import TrainingHistory

    cls = _KIND_CLASSES[kind]
    params = dict(cast(dict[str, object], extras["params"]))
    metric = extras.get("metric")
    if kind == "distiller":
        params["metric"] = metric
    estimator = cls(config=config, **params)  # pyright: ignore[reportArgumentType]

    estimator.module_ = module
    if extras.get("n_features_in") is not None:
        estimator.n_features_in_ = cast(int, extras["n_features_in"])
    history = cast(dict[str, object] | None, extras.get("history"))
    if history is not None:
        estimator.history_ = TrainingHistory(
            train_loss=cast(list[float], history["train_loss"]),
            val_loss=cast(list[float], history["val_loss"]),
            val_metric=cast(list[float], history["val_metric"]),
            truncation_k=cast(list[float], history["truncation_k"]),
            best_epoch=cast(int | None, history["best_epoch"]),
            best_metric=cast(float | None, history["best_metric"]),
            stopped_epoch=cast(int | None, history["stopped_epoch"]),
            early_stopped=cast(bool, history["early_stopped"]),
        )
    curve = cast(dict[int, float], extras.get("dimension_curve") or {})
    if curve:
        estimator.dimension_curve_ = curve
    if extras.get("effective_dimension") is not None:
        estimator.effective_dimension_ = cast(int, extras["effective_dimension"])

    if preprocessing.get("scaler_mean") is not None:
        scaler = StandardScaler()
        scaler.mean_ = cast(torch.Tensor, preprocessing["scaler_mean"]).numpy()
        scaler.scale_ = cast(torch.Tensor, preprocessing["scaler_scale"]).numpy()
        scaler.var_ = scaler.scale_**2
        scaler.n_features_in_ = len(scaler.mean_)  # pyright: ignore[reportAttributeAccessIssue]
        estimator.scaler_ = scaler  # pyright: ignore[reportAttributeAccessIssue]
    if isinstance(estimator, IGLDistiller):
        estimator.input_mean_ = cast(torch.Tensor, preprocessing["input_mean"])
        estimator.input_std_ = cast(float, preprocessing["input_std"])
        estimator.whitener_ = TargetWhitener.from_state_dict(cast(dict[str, torch.Tensor], preprocessing["whitener"]))
    if extras.get("classes") is not None and isinstance(estimator, IGLClassifier):
        estimator.classes_ = np.asarray(cast(list[int], extras["classes"]))
    if extras.get("n_outputs") is not None and isinstance(estimator, IGLRegressor):
        estimator.n_outputs_ = cast(int, extras["n_outputs"])
    return estimator
