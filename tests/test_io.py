"""Tests for :mod:`igl.io` (igl.save / igl.load)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from igl import (
    IGLAutoencoder,
    IGLClassifier,
    IGLConfig,
    IGLConfigError,
    IGLDistiller,
    IGLModule,
    IGLRegressor,
    IGLSerializationError,
    KernelConfig,
    MatryoshkaConfig,
    load,
    read_provenance,
    save,
)
from igl.data import embed_in_high_dim, make_moons
from igl.io import PreprocessingState, Provenance

_QUICK = IGLConfig(max_dim=4, matryoshka=MatryoshkaConfig(epochs=15, batch_size=64, early_stop_patience=None))
_SMALL_KERNEL = KernelConfig(n_anchors=8, n_scales=2)
_BARE = IGLConfig(max_dim=4, kernel=_SMALL_KERNEL)


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray]:
    x_2d, y = make_moons(200, noise=0.08, seed=42)
    return embed_in_high_dim(x_2d, target_dim=8, seed=123).numpy(), y.numpy()


def test_module_round_trip_is_bit_identical(tmp_path: Path, data: tuple[np.ndarray, np.ndarray]) -> None:
    x, _ = data
    torch.manual_seed(0)
    module = IGLModule(input_dim=8, max_dim=4, output_dim=2, n_anchors=8, n_scales=2)
    save(module, tmp_path / "m.pt", config=_BARE)
    loaded = load(tmp_path / "m.pt")
    assert isinstance(loaded, IGLModule)
    x_t = torch.from_numpy(x.astype(np.float32))
    with torch.no_grad():
        assert torch.equal(module(x_t), loaded(x_t))


def test_bare_module_without_config_raises(tmp_path: Path) -> None:
    module = IGLModule(input_dim=4, max_dim=2, output_dim=1, n_anchors=4, n_scales=2)
    with pytest.raises(IGLSerializationError, match="requires config="):
        save(module, tmp_path / "m.pt")


@pytest.mark.parametrize("kind", ["classifier", "regressor", "autoencoder", "distiller"])
def test_estimator_round_trip_predictions_match(kind: str, tmp_path: Path, data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = data
    if kind == "classifier":
        est = IGLClassifier(max_dim=4, config=_QUICK, random_state=0).fit(x, y)
        reference = est.predict_proba(x)
    elif kind == "regressor":
        est = IGLRegressor(max_dim=4, config=_QUICK, random_state=0).fit(x, y.astype(float))
        reference = est.predict(x)
    elif kind == "autoencoder":
        est = IGLAutoencoder(max_dim=4, config=_QUICK, random_state=0).fit(x)
        reference = est.reconstruct(x)
    else:
        est = IGLDistiller(max_dim=4, config=_QUICK, random_state=0).fit(x)
        reference = est.reconstruct(x)
    save(est, tmp_path / "e.pt")
    loaded = load(tmp_path / "e.pt")
    assert type(loaded) is type(est)
    if kind == "classifier":
        result = loaded.predict_proba(x)  # pyright: ignore[reportAttributeAccessIssue]
    elif kind == "regressor":
        result = loaded.predict(x)  # pyright: ignore[reportAttributeAccessIssue]
    else:
        result = loaded.reconstruct(x)  # pyright: ignore[reportAttributeAccessIssue]
    assert np.allclose(result, reference, atol=1e-6)


def test_estimator_extras_survive(tmp_path: Path, data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = data
    est = IGLClassifier(max_dim=4, config=_QUICK, random_state=0).fit(x, y)
    save(est, tmp_path / "e.pt")
    loaded = load(tmp_path / "e.pt")
    assert isinstance(loaded, IGLClassifier)
    assert list(loaded.classes_) == list(est.classes_)
    assert loaded.effective_dimension_ == est.effective_dimension_
    assert loaded.dimension_curve_ == est.dimension_curve_
    assert loaded.history_.train_loss == est.history_.train_loss


def test_estimator_save_rejects_explicit_config(tmp_path: Path, data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = data
    est = IGLRegressor(max_dim=4, config=_QUICK, random_state=0).fit(x, y.astype(float))
    with pytest.raises(IGLConfigError, match="derived from fitted estimators"):
        save(est, tmp_path / "e.pt", config=_QUICK)


def test_unfitted_estimator_raises(tmp_path: Path) -> None:
    with pytest.raises(IGLSerializationError, match="not fitted"):
        save(IGLRegressor(max_dim=4), tmp_path / "e.pt")


def test_quick_profile_is_refused_by_default(tmp_path: Path) -> None:
    module = IGLModule(input_dim=4, max_dim=2, output_dim=1, n_anchors=8, n_scales=2)
    save(module, tmp_path / "m.pt", config=IGLConfig(max_dim=2, kernel=_SMALL_KERNEL), provenance=Provenance(profile="quick"))
    with pytest.raises(IGLSerializationError, match="allow_quick"):
        load(tmp_path / "m.pt")
    assert isinstance(load(tmp_path / "m.pt", allow_quick=True), IGLModule)


def test_read_provenance(tmp_path: Path) -> None:
    module = IGLModule(input_dim=4, max_dim=2, output_dim=1, n_anchors=4, n_scales=2)
    save(module, tmp_path / "m.pt", config=_QUICK, provenance=Provenance(seed=7, epochs=15))
    provenance = read_provenance(tmp_path / "m.pt")
    assert provenance["seed"] == 7
    assert provenance["epochs"] == 15
    assert provenance["profile"] == "full"
    assert isinstance(provenance["package_version"], str)


def test_tampered_payload_raises(tmp_path: Path) -> None:
    torch.save({"schema_version": 99}, tmp_path / "bad.pt")
    with pytest.raises(IGLSerializationError, match="missing keys"):
        load(tmp_path / "bad.pt")
    torch.save([1, 2, 3], tmp_path / "worse.pt")
    with pytest.raises(IGLSerializationError, match="expected a dict"):
        load(tmp_path / "worse.pt")


def test_bare_module_preprocessing_round_trip(tmp_path: Path) -> None:
    from igl.whitening import TargetWhitener

    module = IGLModule(input_dim=4, max_dim=2, output_dim=4, n_anchors=8, n_scales=2)
    whitener = TargetWhitener().fit(torch.randn(30, 4))
    save(
        module,
        tmp_path / "m.pt",
        config=IGLConfig(max_dim=2, kernel=_SMALL_KERNEL),
        preprocessing=PreprocessingState(mu=torch.zeros(4), sd=1.5, y_scale=whitener.y_scale_, whitener=whitener),
    )
    assert isinstance(load(tmp_path / "m.pt"), IGLModule)
