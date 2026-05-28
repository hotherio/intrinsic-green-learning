"""Tests for the sklearn-compatible IGL estimators."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

import igl
from igl import (
    IGLAutoencoder,
    IGLClassifier,
    IGLConfig,
    IGLConfigError,
    IGLNotFittedError,
    IGLRegressor,
    MatryoshkaConfig,
)
from igl.data import embed_in_high_dim, make_moons, make_swiss_roll


def _fast_config(epochs: int = 8) -> IGLConfig:
    return IGLConfig(
        matryoshka=MatryoshkaConfig(
            epochs=epochs,
            batch_size=32,
            inner_batch_size=120,
            scheduler="none",
            early_stop_patience=None,
            verbose=False,
        ),
    )


@pytest.fixture
def moons_data() -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(0)
    np.random.seed(0)
    x_2d, y = make_moons(120, noise=0.1, seed=42)
    x = embed_in_high_dim(x_2d, target_dim=10, seed=123)
    return x.numpy(), y.numpy()


@pytest.fixture
def swiss_data() -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(0)
    np.random.seed(0)
    x, params = make_swiss_roll(120, seed=42)
    return x.numpy(), params.numpy()


# ----- IGLClassifier -----


def test_classifier_fits_and_predicts(moons_data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = moons_data
    clf = IGLClassifier(
        max_dim=6,
        n_anchors=16,
        n_scales=2,
        encoder_hidden=32,
        random_state=0,
        config=_fast_config(),
    ).fit(x, y)
    preds = clf.predict(x)
    assert preds.shape == y.shape
    assert set(np.unique(preds).tolist()) <= set(np.unique(y).tolist())
    assert hasattr(clf, "classes_")
    assert hasattr(clf, "module_")
    assert hasattr(clf, "dimension_curve_")
    assert hasattr(clf, "effective_dimension_")


def test_classifier_predict_proba_sums_to_one(moons_data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = moons_data
    clf = IGLClassifier(
        max_dim=4,
        n_anchors=12,
        n_scales=2,
        random_state=0,
        config=_fast_config(),
    ).fit(x, y)
    probs = clf.predict_proba(x)
    assert probs.shape == (len(x), 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_classifier_score_returns_accuracy(moons_data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = moons_data
    clf = IGLClassifier(
        max_dim=4,
        n_anchors=12,
        n_scales=2,
        encoder_hidden=32,
        random_state=0,
        config=_fast_config(epochs=15),
    ).fit(x, y)
    score = clf.score(x, y)
    assert 0.0 <= score <= 1.0


def test_classifier_raises_when_unfitted_and_predicts() -> None:
    clf = IGLClassifier(max_dim=4, n_anchors=8, n_scales=2)
    with pytest.raises(IGLNotFittedError):
        clf.predict(np.zeros((3, 4), dtype=np.float32))


def test_classifier_rejects_single_class() -> None:
    x = np.random.RandomState(0).randn(20, 6).astype(np.float32)
    y = np.zeros(20, dtype=np.int64)
    clf = IGLClassifier(max_dim=4, n_anchors=8, n_scales=2, config=_fast_config(epochs=2))
    with pytest.raises(IGLConfigError, match="classes"):
        clf.fit(x, y)


def test_classifier_handles_string_labels(moons_data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = moons_data
    y_str = np.where(y == 0, "neg", "pos")
    clf = IGLClassifier(
        max_dim=4,
        n_anchors=12,
        n_scales=2,
        random_state=0,
        config=_fast_config(),
    ).fit(x, y_str)
    preds = clf.predict(x[:5])
    assert set(preds.tolist()) <= {"neg", "pos"}


def test_classifier_get_params_and_clone() -> None:
    from sklearn.base import clone  # noqa: PLC0415

    clf = IGLClassifier(max_dim=8, n_anchors=16, random_state=7)
    params = clf.get_params()
    assert params["max_dim"] == 8
    assert params["random_state"] == 7
    clf_clone = clone(clf)
    assert clf_clone.max_dim == 8


def test_classifier_accepts_1d_input(moons_data: tuple[np.ndarray, np.ndarray]) -> None:
    """1-D input gets reshaped to (n, 1) internally."""
    x, y = moons_data
    # Take a single feature and pass it as a 1-D array.
    x_1d = x[:, 0]
    clf = IGLClassifier(
        max_dim=2,
        n_anchors=8,
        n_scales=2,
        random_state=0,
        config=_fast_config(epochs=3),
    ).fit(x_1d, y)
    assert clf.n_features_in_ == 1


def test_classifier_runs_without_validation_split(moons_data: tuple[np.ndarray, np.ndarray]) -> None:
    """validation_fraction=None uses all data for training."""
    x, y = moons_data
    clf = IGLClassifier(
        max_dim=4,
        n_anchors=12,
        n_scales=2,
        random_state=0,
        validation_fraction=None,
        config=_fast_config(),
    ).fit(x, y)
    assert hasattr(clf, "dimension_curve_")


def test_classifier_uses_encoder_hidden_shorthand(moons_data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = moons_data
    clf = IGLClassifier(
        max_dim=4,
        n_anchors=12,
        n_scales=2,
        encoder_hidden=(48, 24),
        random_state=0,
        config=_fast_config(),
    ).fit(x, y)
    # Inspect the inner MLPEncoder shape.
    inner = clf.module_.encoder
    if isinstance(inner, torch.nn.Sequential):
        inner = inner[1]
    assert hasattr(inner, "hidden_widths")
    assert inner.hidden_widths == (48, 24)  # type: ignore[attr-defined]


# ----- IGLRegressor -----


def test_regressor_scalar_target(swiss_data: tuple[np.ndarray, np.ndarray]) -> None:
    x, params = swiss_data
    y = params[:, 0]  # scalar regression
    reg = IGLRegressor(
        max_dim=4,
        n_anchors=12,
        n_scales=2,
        random_state=0,
        config=_fast_config(epochs=20),
    ).fit(x, y)
    preds = reg.predict(x)
    assert preds.ndim == 1
    assert preds.shape == y.shape


def test_regressor_multi_output(swiss_data: tuple[np.ndarray, np.ndarray]) -> None:
    x, params = swiss_data
    reg = IGLRegressor(
        max_dim=4,
        n_anchors=12,
        n_scales=2,
        random_state=0,
        config=_fast_config(epochs=20),
    ).fit(x, params)
    preds = reg.predict(x)
    assert preds.shape == params.shape
    assert reg.n_outputs_ == 2


def test_regressor_score_returns_r2(swiss_data: tuple[np.ndarray, np.ndarray]) -> None:
    x, params = swiss_data
    reg = IGLRegressor(
        max_dim=4,
        n_anchors=12,
        n_scales=2,
        random_state=0,
        config=_fast_config(epochs=20),
    ).fit(x, params[:, 0])
    score = reg.score(x, params[:, 0])
    assert isinstance(score, float)


# ----- IGLAutoencoder -----


def test_autoencoder_fits_and_reconstructs(swiss_data: tuple[np.ndarray, np.ndarray]) -> None:
    x, _ = swiss_data
    ae = IGLAutoencoder(
        max_dim=4,
        n_anchors=12,
        n_scales=2,
        random_state=0,
        config=_fast_config(epochs=20),
    ).fit(x)
    assert ae.n_outputs_ == x.shape[1]
    recon = ae.reconstruct(x[:5])
    assert recon.shape == (5, x.shape[1])


def test_autoencoder_transform_returns_scaled_space(swiss_data: tuple[np.ndarray, np.ndarray]) -> None:
    x, _ = swiss_data
    ae = IGLAutoencoder(
        max_dim=4,
        n_anchors=12,
        n_scales=2,
        random_state=0,
        config=_fast_config(),
    ).fit(x)
    out = ae.transform(x[:3])
    assert out.shape == (3, x.shape[1])


# ----- compare_d_eff -----


def test_compare_d_eff_returns_dimension_comparison(
    moons_data: tuple[np.ndarray, np.ndarray],
    swiss_data: tuple[np.ndarray, np.ndarray],
) -> None:
    x_moons, y_moons = moons_data
    x_swiss, params = swiss_data

    clf = IGLClassifier(
        max_dim=4,
        n_anchors=12,
        n_scales=2,
        encoder_hidden=32,
        random_state=0,
        config=_fast_config(epochs=20),
    ).fit(x_moons, y_moons)
    reg = IGLRegressor(
        max_dim=4,
        n_anchors=12,
        n_scales=2,
        encoder_hidden=32,
        random_state=0,
        config=_fast_config(epochs=20),
    ).fit(x_swiss, params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        report = igl.compare_d_eff(cls=clf.dimension_curve_, reg=reg.dimension_curve_)
    assert "cls" in report.d_effs
    assert "reg" in report.d_effs
    assert isinstance(report.hierarchy_holds, bool)


def test_d_eff_from_curve_matches_detect_elbow() -> None:
    curve = {1: 1.0, 2: 0.1, 3: 0.09}
    assert igl.d_eff_from_curve(curve) == igl.detect_elbow(curve)
