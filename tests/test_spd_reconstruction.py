"""Tests for ``igl.spd.IGLReconSPDClassifier`` and the synthetic SPD generator."""

import warnings

import numpy as np
import pytest
import torch

import igl
from igl import IGLConfig, IGLConfigError, IGLNotFittedError, MatryoshkaConfig
from igl.data import make_spd_dataset
from igl.spd import IGLReconSPDClassifier, LogEigVectorizer


def _fast_config(epochs: int = 12) -> IGLConfig:
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
def spd_dataset() -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(0)
    np.random.seed(0)
    x, y = make_spd_dataset(120, d=4, n_classes=2, class_separation=1.0, seed=42)
    vec = LogEigVectorizer().fit(x.numpy()).transform(x.numpy())
    return np.asarray(vec, dtype=np.float64), y.numpy()


def test_make_spd_dataset_shapes() -> None:
    x, y = make_spd_dataset(30, d=3, n_classes=3, seed=1)
    assert x.shape == (30, 3, 3)
    assert y.shape == (30,)
    assert set(y.tolist()) == {0, 1, 2}


def test_make_spd_dataset_rejects_invalid_args() -> None:
    with pytest.raises(IGLConfigError, match="n_samples"):
        make_spd_dataset(0, d=3)
    with pytest.raises(IGLConfigError, match="d"):
        make_spd_dataset(10, d=0)
    with pytest.raises(IGLConfigError, match="n_classes"):
        make_spd_dataset(10, d=3, n_classes=1)


def test_make_spd_outputs_are_symmetric_positive_definite() -> None:
    x, _ = make_spd_dataset(15, d=4, seed=2)
    sym_err = (x - x.transpose(-1, -2)).abs().max().item()
    assert sym_err < 1e-4
    eigvals = torch.linalg.eigvalsh(x)  # pyright: ignore[reportUnknownMemberType]
    assert (eigvals > 0).all()


def test_recon_classifier_fits_and_predicts(spd_dataset: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = spd_dataset
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        clf = IGLReconSPDClassifier(
            latent_dim=4,
            max_dim=6,
            n_anchors=12,
            n_scales=2,
            encoder_hidden=32,
            random_state=0,
            config=_fast_config(),
        ).fit(x, y)
    assert hasattr(clf, "module_")
    assert hasattr(clf, "readout_")
    assert hasattr(clf, "dimension_curve_")
    preds = clf.predict(x)
    assert preds.shape == y.shape


def test_recon_classifier_predict_proba(spd_dataset: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = spd_dataset
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        clf = IGLReconSPDClassifier(
            latent_dim=4,
            max_dim=6,
            n_anchors=12,
            n_scales=2,
            random_state=0,
            config=_fast_config(),
        ).fit(x, y)
    probs = clf.predict_proba(x)
    assert probs.shape == (len(x), 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_recon_classifier_with_orthogonality(spd_dataset: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = spd_dataset
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        clf = IGLReconSPDClassifier(
            latent_dim=4,
            max_dim=6,
            n_anchors=12,
            n_scales=2,
            random_state=0,
            config=_fast_config(),
            orthogonality_weight=0.1,
            orthogonality_every=5,
        ).fit(x, y)
    assert clf.predict(x).shape == y.shape


def test_recon_classifier_rejects_input_dim_mismatch() -> None:
    x = np.random.RandomState(0).randn(20, 6).astype(np.float32)
    y = np.random.RandomState(0).randint(0, 2, 20)
    clf = IGLReconSPDClassifier(latent_dim=4, max_dim=4, n_anchors=8, n_scales=2, config=_fast_config(epochs=2))
    # d=4 → expected vec_dim = 10; but x has 6 features.
    with pytest.raises(IGLConfigError, match="latent_dim=4"):
        clf.fit(x, y)


def test_recon_classifier_rejects_single_class() -> None:
    x_spd, _ = make_spd_dataset(20, d=3, seed=0)
    x = LogEigVectorizer().fit(x_spd.numpy()).transform(x_spd.numpy())
    y = np.zeros(20, dtype=np.int64)
    clf = IGLReconSPDClassifier(latent_dim=3, max_dim=4, n_anchors=8, n_scales=2, config=_fast_config(epochs=2))
    with pytest.raises(IGLConfigError, match="classes"):
        clf.fit(x, y)


def test_recon_classifier_rejects_negative_latent_dim() -> None:
    with pytest.raises(IGLConfigError, match="latent_dim"):
        IGLReconSPDClassifier(latent_dim=0)


def test_recon_classifier_not_fitted_error() -> None:
    clf = IGLReconSPDClassifier(latent_dim=4)
    with pytest.raises(IGLNotFittedError):
        clf.predict(np.zeros((3, 10), dtype=np.float32))


def test_extra_loss_plugs_into_trainer() -> None:
    """End-to-end: an ExtraLoss is invoked by MatryoshkaTrainer.fit."""
    from igl.spd import OrthogonalityPenalty  # noqa: PLC0415

    torch.manual_seed(0)
    np.random.seed(0)
    x = torch.randn(60, 8)
    y = (x.norm(dim=-1) > x.norm(dim=-1).median()).long()

    module = igl.IGLModule(input_dim=8, max_dim=4, output_dim=2, n_anchors=8, n_scales=2)
    trainer = igl.MatryoshkaTrainer(
        loss=igl.CrossEntropyLoss(n_classes=2),
        config=igl.MatryoshkaConfig(
            epochs=4,
            batch_size=20,
            inner_batch_size=60,
            scheduler="none",
            early_stop_patience=None,
            verbose=False,
        ),
    )
    penalty = OrthogonalityPenalty(weight=0.05, every=1)
    history = trainer.fit(module, x, y, extra_losses=[penalty])
    assert len(history.train_loss) == 4  # noqa: PLR2004
