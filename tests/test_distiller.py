"""Tests for :class:`igl.IGLDistiller`."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from sklearn.base import clone

from igl import IGLConfigError, IGLDistiller, IGLNotFittedError
from igl.data import embed_in_high_dim, make_moons
from igl.whitening import fisher_pullback


@pytest.fixture
def states() -> np.ndarray:
    x_2d, _ = make_moons(300, noise=0.05, seed=42)
    return embed_in_high_dim(x_2d, target_dim=10, seed=123).numpy()


def _quick(**kwargs: object) -> IGLDistiller:
    from igl import IGLConfig, MatryoshkaConfig

    config = IGLConfig(max_dim=4, matryoshka=MatryoshkaConfig(epochs=30, batch_size=64, early_stop_patience=None))
    return IGLDistiller(max_dim=4, config=config, random_state=0, **kwargs)  # pyright: ignore[reportArgumentType]


def test_distiller_fits_and_sets_post_fit_attributes(states: np.ndarray) -> None:
    distiller = _quick().fit(states)
    assert distiller.n_features_in_ == 10
    assert distiller.module_.max_dim == 4
    assert distiller.whitener_.is_fitted
    assert len(distiller.history_.train_loss) > 0
    assert set(distiller.dimension_curve_) == {1, 2, 3, 4}
    assert 1 <= distiller.effective_dimension_ <= 4


def test_distiller_validation_split_is_on_by_default(states: np.ndarray) -> None:
    distiller = _quick().fit(states)
    assert len(distiller.history_.val_loss) > 0


def test_distiller_project_shapes_and_truncation(states: np.ndarray) -> None:
    distiller = _quick().fit(states)
    full = distiller.project(states)
    assert full.shape == (300, 4)
    truncated = distiller.project(states, k=2)
    assert truncated.shape == (300, 2)
    assert np.allclose(truncated, full[:, :2])


def test_distiller_reconstruct_returns_original_space(states: np.ndarray) -> None:
    distiller = _quick().fit(states)
    recon = distiller.reconstruct(states)
    assert recon.shape == states.shape
    assert np.isfinite(recon).all()
    # The moons manifold is 2-D: a 4-wide chart should reconstruct much
    # better than predicting the mean.
    mse = float(np.mean((recon - states) ** 2))
    baseline = float(np.mean((states - states.mean(axis=0)) ** 2))
    assert mse < baseline


def test_distiller_reconstruct_tight_budget_is_worse_or_equal(states: np.ndarray) -> None:
    distiller = _quick().fit(states)
    full = float(np.mean((distiller.reconstruct(states) - states) ** 2))
    tight = float(np.mean((distiller.reconstruct(states, k=1) - states) ** 2))
    assert tight >= full - 1e-6


def test_distiller_k_out_of_range_raises(states: np.ndarray) -> None:
    distiller = _quick().fit(states)
    with pytest.raises(IGLConfigError, match="k must be in"):
        distiller.project(states, k=9)


def test_distiller_unfitted_raises(states: np.ndarray) -> None:
    with pytest.raises(IGLNotFittedError):
        _quick().project(states)


def test_distiller_get_params_and_clone_round_trip() -> None:
    metric = torch.eye(10)
    distiller = IGLDistiller(max_dim=4, metric=metric, clamp=1e-5, random_state=7, device="cpu")
    params = distiller.get_params()
    assert params["clamp"] == 1e-5
    assert params["device"] == "cpu"
    assert params["metric"] is metric
    cloned = clone(distiller)
    assert cloned.random_state == 7
    assert cloned.metric is not None
    assert torch.equal(cloned.metric, metric)


def test_fisher_metric_beats_identity_on_downstream_kl(states: np.ndarray) -> None:
    """The acceptance check: the metric changes what a tight budget keeps."""
    generator = torch.Generator().manual_seed(0)
    states_t = torch.from_numpy(states.astype(np.float32))
    # A head that barely reads the highest-variance directions.
    w = torch.randn(24, 10, generator=generator) / 10**0.5
    order = states_t.var(dim=0).argsort(descending=True)
    w[:, order[:3]] *= 0.05

    def downstream_kl(recon: np.ndarray) -> float:
        p = F.softmax(states_t @ w.T, dim=-1)
        log_q = F.log_softmax(torch.from_numpy(recon.astype(np.float32)) @ w.T, dim=-1)
        return float(F.kl_div(log_q, p, reduction="batchmean"))

    fisher = _quick(metric=fisher_pullback(w, states_t, n_sub=300)).fit(states)
    identity = _quick().fit(states)
    assert downstream_kl(fisher.reconstruct(states, k=1)) < downstream_kl(identity.reconstruct(states, k=1))
