"""Tests for :mod:`igl.whitening`."""

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from igl import IGLConfigError, IGLModule, IGLNotFittedError, MatryoshkaConfig, MatryoshkaTrainer
from igl.whitening import (
    TargetWhitener,
    WhitenedMSELoss,
    damped_metric,
    fisher_pullback,
    logit_metric,
    psd_sqrt_inv,
    tail_metric,
)


def _random_psd(dim: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    m = torch.randn(dim, dim, generator=generator)
    return m @ m.T + 0.1 * torch.eye(dim)


def test_psd_sqrt_inv_roundtrip() -> None:
    g = _random_psd(6)
    a, a_inv = psd_sqrt_inv(g)
    assert torch.allclose(a @ a, g, atol=1e-4)
    assert torch.allclose(a @ a_inv, torch.eye(6), atol=1e-4)


def test_psd_sqrt_inv_clamps_null_directions() -> None:
    g = torch.diag(torch.tensor([1.0, 1.0, 0.0]))
    a, a_inv = psd_sqrt_inv(g, clamp=1e-2)
    assert torch.isfinite(a).all()
    assert torch.isfinite(a_inv).all()
    assert float(a_inv[2, 2]) <= 1.0 / (1e-2**0.5) + 1e-3


def test_psd_sqrt_inv_rejects_non_square() -> None:
    with pytest.raises(IGLConfigError, match="square"):
        psd_sqrt_inv(torch.randn(3, 4))


def test_logit_metric_is_gram_matrix() -> None:
    w = torch.randn(10, 4)
    assert torch.allclose(logit_metric(w), w.T @ w)


def test_fisher_pullback_matches_dense_computation() -> None:
    generator = torch.Generator().manual_seed(0)
    w = torch.randn(5, 3, generator=generator)
    states = torch.randn(40, 3, generator=generator)
    estimated = fisher_pullback(w, states, n_sub=40, batch=8)
    p = F.softmax(states @ w.T, dim=-1)
    exact = torch.zeros(3, 3, dtype=torch.float64)
    for i in range(40):
        d = torch.diag(p[i].double()) - p[i].double().outer(p[i].double())
        exact += w.double().T @ d @ w.double()
    exact /= 40
    assert torch.allclose(estimated, exact.float(), atol=1e-4)


def test_fisher_pullback_is_symmetric_psd() -> None:
    generator = torch.Generator().manual_seed(1)
    w = torch.randn(20, 6, generator=generator)
    states = torch.randn(100, 6, generator=generator)
    g = fisher_pullback(w, states)
    assert torch.allclose(g, g.T, atol=1e-6)
    assert float(torch.linalg.eigvalsh(g.double()).min()) >= -1e-6  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


def test_damped_metric_trace_scaling() -> None:
    g = _random_psd(4, seed=2)
    m = _random_psd(4, seed=3)
    damped = damped_metric(g, m, lam=0.1)
    scale = 0.1 * float(torch.trace(g)) / float(torch.trace(m))
    assert torch.allclose(damped, g + scale * m, atol=1e-5)


def test_tail_metric_recovers_linear_pullback() -> None:
    generator = torch.Generator().manual_seed(4)
    j = torch.randn(5, 3, generator=generator)  # linear map: x [N,3] -> x @ j.T [N,5]
    g = _random_psd(5, seed=5)
    states = torch.randn(30, 3, generator=generator)
    estimated = tail_metric(lambda x: x @ j.T, states, g, n_probes=4096, generator=generator)
    exact = j.T @ g @ j
    assert torch.allclose(estimated, exact, rtol=0.15, atol=0.15 * float(exact.abs().max()))


def test_target_whitener_roundtrip() -> None:
    g = _random_psd(5, seed=6)
    y = torch.randn(50, 5) * 3.0 + 1.0
    whitener = TargetWhitener(g).fit(y)
    restored = whitener.inverse_transform(whitener.transform(y))
    assert torch.allclose(restored, y, atol=1e-3)
    assert abs(float(whitener.transform(y).std()) - 1.0) < 0.05


def test_target_whitener_identity_metric_centers_and_scales() -> None:
    y = torch.randn(50, 3) * 2.0 + 5.0
    whitener = TargetWhitener().fit(y)
    transformed = whitener.transform(y)
    assert float(transformed.mean().abs()) < 0.1
    assert abs(float(transformed.std()) - 1.0) < 0.05


def test_target_whitener_unfitted_raises() -> None:
    with pytest.raises(IGLNotFittedError):
        TargetWhitener().transform(torch.randn(4, 3))


def test_target_whitener_state_dict_roundtrip() -> None:
    g = _random_psd(4, seed=7)
    y = torch.randn(30, 4)
    whitener = TargetWhitener(g).fit(y)
    rebuilt = TargetWhitener.from_state_dict(whitener.state_dict())
    assert torch.allclose(rebuilt.transform(y), whitener.transform(y), atol=1e-6)
    assert rebuilt.is_fitted


def test_target_whitener_state_dict_missing_key_raises() -> None:
    g = _random_psd(4, seed=8)
    state = TargetWhitener(g).fit(torch.randn(30, 4)).state_dict()
    del state["a_inv"]
    with pytest.raises(IGLConfigError, match="missing keys"):
        TargetWhitener.from_state_dict(state)


def test_whitened_mse_loss_requires_fitted_whitener() -> None:
    with pytest.raises(IGLNotFittedError):
        WhitenedMSELoss(TargetWhitener())


def test_whitened_mse_tracks_second_order_kl() -> None:
    """Whitened MSE is proportional to the KL of the softmax read-out for small errors."""
    generator = torch.Generator().manual_seed(9)
    w = torch.randn(12, 4, generator=generator)
    y = torch.randn(200, 4, generator=generator)
    g = fisher_pullback(w, y, n_sub=200)
    whitener = TargetWhitener(g).fit(y)
    loss = WhitenedMSELoss(whitener)
    perturbation = 0.01 * torch.randn(200, 4, generator=generator)
    y_hat = y + perturbation

    whitened_gap = loss.metric(loss.target(y_hat), loss.target(y))
    p = F.softmax(y @ w.T, dim=-1)
    log_q = F.log_softmax(y_hat @ w.T, dim=-1)
    kl = float(F.kl_div(log_q, p, reduction="batchmean"))
    # metric() averages squared error over N*C and rescales by 1/y_scale^2;
    # second-order KL per sample is 0.5 * |A delta|^2. Undo both to compare.
    implied_kl = 0.5 * whitened_gap * y.shape[1] * whitener.y_scale_**2
    assert implied_kl == pytest.approx(kl, rel=0.1)


def test_whitened_loss_trains_with_matryoshka_trainer() -> None:
    generator = torch.Generator().manual_seed(10)
    w = torch.randn(8, 6, generator=generator)
    y = torch.randn(120, 6, generator=generator)
    whitener = TargetWhitener(fisher_pullback(w, y, n_sub=120)).fit(y)
    module = IGLModule(input_dim=6, max_dim=4, output_dim=6, n_anchors=8, n_scales=2)
    config = MatryoshkaConfig(epochs=5, batch_size=32, early_stop_patience=None)
    trainer = MatryoshkaTrainer(loss=WhitenedMSELoss(whitener), config=config)
    history = trainer.fit(module, y, y)
    assert len(history.train_loss) == 5
    assert all(torch.isfinite(torch.tensor(loss_value)) for loss_value in history.train_loss)
