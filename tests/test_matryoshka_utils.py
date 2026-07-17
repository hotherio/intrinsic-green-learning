"""Tests for PrefixForward training, greedy knockout, seeding, and stop reasons."""

import warnings

import torch
from torch import nn

from igl import (
    CrossEntropyLoss,
    IGLConfig,
    IGLModule,
    IGLRegressor,
    KnockoutResult,
    MatryoshkaConfig,
    MatryoshkaTrainer,
    MSELoss,
    PrefixForward,
    detect_knockout_knee,
    eval_dimension_curve,
    fit_seeds,
    greedy_knockout,
    seed_everything,
)
from igl.data import embed_in_high_dim, make_moons


class PrefixMaskedAE(nn.Module):
    """Minimal autoencoder satisfying :class:`igl.PrefixForward`."""

    def __init__(self, dim: int, max_dim: int) -> None:
        super().__init__()
        self.max_dim = max_dim
        self.enc = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, max_dim))
        self.dec = nn.Sequential(nn.Linear(max_dim, 32), nn.GELU(), nn.Linear(32, dim))

    def forward(self, x: torch.Tensor, *, gate_mask: torch.Tensor | None = None) -> torch.Tensor:
        z = self.enc(x)
        if gate_mask is not None:
            z = z * gate_mask.unsqueeze(0)
        return self.dec(z)


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    x_2d, y = make_moons(200, noise=0.08, seed=42)
    return embed_in_high_dim(x_2d, target_dim=10, seed=123), y


def test_prefix_forward_protocol_is_satisfied() -> None:
    assert isinstance(PrefixMaskedAE(10, 4), PrefixForward)


def test_trainer_fits_prefix_forward_module() -> None:
    x, _ = _data()
    ae = PrefixMaskedAE(10, 4)
    config = MatryoshkaConfig(epochs=20, batch_size=64, early_stop_patience=None)
    history = MatryoshkaTrainer(loss=MSELoss(), config=config).fit(ae, x, x)
    assert len(history.train_loss) == 20
    assert history.train_loss[-1] < history.train_loss[0]


def test_dimension_curve_for_prefix_forward_module() -> None:
    x, _ = _data()
    ae = PrefixMaskedAE(10, 4)
    config = MatryoshkaConfig(epochs=30, batch_size=64, early_stop_patience=None)
    MatryoshkaTrainer(loss=MSELoss(), config=config).fit(ae, x, x)
    curve = eval_dimension_curve(ae, x, x, loss=MSELoss())
    assert set(curve) == {1, 2, 3, 4}
    assert curve[4] <= curve[1] + 1e-6


def test_greedy_knockout_certifies_moons() -> None:
    x, y = _data()
    config = IGLConfig(max_dim=4, matryoshka=MatryoshkaConfig(epochs=40, batch_size=64, early_stop_patience=None))
    est = IGLRegressor(max_dim=4, config=config, random_state=0).fit(x.numpy(), y.numpy().astype(float))
    result = greedy_knockout(est.module_, x, y.float(), loss=MSELoss())
    assert isinstance(result, KnockoutResult)
    assert set(result.curve) == {1, 2, 3, 4}
    assert len(result.removal_order) == 3
    assert 1 <= result.knee <= 4


def test_knockout_scores_classification_by_error_rate() -> None:
    x, y = _data()
    config = IGLConfig(max_dim=4, matryoshka=MatryoshkaConfig(epochs=40, batch_size=64, early_stop_patience=None))
    from igl import IGLClassifier

    est = IGLClassifier(max_dim=4, config=config, random_state=0).fit(x.numpy(), y.numpy())
    result = greedy_knockout(est.module_, x, y, loss=CrossEntropyLoss(n_classes=2))
    assert all(0.0 <= score <= 1.0 for score in result.curve.values())


def test_detect_knockout_knee_does_not_fire_unconditionally_at_one() -> None:
    curve = {1: 10.0, 2: 0.11, 3: 0.10, 4: 0.10}
    assert detect_knockout_knee(curve) == 2


def test_seed_everything_makes_torch_reproducible() -> None:
    seed_everything(7)
    first = torch.randn(4)
    seed_everything(7)
    assert torch.equal(first, torch.randn(4))


def test_fit_seeds_aggregates() -> None:
    x, y = _data()
    config = IGLConfig(max_dim=4, matryoshka=MatryoshkaConfig(epochs=15, batch_size=64, early_stop_patience=None))
    result = fit_seeds(
        lambda seed: IGLRegressor(max_dim=4, config=config, random_state=seed),
        x.numpy(),
        y.numpy().astype(float),
        seeds=[0, 1, 2],
    )
    assert result["seeds"] == [0, 1, 2]
    aggregate = result["aggregate"]
    assert isinstance(aggregate, dict)
    assert {"mean", "std", "median", "iqr"} <= set(aggregate["effective_dimension"])


def test_stop_reason_max_epochs() -> None:
    x, y = _data()
    module = IGLModule(input_dim=10, max_dim=4, output_dim=2, n_anchors=8, n_scales=2)
    config = MatryoshkaConfig(epochs=3, batch_size=64, early_stop_patience=None)
    history = MatryoshkaTrainer(loss=CrossEntropyLoss(n_classes=2), config=config).fit(module, x, y)
    assert history.stop_reason == "max_epochs"
    assert not history.converged


def test_stop_reason_plateau_on_early_stop() -> None:
    x, y = _data()
    module = IGLModule(input_dim=10, max_dim=4, output_dim=2, n_anchors=8, n_scales=2)
    config = MatryoshkaConfig(epochs=400, batch_size=64, early_stop_patience=5, early_stop_min_epochs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        history = MatryoshkaTrainer(loss=CrossEntropyLoss(n_classes=2), config=config).fit(
            module, x[:150], y[:150], x_val=x[150:], y_val=y[150:]
        )
    assert history.early_stopped
    assert history.stop_reason in {"plateau", "improving_at_stop"}


def test_classify_stop_flags_improving_series() -> None:
    trainer = MatryoshkaTrainer(loss=MSELoss(), config=MatryoshkaConfig(early_stop_patience=5))
    from igl import TrainingHistory

    improving = TrainingHistory(val_metric=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    assert trainer._classify_stop(improving) == "improving_at_stop"  # noqa: SLF001
    flat = TrainingHistory(val_metric=[0.5, 0.5001, 0.4999, 0.5, 0.5, 0.5])
    assert trainer._classify_stop(flat) == "plateau"  # noqa: SLF001
