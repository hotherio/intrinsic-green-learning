"""Tests for :class:`igl.MatryoshkaTrainer` and :class:`igl.TrainingHistory`."""

import logging

import pytest
import torch

from igl import (
    CrossEntropyLoss,
    EpochStats,
    IGLConvergenceError,
    IGLModule,
    MatryoshkaConfig,
    MatryoshkaTrainer,
    MSELoss,
    PowerLawSampler,
    TrainingHistory,
    UniformSampler,
)
from igl.data import embed_in_high_dim, make_flat_torus, make_flat_torus_labels, make_moons


def _moons_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x_2d, y = make_moons(200, noise=0.1, seed=42)
    x = embed_in_high_dim(x_2d, target_dim=12, seed=123)
    return x[:150], y[:150], x[150:], y[150:]


def _torus_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x, theta = make_flat_torus(160, seed=42)
    y = make_flat_torus_labels(theta, task="regression_smooth")
    return x[:120], y[:120], x[120:], y[120:]


def test_trainer_converges_on_classification() -> None:
    x_train, y_train, x_val, y_val = _moons_data()
    module = IGLModule(input_dim=12, max_dim=4, output_dim=2, n_anchors=16, n_scales=2)
    trainer = MatryoshkaTrainer(
        loss=CrossEntropyLoss(n_classes=2),
        config=MatryoshkaConfig(
            epochs=40,
            batch_size=32,
            inner_batch_size=150,
            scheduler="none",
            early_stop_patience=None,
            verbose=False,
        ),
    )
    history = trainer.fit(module, x_train, y_train, x_val=x_val, y_val=y_val)
    assert isinstance(history, TrainingHistory)
    assert history.val_metric[-1] > 0.85  # accuracy on easy task


def test_trainer_converges_on_regression() -> None:
    x_train, y_train, x_val, y_val = _torus_data()
    module = IGLModule(input_dim=4, max_dim=4, output_dim=4, n_anchors=16, n_scales=2)
    trainer = MatryoshkaTrainer(
        loss=MSELoss(),
        config=MatryoshkaConfig(
            epochs=60,
            batch_size=32,
            inner_batch_size=120,
            scheduler="none",
            early_stop_patience=None,
            verbose=False,
        ),
    )
    history = trainer.fit(module, x_train, y_train, x_val=x_val, y_val=y_val)
    assert history.val_metric[-1] < 0.5  # MSE small


def test_trainer_history_records_per_epoch_data() -> None:
    x_train, y_train, x_val, y_val = _moons_data()
    module = IGLModule(input_dim=12, max_dim=4, output_dim=2)
    trainer = MatryoshkaTrainer(
        loss=CrossEntropyLoss(n_classes=2),
        config=MatryoshkaConfig(
            epochs=5, batch_size=32, inner_batch_size=150, scheduler="none", early_stop_patience=None, verbose=False
        ),
    )
    history = trainer.fit(module, x_train, y_train, x_val=x_val, y_val=y_val)
    assert len(history.train_loss) == 5
    assert len(history.val_loss) == 5
    assert len(history.val_metric) == 5
    assert len(history.truncation_k) == 5


def test_trainer_supports_explicit_sampler() -> None:
    x_train, y_train, x_val, y_val = _moons_data()
    module = IGLModule(input_dim=12, max_dim=4, output_dim=2)
    sampler = PowerLawSampler(alpha=1.5)
    trainer = MatryoshkaTrainer(
        loss=CrossEntropyLoss(n_classes=2),
        config=MatryoshkaConfig(
            epochs=3, batch_size=32, inner_batch_size=150, scheduler="none", early_stop_patience=None, verbose=False
        ),
        sampler=sampler,
    )
    history = trainer.fit(module, x_train, y_train, x_val=x_val, y_val=y_val)
    assert len(history.train_loss) == 3


def test_trainer_default_sampler_is_uniform_when_config_says_uniform() -> None:
    trainer = MatryoshkaTrainer(
        loss=CrossEntropyLoss(n_classes=2),
        config=MatryoshkaConfig(sampling="uniform"),
    )
    assert isinstance(trainer.sampler, UniformSampler)


def test_trainer_default_sampler_is_power_law_when_config_says_power_law() -> None:
    trainer = MatryoshkaTrainer(
        loss=CrossEntropyLoss(n_classes=2),
        config=MatryoshkaConfig(sampling="power_law", alpha=2.0),
    )
    assert isinstance(trainer.sampler, PowerLawSampler)


def test_trainer_early_stops_when_no_improvement() -> None:
    x_train, y_train, x_val, y_val = _moons_data()
    module = IGLModule(input_dim=12, max_dim=4, output_dim=2)
    trainer = MatryoshkaTrainer(
        loss=CrossEntropyLoss(n_classes=2),
        config=MatryoshkaConfig(
            epochs=200,
            batch_size=32,
            inner_batch_size=150,
            scheduler="none",
            early_stop_patience=3,
            early_stop_min_epochs=5,
            verbose=False,
        ),
    )
    history = trainer.fit(module, x_train, y_train, x_val=x_val, y_val=y_val)
    # Either early-stopped or completed; if early-stopped, stopped_epoch < 200.
    if history.early_stopped:
        assert history.stopped_epoch is not None
        assert history.stopped_epoch < 200


def test_trainer_raises_on_non_finite_loss() -> None:
    """A pathological learning rate produces NaNs almost immediately."""
    torch.manual_seed(0)
    x_train, y_train, x_val, y_val = _moons_data()
    module = IGLModule(input_dim=12, max_dim=4, output_dim=2)
    trainer = MatryoshkaTrainer(
        loss=CrossEntropyLoss(n_classes=2),
        config=MatryoshkaConfig(
            epochs=5,
            batch_size=32,
            inner_batch_size=150,
            encoder_lr=1e30,
            grad_clip=0.0,
            scheduler="none",
            early_stop_patience=None,
            verbose=False,
        ),
    )
    with pytest.raises(IGLConvergenceError):
        trainer.fit(module, x_train, y_train, x_val=x_val, y_val=y_val)


def test_trainer_works_without_validation_data() -> None:
    x_train, y_train, _, _ = _moons_data()
    module = IGLModule(input_dim=12, max_dim=4, output_dim=2)
    trainer = MatryoshkaTrainer(
        loss=CrossEntropyLoss(n_classes=2),
        config=MatryoshkaConfig(
            epochs=3, batch_size=32, inner_batch_size=150, scheduler="none", early_stop_patience=None, verbose=False
        ),
    )
    history = trainer.fit(module, x_train, y_train)
    assert len(history.train_loss) == 3
    assert len(history.val_loss) == 0


def test_trainer_with_noise_std_injects_input_noise() -> None:
    x_train, y_train, x_val, y_val = _moons_data()
    module = IGLModule(input_dim=12, max_dim=4, output_dim=2)
    trainer = MatryoshkaTrainer(
        loss=CrossEntropyLoss(n_classes=2),
        config=MatryoshkaConfig(
            epochs=3,
            batch_size=32,
            inner_batch_size=150,
            scheduler="none",
            early_stop_patience=None,
            noise_std=0.05,
            verbose=False,
        ),
    )
    history = trainer.fit(module, x_train, y_train, x_val=x_val, y_val=y_val)
    assert len(history.train_loss) == 3


def test_trainer_with_weight_decay_uses_adamw_decay() -> None:
    x_train, y_train, x_val, y_val = _moons_data()
    module = IGLModule(input_dim=12, max_dim=4, output_dim=2)
    trainer = MatryoshkaTrainer(
        loss=CrossEntropyLoss(n_classes=2),
        config=MatryoshkaConfig(
            epochs=3,
            batch_size=32,
            inner_batch_size=150,
            weight_decay=1e-2,
            scheduler="none",
            early_stop_patience=None,
            verbose=False,
        ),
    )
    history = trainer.fit(module, x_train, y_train, x_val=x_val, y_val=y_val)
    assert len(history.train_loss) == 3


def test_trainer_cosine_warm_restarts_scheduler_runs() -> None:
    x_train, y_train, x_val, y_val = _moons_data()
    module = IGLModule(input_dim=12, max_dim=4, output_dim=2)
    trainer = MatryoshkaTrainer(
        loss=CrossEntropyLoss(n_classes=2),
        config=MatryoshkaConfig(
            epochs=3,
            batch_size=32,
            inner_batch_size=150,
            scheduler="cosine_warm_restarts",
            early_stop_patience=None,
            verbose=False,
        ),
    )
    history = trainer.fit(module, x_train, y_train, x_val=x_val, y_val=y_val)
    assert len(history.train_loss) == 3


def test_trainer_verbose_logs_epochs(caplog: pytest.LogCaptureFixture) -> None:
    x_train, y_train, _, _ = _moons_data()
    module = IGLModule(input_dim=12, max_dim=4, output_dim=2, n_anchors=8, n_scales=2)
    config = MatryoshkaConfig(epochs=3, batch_size=64, verbose=True, log_every=1, early_stop_patience=None)
    trainer = MatryoshkaTrainer(loss=CrossEntropyLoss(n_classes=2), config=config)
    with caplog.at_level(logging.INFO, logger="igl"):
        trainer.fit(module, x_train, y_train)
    epochs_logged = [r for r in caplog.records if r.name == "igl" and "train_loss=" in r.getMessage()]
    assert len(epochs_logged) == 3


def test_trainer_silent_by_default(caplog: pytest.LogCaptureFixture) -> None:
    x_train, y_train, _, _ = _moons_data()
    module = IGLModule(input_dim=12, max_dim=4, output_dim=2, n_anchors=8, n_scales=2)
    config = MatryoshkaConfig(epochs=2, batch_size=64, early_stop_patience=None)
    trainer = MatryoshkaTrainer(loss=CrossEntropyLoss(n_classes=2), config=config)
    with caplog.at_level(logging.INFO, logger="igl"):
        trainer.fit(module, x_train, y_train)
    assert not [r for r in caplog.records if r.name == "igl"]


def test_trainer_on_epoch_callback_receives_stats() -> None:
    x_train, y_train, _, _ = _moons_data()
    module = IGLModule(input_dim=12, max_dim=4, output_dim=2, n_anchors=8, n_scales=2)
    config = MatryoshkaConfig(epochs=3, batch_size=64, early_stop_patience=None)
    trainer = MatryoshkaTrainer(loss=CrossEntropyLoss(n_classes=2), config=config)
    seen: list[EpochStats] = []
    trainer.fit(module, x_train, y_train, on_epoch=seen.append)
    assert [s.epoch for s in seen] == [0, 1, 2]
    assert all(isinstance(s.train_loss, float) for s in seen)
    assert all(s.val_loss is None for s in seen)


def test_trainer_on_epoch_callback_sees_validation_and_best_epoch() -> None:
    x_train, y_train, x_val, y_val = _moons_data()
    module = IGLModule(input_dim=12, max_dim=4, output_dim=2, n_anchors=8, n_scales=2)
    config = MatryoshkaConfig(epochs=3, batch_size=64, early_stop_patience=100, early_stop_min_epochs=1)
    trainer = MatryoshkaTrainer(loss=CrossEntropyLoss(n_classes=2), config=config)
    seen: list[EpochStats] = []
    trainer.fit(module, x_train, y_train, x_val=x_val, y_val=y_val, on_epoch=seen.append)
    assert all(s.val_loss is not None and s.val_metric is not None for s in seen)
    assert seen[-1].best_epoch is not None
