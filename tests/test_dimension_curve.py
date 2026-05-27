"""Tests for :func:`igl.eval_dimension_curve`, :func:`igl.detect_elbow`, and friends."""

import pytest
import torch

from igl import (
    CrossEntropyLoss,
    IGLConfigError,
    IGLModule,
    MatryoshkaConfig,
    MatryoshkaTrainer,
    MSELoss,
    d_eff_from_curve,
    detect_elbow,
    eval_dimension_curve,
)
from igl.data import embed_in_high_dim, make_flat_torus, make_flat_torus_labels, make_moons


def test_detect_elbow_on_two_point_curve() -> None:
    """With only one transition, the elbow is at the second point."""
    curve = {1: 1.0, 2: 0.01}
    assert detect_elbow(curve) == 2


def test_detect_elbow_picks_dimension_with_largest_drop() -> None:
    # Clear elbow at k=3.
    curve = {1: 1.0, 2: 0.5, 3: 0.01, 4: 0.009, 5: 0.0089}
    assert detect_elbow(curve) == 3


def test_detect_elbow_returns_first_when_curve_is_flat() -> None:
    curve = {1: 0.5, 2: 0.5, 3: 0.5}
    assert detect_elbow(curve) == 1


def test_detect_elbow_handles_single_point() -> None:
    curve = {1: 0.5}
    assert detect_elbow(curve) == 1


def test_detect_elbow_handles_all_zero_losses() -> None:
    curve = {1: 0.0, 2: 0.0, 3: 0.0}
    assert detect_elbow(curve) == 1


def test_detect_elbow_increasing_curve_returns_first_dim() -> None:
    curve = {1: 0.1, 2: 0.2, 3: 0.3}
    assert detect_elbow(curve) == 1


def test_detect_elbow_rejects_empty_curve() -> None:
    with pytest.raises(IGLConfigError, match="at least one entry"):
        detect_elbow({})


def test_detect_elbow_rejects_invalid_ratio() -> None:
    with pytest.raises(IGLConfigError, match="ratio"):
        detect_elbow({1: 1.0, 2: 0.5}, ratio=0.0)


def test_d_eff_from_curve_is_alias_of_detect_elbow() -> None:
    curve = {1: 1.0, 2: 0.1, 3: 0.09}
    assert d_eff_from_curve(curve) == detect_elbow(curve)


def test_eval_dimension_curve_returns_one_entry_per_k() -> None:
    torch.manual_seed(0)
    x_2d, y = make_moons(100, noise=0.1, seed=42)
    x = embed_in_high_dim(x_2d, target_dim=8, seed=123)

    module = IGLModule(input_dim=8, max_dim=5, output_dim=2, n_anchors=12, n_scales=2)
    trainer = MatryoshkaTrainer(
        loss=CrossEntropyLoss(n_classes=2),
        config=MatryoshkaConfig(
            epochs=10, batch_size=32, inner_batch_size=100, scheduler="none", early_stop_patience=None, verbose=False
        ),
    )
    trainer.fit(module, x, y)
    curve = eval_dimension_curve(module, x, y, loss=CrossEntropyLoss(n_classes=2))
    assert sorted(curve.keys()) == [1, 2, 3, 4, 5]
    for v in curve.values():
        assert v >= 0  # cross-entropy is non-negative


def test_eval_dimension_curve_works_for_regression() -> None:
    torch.manual_seed(0)
    x, theta = make_flat_torus(80, seed=42)
    y = make_flat_torus_labels(theta, task="regression_smooth")

    module = IGLModule(input_dim=4, max_dim=3, output_dim=4, n_anchors=12, n_scales=2)
    trainer = MatryoshkaTrainer(
        loss=MSELoss(),
        config=MatryoshkaConfig(
            epochs=20, batch_size=32, inner_batch_size=80, scheduler="none", early_stop_patience=None, verbose=False
        ),
    )
    trainer.fit(module, x, y)
    curve = eval_dimension_curve(module, x, y, loss=MSELoss())
    assert sorted(curve.keys()) == [1, 2, 3]


def test_full_pipeline_finds_low_dimension_on_moons() -> None:
    """End-to-end: train a small model on moons and verify the discovered
    effective dimension is small (the true intrinsic dimension is 1)."""
    torch.manual_seed(0)
    x_2d, y = make_moons(300, noise=0.1, seed=42)
    x = embed_in_high_dim(x_2d, target_dim=16, seed=123)

    module = IGLModule(input_dim=16, max_dim=6, output_dim=2, n_anchors=24, n_scales=3)
    trainer = MatryoshkaTrainer(
        loss=CrossEntropyLoss(n_classes=2),
        config=MatryoshkaConfig(
            epochs=50, batch_size=64, inner_batch_size=300, scheduler="none", early_stop_patience=None, verbose=False
        ),
    )
    trainer.fit(module, x, y)
    curve = eval_dimension_curve(module, x, y, loss=CrossEntropyLoss(n_classes=2))
    d_eff = detect_elbow(curve)
    assert d_eff <= 4  # well below max_dim=6; moons is genuinely low-D
