"""Tests for the synthetic data generators."""

import math

import pytest
import torch

from igl import IGLConfigError
from igl.data import (
    embed_in_high_dim,
    make_flat_torus,
    make_flat_torus_labels,
    make_moons,
    make_swiss_roll,
)


def test_make_flat_torus_shape_and_range() -> None:
    x, theta = make_flat_torus(100, seed=0)
    assert x.shape == (100, 4)
    assert theta.shape == (100, 2)
    assert theta.min() >= 0
    assert theta.max() <= 2 * math.pi


def test_make_flat_torus_points_on_torus() -> None:
    """Embedded points should satisfy ``x₁² + x₂² = 1`` and ``x₃² + x₄² = 1``."""
    x, _ = make_flat_torus(50, seed=1)
    assert torch.allclose(x[:, 0] ** 2 + x[:, 1] ** 2, torch.ones(50), atol=1e-4)
    assert torch.allclose(x[:, 2] ** 2 + x[:, 3] ** 2, torch.ones(50), atol=1e-4)


def test_make_flat_torus_noise_adds_jitter() -> None:
    x_clean, _ = make_flat_torus(30, noise=0.0, seed=2)
    x_noisy, _ = make_flat_torus(30, noise=0.5, seed=2)
    assert not torch.allclose(x_clean, x_noisy)


def test_make_flat_torus_rejects_zero_samples() -> None:
    with pytest.raises(IGLConfigError, match="n_samples"):
        make_flat_torus(0)


def test_make_flat_torus_without_seed_still_returns_correct_shape() -> None:
    """When ``seed`` is ``None`` we use a fresh non-deterministic generator."""
    x, theta = make_flat_torus(15)
    assert x.shape == (15, 4)
    assert theta.shape == (15, 2)


def test_make_flat_torus_labels_regression_smooth_shape() -> None:
    _, theta = make_flat_torus(50, seed=0)
    y = make_flat_torus_labels(theta, task="regression_smooth")
    assert y.shape == (50, 4)
    assert y.dtype == torch.float32


def test_make_flat_torus_labels_hemisphere() -> None:
    _, theta = make_flat_torus(50, seed=0)
    y = make_flat_torus_labels(theta, task="hemisphere")
    assert y.shape == (50,)
    assert y.dtype == torch.int64
    assert set(y.unique().tolist()) <= {0, 1}


def test_make_flat_torus_labels_xor() -> None:
    _, theta = make_flat_torus(50, seed=0)
    y = make_flat_torus_labels(theta, task="xor")
    assert y.shape == (50,)
    assert set(y.unique().tolist()) <= {0, 1}


def test_make_flat_torus_labels_unknown_task_raises() -> None:
    _, theta = make_flat_torus(10, seed=0)
    with pytest.raises(IGLConfigError, match="unknown task"):
        make_flat_torus_labels(theta, task="not_a_task")


def test_make_swiss_roll_shape() -> None:
    x, params = make_swiss_roll(80, seed=3)
    assert x.shape == (80, 3)
    assert params.shape == (80, 2)


def test_make_swiss_roll_noise() -> None:
    x_clean, _ = make_swiss_roll(30, noise=0.0, seed=4)
    x_noisy, _ = make_swiss_roll(30, noise=0.5, seed=4)
    assert not torch.allclose(x_clean, x_noisy)


def test_make_swiss_roll_rejects_zero_samples() -> None:
    with pytest.raises(IGLConfigError, match="n_samples"):
        make_swiss_roll(0)


def test_make_moons_shape_and_labels() -> None:
    x, y = make_moons(60, seed=5)
    assert x.shape == (60, 2)
    assert y.shape == (60,)
    assert y.dtype == torch.int64
    assert set(y.unique().tolist()) == {0, 1}


def test_make_moons_noise_changes_layout() -> None:
    x_clean, _ = make_moons(40, noise=0.0, seed=6)
    x_noisy, _ = make_moons(40, noise=0.3, seed=6)
    assert not torch.allclose(x_clean, x_noisy)


def test_make_moons_rejects_too_few_samples() -> None:
    with pytest.raises(IGLConfigError, match="n_samples"):
        make_moons(1)


def test_embed_in_high_dim_preserves_distances_approximately() -> None:
    """Orthogonal rotations preserve pairwise distances exactly."""
    torch.manual_seed(0)
    x_low = torch.randn(20, 3)
    x_high = embed_in_high_dim(x_low, target_dim=16, seed=10)
    d_low = (x_low[None, :, :] - x_low[:, None, :]).norm(dim=-1)
    d_high = (x_high[None, :, :] - x_high[:, None, :]).norm(dim=-1)
    torch.testing.assert_close(d_low, d_high, rtol=1e-4, atol=1e-4)


def test_embed_in_high_dim_shape() -> None:
    x = torch.randn(15, 5)
    y = embed_in_high_dim(x, target_dim=20, seed=0)
    assert y.shape == (15, 20)


def test_embed_in_high_dim_target_eq_low_returns_clone() -> None:
    x = torch.randn(10, 4)
    y = embed_in_high_dim(x, target_dim=4, seed=0)
    torch.testing.assert_close(x, y)
    assert y is not x  # cloned, not aliased


def test_embed_in_high_dim_rejects_smaller_target() -> None:
    x = torch.randn(5, 6)
    with pytest.raises(IGLConfigError, match="target_dim"):
        embed_in_high_dim(x, target_dim=4)
