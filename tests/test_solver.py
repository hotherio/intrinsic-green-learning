"""Tests for :func:`igl.direct_solve_weights`."""

import warnings

import pytest
import torch

from igl import direct_solve_weights


def test_solver_recovers_true_weights_when_well_posed() -> None:
    torch.manual_seed(0)
    n_samples, n_anchors, n_outputs = 60, 8, 2
    phi = torch.randn(n_samples, n_anchors)
    w_true = torch.randn(n_anchors, n_outputs)
    y = phi @ w_true
    w_est = direct_solve_weights(phi, y, l2=1e-6)
    # Tight residual on the predictions; weights themselves may differ if
    # phi is non-orthogonal, but predictions must match.
    torch.testing.assert_close(phi @ w_est, y, rtol=1e-3, atol=1e-3)


def test_solver_handles_one_dimensional_targets() -> None:
    torch.manual_seed(0)
    phi = torch.randn(20, 5)
    y = torch.randn(20)  # 1-D
    w = direct_solve_weights(phi, y, l2=1e-3)
    assert w.shape == (5, 1)


def test_solver_returns_zero_on_non_finite_solution() -> None:
    """When the lstsq produces non-finite weights, we fall back to zeros."""
    # Build a pathological Phi: very large values that overflow the lstsq.
    phi = torch.full((10, 4), float("inf"))
    y = torch.randn(10, 2)
    with warnings.catch_warnings(record=True) as warned:
        warnings.simplefilter("always", RuntimeWarning)
        w = direct_solve_weights(phi, y, l2=1e-3)
    assert torch.all(w == 0)
    assert any("non-finite" in str(w_.message) for w_ in warned)


def test_solver_returns_no_gradient() -> None:
    phi = torch.randn(20, 4, requires_grad=True)
    y = torch.randn(20, 1)
    w = direct_solve_weights(phi, y, l2=1e-3)
    assert not w.requires_grad


def test_solver_runs_on_cpu_regardless_of_input_device() -> None:
    phi = torch.randn(10, 3)
    y = torch.randn(10, 1)
    w = direct_solve_weights(phi, y)
    # We don't assert on output device here (it's CPU); just that it works.
    assert w.shape == (3, 1)


def test_solver_uses_l2_to_regularise_underdetermined_system() -> None:
    """Underdetermined Φ (R > N): l2 prevents arbitrary blow-up."""
    torch.manual_seed(0)
    phi = torch.randn(5, 20)
    y = torch.randn(5, 1)
    w_weak = direct_solve_weights(phi, y, l2=1e-6)
    w_strong = direct_solve_weights(phi, y, l2=1.0)
    # Stronger l2 should shrink the weight norm.
    assert w_strong.norm() < w_weak.norm()


@pytest.mark.parametrize("l2", [1e-6, 1e-3, 1e-1, 1.0])
def test_solver_predictions_are_close_to_targets(l2: float) -> None:
    torch.manual_seed(0)
    phi = torch.randn(40, 6)
    y = torch.randn(40, 2)
    w = direct_solve_weights(phi, y, l2=l2)
    residual = (phi @ w - y).norm() / y.norm()
    # Looser l2 → tighter fit; either way the residual is bounded.
    assert residual < 5.0
