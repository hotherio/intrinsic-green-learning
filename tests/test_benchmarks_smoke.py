"""Smoke tests keeping the VP benchmark suite honest.

Every zoo solver must agree with the package's ``direct_solve_weights`` on a
small well-conditioned problem; the harness utilities must run. These tests
are fast — the actual benchmarks are run manually, never in CI.
"""

import pytest
import torch

from benchmarks.vp._harness import machine_state, peak_rss_mb, time_median
from benchmarks.vp.solvers import (
    adamw_to_tol,
    block_cg,
    cholesky_normal,
    direct_lstsq,
    effective_l2,
    lbfgs_convex,
    nystrom_pcg,
    sgd_to_tol,
)

_N, _P, _D = 200, 40, 8


@pytest.fixture
def problem() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    phi = torch.randn(_N, _P, generator=generator)
    y = torch.randn(_N, _D, generator=generator)
    return phi, y


def test_iterative_solvers_match_direct(problem: tuple[torch.Tensor, torch.Tensor]) -> None:
    phi, y = problem
    reference = direct_lstsq(phi, y).w
    for name, solve in {
        "cholesky": lambda: cholesky_normal(phi, y),
        "cg": lambda: block_cg(phi, y, tol=1e-10),
        "cg_matrix_free": lambda: block_cg(phi, y, tol=1e-10, matrix_free=True),
        "nystrom_pcg": lambda: nystrom_pcg(phi, y, tol=1e-10, sketch_rank=16),
        "gd": lambda: sgd_to_tol(phi, y, tol=1e-6),
        "sgd": lambda: sgd_to_tol(phi, y, tol=1e-6, batch_size=64),
        "adamw": lambda: adamw_to_tol(phi, y, tol=1e-6),
    }.items():
        result = solve()
        assert result.converged, name
        error = float((result.w - reference).norm() / reference.norm())
        assert error < 1e-3, f"{name}: relative error {error:.2e}"


def test_warm_start_reduces_cg_iterations(problem: tuple[torch.Tensor, torch.Tensor]) -> None:
    phi, y = problem
    cold = block_cg(phi, y, tol=1e-8)
    warm = block_cg(phi, y, tol=1e-8, x0=cold.w + 1e-4 * torch.randn_like(cold.w))
    assert warm.iterations < cold.iterations


def test_lbfgs_convex_solves_ridge(problem: tuple[torch.Tensor, torch.Tensor]) -> None:
    phi, y = problem
    lam = effective_l2(phi, 1e-3)
    reference = direct_lstsq(phi, y).w

    def objective(w: torch.Tensor) -> torch.Tensor:
        return 0.5 * ((phi @ w - y) ** 2).sum() + 0.5 * lam * (w**2).sum()

    w, evals = lbfgs_convex(objective, torch.zeros_like(reference))
    assert evals > 0
    assert float((w - reference).norm() / reference.norm()) < 1e-3


def test_harness_utilities_run() -> None:
    state = machine_state(gate=False)
    assert "load_avg" in state and state["python"]
    assert peak_rss_mb() > 0
    timing = time_median(lambda: sum(range(1000)), repeats=3, warmup=1)
    assert timing.median_s >= 0 and len(timing.samples_s) == 3
