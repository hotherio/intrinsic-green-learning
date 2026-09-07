"""Smoke tests keeping the device benchmark suite importable and runnable.

The suite itself is run manually (never in CI); these tests only make sure
its scripts still execute on a tiny problem after library changes.
"""

from __future__ import annotations

import torch

from benchmarks.device._harness import machine_state, peak_rss_mb, resolve_device, time_median
from benchmarks.device.regions import REGIONS, time_regions
from benchmarks.device.workloads import PROBLEMS, make_config, make_data, make_module


def test_harness_utilities_run() -> None:
    state = machine_state("cpu", gate=False)
    assert state["device"] == "cpu"
    assert peak_rss_mb() > 0
    timing = time_median(lambda: torch.zeros(1), device="cpu", repeats=2, warmup=1)
    assert timing.median_s >= 0
    assert resolve_device("cpu").type == "cpu"


def test_region_timer_runs_on_a_tiny_problem() -> None:
    result = time_regions("small", torch.device("cpu"), batches=1, warmup=0)
    assert set(result) == {*REGIONS, "total"}
    assert all(v >= 0 for v in result.values())


def test_workload_factories_agree_on_shapes() -> None:
    problem = PROBLEMS["small"]
    x, y, x_val, y_val = make_data(problem, torch.device("cpu"))
    module = make_module(problem, torch.device("cpu"))
    assert x.shape == (problem.n, problem.input_dim)
    assert y.shape == (problem.n,)
    assert x_val.shape[1] == problem.input_dim and y_val.shape[0] == x_val.shape[0]
    assert module.max_dim == problem.max_dim
    assert make_config(problem, epochs=1).epochs == 1
