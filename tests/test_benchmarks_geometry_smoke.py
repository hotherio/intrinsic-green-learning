"""Smoke tests keeping the geometry benchmark suite importable and runnable on tiny settings.

The suite itself is run manually (never in CI); these tests only make sure its test beds, measures
and blocks still execute after library changes. Each test stays well under a few seconds on CPU.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from benchmarks.geometry import charts, geometry_report
from benchmarks.geometry.blocks import g1_rho_law, g2_gauge, g4_oracle_floor
from benchmarks.geometry.testbeds import graphs, hills, manifolds, spd_beds, targets


def test_hill_port_reproduces_the_deck_at_full_height() -> None:
    r = hills.analyse("round", 1.0, charts=False, n_born=12)
    assert r["rho"] == pytest.approx(0.507, abs=0.01)
    assert r["depth"]["1e-3"] == 8
    assert r["rank_f"] == 1 and r["rank_fhat"] == 4 and r["rank_u"] == 3
    assert r["r_g"] == 13 and r["bound"] == 40


def test_graph_bed_matches_the_hill_in_two_dimensions_and_runs_in_three() -> None:
    r2 = graphs.analyse("radial", 2, 1.0, n=25, n_born=6)
    assert 0.3 < r2["rho"] < 0.7
    r3 = graphs.analyse("generic", 3, 0.5, n=9, n_born=4)
    assert r3["d"] == 3 and len(r3["tt_rank_u"]) == 2
    assert r3["naive_error"] > 0


def test_weyl_and_cotton_shares_of_known_manifolds() -> None:
    torch.manual_seed(0)
    t, _ = manifolds.product_sphere_flat(2, 2).sample(1, seed=0)
    assert manifolds.weyl_share(manifolds.product_sphere_flat(2, 2), t[0])["share"] == pytest.approx(1 / 3, abs=1e-6)
    t, _ = manifolds.sphere_cap(4).sample(1, seed=0)
    assert manifolds.weyl_share(manifolds.sphere_cap(4), t[0])["share"] == pytest.approx(0.0, abs=1e-8)
    t, _ = manifolds.sphere_cap(3).sample(1, seed=0)
    assert manifolds.cotton_share(manifolds.sphere_cap(3), t[0])["share"] == pytest.approx(0.0, abs=1e-6)


def test_spd_airm_bed_is_conformally_flat_for_n_2_and_flat_in_log_euclidean() -> None:
    bed = spd_beds.spd_bed(2)
    t, _ = bed.sample(1, seed=1)
    assert manifolds.cotton_share(bed, t[0])["share"] == pytest.approx(0.0, abs=1e-6)
    flat = spd_beds.spd_bed(2, metric="logeuclid")
    c = manifolds.curvature(flat, t[0])
    assert float(c["riem"].abs().max()) == pytest.approx(0.0, abs=1e-9)


def test_grid_targets_and_chart_measures_on_a_small_bed() -> None:
    bed = manifolds.sphere_cap(2)
    gt = targets.grid_poisson(bed, n=11)
    t, _ = bed.sample(200, seed=0)
    u = gt.u_at(t)
    assert u.shape == (200,) and torch.isfinite(u).all()
    assert charts.chart_rank(t, u, points=9)["rank"] >= 1
    bc = charts.bump_count(t, u, max_atoms=8, centres_per_axis=3)
    assert len(bc["curve"]) <= 8


def test_blocks_run_on_tiny_settings() -> None:
    r = g1_rho_law.torus_chart_distortion(4, "incoherent", 0.05, n_points=50, n_modes=3)
    assert r["mean_dev"] > 0
    case = g2_gauge.one_case(3, "additive", n=7, deltas=[0.01])
    assert case["identity_fd_residual"] < 0.1
    info = g4_oracle_floor.optimise(manifolds.hill_surface("round"), beta=1.0, n_points=32, steps=3)
    assert 0 <= info["sqrt_floor"] <= info["start"] ** 0.5 + 1e-9


def test_geometry_report_on_a_linear_chart() -> None:
    encoder = nn.Linear(5, 3)
    with torch.no_grad():
        encoder.weight.copy_(torch.tensor([[1.0, 0, 0, 0, 0], [0, 2.0, 0, 0, 0], [0, 0, 1.0, 0, 0]]))
    x = torch.randn(16, 5)
    rep = geometry_report.report(encoder, x, k=3, output_dim=3)
    assert rep["full"]["sqrt_conf"] == pytest.approx(0.0, abs=1e-6)  # diagonal metric: zero after axis normalisation
    assert rep["full"]["sqrt_conf_plain"] > 0.1
    assert rep["full"]["sqrt_orth"] == pytest.approx(0.0, abs=1e-6)
