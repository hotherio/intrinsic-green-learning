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


# ----------------------------------------------------------------------------------- plan v2: the scale protocol


def _smoke_dataset():  # noqa: ANN202
    from benchmarks.geometry.blocks._scale import smoke_set
    from benchmarks.geometry.fitting import Dataset

    rs = smoke_set()
    return Dataset.from_arrays(rs.name, rs.x, rs.y, d=4, task=rs.task, n_classes=rs.n_classes, tangent_dim=8)


def test_from_arrays_builds_the_pca_chart_and_tangent_proxy() -> None:
    ds = _smoke_dataset()
    assert ds.task == "classification" and ds.n_classes == 4
    assert ds.t["train"].shape == (1200, 4) and ds.extra["pca_basis"].shape == (16, 8)
    basis = ds.extra["pca_basis"]
    assert torch.allclose(basis.T @ basis, torch.eye(8), atol=1e-4)


def test_classification_fit_records_the_v2_row(tmp_path) -> None:  # noqa: ANN001
    import igl
    from benchmarks.geometry.fitting import FitSpec, calibrate_weight, fit_arm

    ds = _smoke_dataset()
    spec = FitSpec(steps_budget=20, batch_size=128, n_anchors=8, anchor_init="kmeans-keep")
    modules: dict[str, igl.IGLModule] = {}
    row = fit_arm(ds, spec, "none", seed=0, prune=False, modules=modules)
    assert 0.0 <= row["err_test"] <= 1.0 and row["n_test"] == 400 and row["anchor_init"] == "kmeans-keep"
    assert row["inner_batch_size"] == 4096 and row["steps"] == row["epochs"] * 10
    assert set(row["score_test"]["4"]) >= {"err", "nll", "wrong"} and row["phi"]["dead_columns"] == 0
    assert row["report_test"]["tan"]["k"] == 4 and "off_manifold_share" in row["report_test"]
    cal = calibrate_weight(modules["none"], ds.x["train"], ds.y["train"], k=4, loss=igl.CrossEntropyLoss(n_classes=4))
    assert cal["weight"] > 0 and cal["grad_task"] > 0
    conf = fit_arm(
        ds, FitSpec(steps_budget=20, batch_size=128, n_anchors=8, penalty_weight=cal["weight"]), "conf", seed=0, prune=False
    )
    assert conf["grad_ratios"] and all(r["ratio"] > 0 for r in conf["grad_ratios"])
    frozen = fit_arm(ds, spec, "none", seed=0, inputs="param", prune=False)
    assert frozen["encoder"] == "identity (frozen chart)" and frozen["param_chart"] == "PCA-4 chart"


def test_baselines_and_paired_tests_on_the_smoke_set() -> None:
    from benchmarks.geometry import baselines
    from benchmarks.geometry.fitting import FitSpec
    from benchmarks.geometry.report_scale import paired_tests

    ds = _smoke_dataset()
    lin = baselines.linear_probe(ds)
    head = baselines.mlp_head(ds, FitSpec(steps_budget=20, batch_size=128, n_anchors=8, max_dim=4), seed=0)
    assert 0.0 <= lin["err_test"] <= 1.0 and 0.0 <= head["err_test"] <= 1.0 and head["n_params"] > 0
    same = paired_tests([(lin["wrong"], lin["wrong"], 400)])
    assert same["delta"] == 0.0 and same["mcnemar_p"] == 1.0 and same["discordant"] == 0
    diff = paired_tests([(lin["wrong"], head["wrong"], 400)])
    assert diff["ci_lo"] <= diff["delta"] <= diff["ci_hi"]


def test_scale_block_runs_on_the_smoke_set(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    import sys

    from benchmarks.geometry.blocks import w1_gate

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["w1", "--smoke", "--no-gate", "--device", "cpu", "--label", "t", "--parts", "noise,h0", "--seeds", "0"]
    )
    w1_gate.main()
    files = sorted((tmp_path / "results" / "benchmarks" / "geometry" / "t" / "w1_gate").glob("*/*/seed*.json"))
    assert [f.parent.name for f in files] == ["frozen-pca", "none"]
