"""Shared machinery of the scale blocks (W1, S0–S3): data sets, specs, the one fit-and-write routine.

Every scale block loads a real set by name (``--sets``), builds a :class:`~benchmarks.geometry.fitting.Dataset`
from its arrays with ``d = k`` (the latent width is a choice on real data, not a property of the set), runs
:func:`~benchmarks.geometry.fitting.fit_arm` under the v2 protocol and writes one JSON per (set, R, k, arm, seed)
under ``results/benchmarks/geometry/<label>/<block>/<set>-R<R>-k<k>/<arm>/seed<s>.json``. Under ``--smoke`` the
set is a synthetic stand-in (no download, a few seconds on CPU) so the blocks are exercised by the test suite.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import replace
from typing import Any

import torch

import igl
from benchmarks.device._harness import machine_state
from benchmarks.geometry._harness import write_block_result
from benchmarks.geometry.fitting import Dataset, FitSpec, calibrate_weight, fit_arm, loss_for
from benchmarks.geometry.testbeds import real
from igl.data import embed_in_high_dim

__all__ = [
    "STEPS_BUDGET",
    "add_scale_args",
    "base_spec",
    "calibrated_weight",
    "load_set",
    "resolve_device",
    "run_fit",
    "subsample",
    "tag",
]

STEPS_BUDGET = 12_000
"""Optimiser steps of one fit at batch 256 whatever ``N`` (50 epochs at 60k rows, 200 at 16k, 400 at 8k)."""
SMOKE_STEPS = 24


def add_scale_args(p: argparse.ArgumentParser, *, default_sets: str = "mnist") -> argparse.ArgumentParser:
    """The options shared by the scale blocks."""
    p.add_argument("--sets", default=default_sets, help=f"comma-separated set names among {sorted(real.LOADERS)}")
    p.add_argument("--k", default="32", help="comma-separated latent widths (max_dim)")
    p.add_argument("--anchors", default=None, help="comma-separated anchor counts R (default: the block's ladder)")
    p.add_argument("--anchor-init", default="kmeans-keep", help="random | kmeans-keep | kmeans-median")
    p.add_argument("--steps", type=int, default=STEPS_BUDGET, help="optimiser steps per fit (constant across N)")
    return p


def resolve_device(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    """The device and the machine state, applying the load gate unless ``--no-gate`` or ``--smoke``."""
    import os

    device = args.device or os.environ.get("IGL_BENCH_DEVICE", "cpu")
    state = machine_state(device, gate=not args.no_gate and not args.smoke)
    return device, state


def smoke_set(*, seed: int = 0) -> real.RealSet:
    """A synthetic stand-in for a real set: the flat torus embedded in 16-D, four quadrant classes, 1200 / 200 / 400 rows."""
    gen = torch.Generator().manual_seed(seed)
    theta = torch.rand(1800, 2, generator=gen) * (2 * math.pi)
    x4 = torch.cat([theta.cos(), theta.sin()], dim=1)
    x = embed_in_high_dim(x4, target_dim=16, seed=seed) + 0.01 * torch.randn(1800, 16, generator=gen)
    y = ((theta[:, 0] > math.pi).long() * 2 + (theta[:, 1] > math.pi).long()).long()
    xs = {"train": x[:1200], "val": x[1200:1400], "test": x[1400:]}
    ys = {"train": y[:1200], "val": y[1200:1400], "test": y[1400:]}
    return real.RealSet(name="smoke", task="classification", x=xs, y=ys, n_classes=4, seed=seed)


def load_set(name: str, *, k: int, smoke: bool) -> Dataset:
    """The named real set as a fitting dataset with latent width ``k`` (the smoke stand-in under ``--smoke``)."""
    rs = smoke_set() if smoke else real.LOADERS[name]()
    return Dataset.from_arrays(rs.name, rs.x, rs.y, d=k, task=rs.task, n_classes=rs.n_classes)


def base_spec(ds: Dataset, *, n_anchors: int, k: int, smoke: bool, args: argparse.Namespace, **overrides: Any) -> FitSpec:
    """The v2 protocol spec for one cell; ``overrides`` are applied last."""
    spec = FitSpec(
        n_train=int(ds.x["train"].shape[0]),
        n_val=int(ds.x["val"].shape[0]),
        n_test=int(ds.x["test"].shape[0]),
        steps_budget=SMOKE_STEPS if smoke else int(args.steps),
        batch_size=128 if smoke else 256,
        n_anchors=n_anchors,
        max_dim=k,
        anchor_init=str(args.anchor_init),
        task=ds.task,
        n_classes=ds.n_classes,
    )
    return replace(spec, **overrides) if overrides else spec


def tag(set_name: str, n_anchors: int, k: int) -> str:
    """The results directory of a cell."""
    return f"{set_name}-R{n_anchors}-k{k}"


def subsample(ds: Dataset, n_train: int, *, seed: int) -> Dataset:
    """The dataset with ``n_train`` training rows drawn once per seed (validation and test untouched)."""
    n = int(ds.x["train"].shape[0])
    if n_train >= n:
        return ds
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:n_train]
    return replace(
        ds,
        x={**ds.x, "train": ds.x["train"][idx]},
        y={**ds.y, "train": ds.y["train"][idx]},
        t={**ds.t, "train": ds.t["train"][idx]},
    )


def calibrated_weight(
    ds: Dataset, spec: FitSpec, *, seed: int, device: str, fraction: float = 0.1, beta: float = 1.0
) -> dict[str, float]:
    """The conformality weight calibrated at ``fraction`` of the budget of a penalty-free fit (the v2 rule)."""
    short = replace(spec, steps_budget=max(1, math.ceil((spec.steps_budget or STEPS_BUDGET) * fraction)))
    modules: dict[str, igl.IGLModule] = {}
    fit_arm(ds, short, "none", seed=seed, device=device, prune=False, modules=modules)
    module = modules["none"]
    dev = torch.device(device)
    x, y = ds.x["train"].to(dev), ds.y["train"].to(dev)
    inner = spec.inner_batch_size or max(4096, 16 * spec.n_anchors)
    cal = calibrate_weight(module, x, y, k=ds.d, beta=beta, loss=loss_for(spec), inner=inner, l2=spec.source_l2)
    cal["fraction"] = fraction
    return cal


def run_fit(
    block: str,
    ds: Dataset,
    spec: FitSpec,
    arm: str,
    *,
    seed: int,
    device: str,
    state: dict[str, Any],
    label: str | None,
    set_name: str,
    inputs: str = "ambient",
    result_arm: str | None = None,
    extra: dict[str, Any] | None = None,
    lr_readout: bool = False,
) -> dict[str, Any]:
    """One fit of the protocol, written under the cell's tag; returns the row."""
    t0 = time.perf_counter()
    row = fit_arm(ds, spec, arm, seed=seed, device=device, inputs=inputs, prune=False, lr_readout=lr_readout)
    row["set"] = set_name
    row["k"] = spec.max_dim or ds.d
    if extra:
        row.update(extra)
    name = result_arm or arm
    row["arm"] = name
    write_block_result(
        block, tag(set_name, spec.n_anchors, row["k"]), name, seed, row, device=device, state=state, started_at=t0, label=label
    )
    metric = f"err {row['err_test']:.4f}" if ds.task == "classification" else f"nrmse {row['nrmse_test_dmax']:.4f}"
    flag = " UNSTABLE" if (row.get("stability") or {}).get("flag") else ""
    print(
        f"{set_name:18s} R={spec.n_anchors:<4d} k={row['k']:<3d} {name:16s} seed {seed} {metric} | "
        f"conf {row['report_test']['full']['sqrt_conf']:.3f} eff_rank {row['phi']['eff_rank']:.1f} "
        f"best {row['best_epoch']}/{row['epochs']}{flag} | {row['wall_s']:.0f}s",
        flush=True,
    )
    return row
