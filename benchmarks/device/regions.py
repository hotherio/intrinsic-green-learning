"""Per-batch region timer: where does a training batch spend its time on this device?

Regions: outer forward (encoder + kernel on the batch), inner encode, inner
kernel, readout solve, loss + backward + optimizer step. Every region is
bracketed by device syncs so GPU numbers measure device work.

Run with::

    IGL_BENCH_DEVICE=mps python -m benchmarks.device.regions [--problems small,medium]
"""

from __future__ import annotations

import argparse
import time

import torch

import igl
from benchmarks.device._harness import machine_state, resolve_device, sync, write_result
from benchmarks.device.workloads import PROBLEMS, make_data, make_module
from igl.core.normalization import normalize_phi
from igl.core.solver import direct_solve_weights

REGIONS = ("outer_fwd", "inner_encode", "inner_kernel", "solve", "loss_bwd_step")


def time_regions(problem_name: str, device: torch.device, *, batches: int = 24, warmup: int = 4) -> dict[str, float]:
    """Median-free cumulative timing of each region over ``batches`` batches (after ``warmup``)."""
    problem = PROBLEMS[problem_name]
    x, y, _, _ = make_data(problem, device)
    module = make_module(problem, device)
    loss = igl.CrossEntropyLoss(n_classes=2)
    params = [*module.encoder.parameters(), *module.green.parameters(), module.bias]
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    totals = dict.fromkeys(REGIONS, 0.0)
    n = x.shape[0]
    k = max(1, problem.max_dim // 2)
    mask = torch.zeros(problem.max_dim, device=device)
    mask[:k] = 1.0
    module.train()
    for it in range(warmup + batches):
        idx = torch.randperm(n, device=device)[: problem.batch_size]
        sync(device)
        t0 = time.perf_counter()
        z = module.encoder(x[idx])
        phi = normalize_phi(module.green(z * mask, gate_mask=mask), module.normalize)
        sync(device)
        t1 = time.perf_counter()
        with torch.no_grad():
            inner_idx = torch.randperm(n, device=device)[: problem.inner_batch_size]
            z_inner = module.encoder(x[inner_idx]) * mask
            sync(device)
            t2 = time.perf_counter()
            phi_inner = normalize_phi(module.green(z_inner, gate_mask=mask), module.normalize)
            sync(device)
            t3 = time.perf_counter()
            w = direct_solve_weights(phi_inner, loss.target(y[inner_idx]) - module.bias.detach(), l2=1e-3).to(device)
            sync(device)
            t4 = time.perf_counter()
        out = phi @ w + module.bias
        batch_loss = loss.loss(out, loss.target(y[idx]))
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        float(batch_loss.item())
        sync(device)
        t5 = time.perf_counter()
        if it >= warmup:
            for name, value in zip(REGIONS, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4), strict=True):
                totals[name] += value
    per_batch_ms = {name: value / batches * 1e3 for name, value in totals.items()}
    per_batch_ms["total"] = sum(per_batch_ms.values())
    return per_batch_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--problems", default="small,medium,large")
    parser.add_argument("--no-gate", action="store_true")
    args = parser.parse_args()
    device = resolve_device(args.device)
    state = machine_state(device, gate=not args.no_gate)
    results: dict[str, dict[str, float]] = {}
    start = time.perf_counter()
    for name in args.problems.split(","):
        results[name] = time_regions(name, device)
        row = "  ".join(f"{r}={results[name][r]:7.2f}" for r in (*REGIONS, "total"))
        print(f"{device.type:4s} {name:6s} ms/batch: {row}", flush=True)
    path = write_result("regions", {"problems": results}, device=device, state=state, wall_clock_s=time.perf_counter() - start)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
