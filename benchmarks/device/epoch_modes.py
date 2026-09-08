"""Epoch wall time of ``MatryoshkaTrainer.fit`` per execution mode on CUDA.

Modes: eager (``cuda_graphs=False``), graph (the default on CUDA), compile
(``torch_compile=True`` without the graph), graph+compile. One warm-up fit, then the
median epoch of a multi-epoch fit, device-synced. The region timer cannot see inside a
replayed graph, so this is the measurement for the graph and compile work.

    IGL_BENCH_DEVICE=cuda python -m benchmarks.device.epoch_modes
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any

import torch

import igl
from benchmarks.device._harness import machine_state, resolve_device, sync, write_result
from benchmarks.device.workloads import PROBLEMS, make_config, make_data, make_module

MODES: dict[str, dict[str, bool]] = {
    "eager": {"cuda_graphs": False, "torch_compile": False},
    "graph": {"cuda_graphs": True, "torch_compile": False},
    "compile": {"cuda_graphs": False, "torch_compile": True},
    "graph+compile": {"cuda_graphs": True, "torch_compile": True},
}


def time_mode(problem_name: str, device: torch.device, mode: str, *, epochs: int) -> dict[str, Any]:
    problem = PROBLEMS[problem_name]
    x, y, x_val, y_val = make_data(problem, device)
    torch.manual_seed(0)
    module = make_module(problem, device)
    cfg = make_config(problem, epochs=epochs, **MODES[mode])
    trainer = igl.MatryoshkaTrainer(loss=igl.CrossEntropyLoss(n_classes=2), config=cfg)
    stamps: list[float] = []

    def on_epoch(_: igl.EpochStats) -> None:
        sync(device)
        stamps.append(time.perf_counter())

    t0 = time.perf_counter()
    history = trainer.fit(module, x, y, x_val=x_val, y_val=y_val, on_epoch=on_epoch)
    sync(device)
    first_fit_s = time.perf_counter() - t0
    stamps = [t0, *stamps]
    epoch_ms = [(b - a) * 1e3 for a, b in zip(stamps[:-1], stamps[1:], strict=True)]
    return {
        "mode": mode,
        "first_fit_s": first_fit_s,
        "first_epoch_ms": epoch_ms[0],
        "epoch_ms_median_after_first": statistics.median(epoch_ms[1:]) if len(epoch_ms) > 1 else epoch_ms[0],
        "final_train_loss": float(history.train_loss[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--problems", default="small,medium,large")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--no-gate", action="store_true")
    args = parser.parse_args()
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise SystemExit("epoch modes are a CUDA measurement")
    state = machine_state(device, gate=not args.no_gate)
    start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for problem in args.problems.split(","):
        for mode in MODES:
            row = {"problem": problem, **time_mode(problem, device, mode, epochs=args.epochs)}
            rows.append(row)
            print(
                f"cuda {problem:7s} {mode:14s} epoch {row['epoch_ms_median_after_first']:8.2f} ms "
                f"(first epoch {row['first_epoch_ms']:8.2f} ms, first fit {row['first_fit_s']:.1f} s)",
                flush=True,
            )
    path = write_result(
        "epoch_modes",
        {"epochs": args.epochs, "rows": rows},
        device=device,
        state=state,
        wall_clock_s=time.perf_counter() - start,
    )
    print(f"-> {path}")


if __name__ == "__main__":
    main()
