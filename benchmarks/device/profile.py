"""Op-level profile of one training epoch, with a Chrome trace per device.

Run with::

    IGL_BENCH_DEVICE=cuda python -m benchmarks.device.profile [--problem medium]
"""

from __future__ import annotations

import argparse
import time
import warnings
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

import igl
from benchmarks.device._harness import machine_state, resolve_device, run_dir, sync, write_result
from benchmarks.device.workloads import PROBLEMS, make_config, make_data, make_module


def profile_epoch(problem_name: str, device: torch.device, *, rows: int = 25) -> dict[str, Any]:
    problem = PROBLEMS[problem_name]
    x, y, x_val, y_val = make_data(problem, device)
    module = make_module(problem, device)
    trainer = igl.MatryoshkaTrainer(loss=igl.CrossEntropyLoss(n_classes=2), config=make_config(problem, epochs=1))
    trainer.fit(module, x, y, x_val=x_val, y_val=y_val)  # warm-up epoch (allocator, kernels)
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    sync(device)
    wall0 = time.perf_counter()
    with profile(activities=activities, record_shapes=False, profile_memory=device.type == "cuda") as prof:
        trainer.fit(module, x, y, x_val=x_val, y_val=y_val)
        sync(device)
    wall = time.perf_counter() - wall0
    averages = prof.key_averages()
    sort_key = "self_device_time_total" if device.type == "cuda" else "self_cpu_time_total"
    table = averages.table(sort_by=sort_key, row_limit=rows)
    top: list[dict[str, Any]] = []
    for event in sorted(averages, key=lambda e: getattr(e, sort_key, 0.0), reverse=True)[:rows]:
        top.append(
            {
                "op": event.key,
                "count": event.count,
                "self_cpu_ms": event.self_cpu_time_total / 1e3,
                "self_device_ms": getattr(event, "self_device_time_total", 0.0) / 1e3,
            }
        )
    device_busy_ms = sum(getattr(e, "self_device_time_total", 0.0) for e in averages) / 1e3 if device.type == "cuda" else None
    trace = run_dir(device) / f"trace_{problem_name}.json"
    prof.export_chrome_trace(str(trace))
    return {
        "problem": problem_name,
        "wall_ms": wall * 1e3,
        "device_busy_ms": device_busy_ms,
        "device_busy_fraction": (device_busy_ms / (wall * 1e3)) if device_busy_ms is not None else None,
        "kernel_launches": sum(e.count for e in averages if e.device_type.name != "CPU") if device.type == "cuda" else None,
        "peak_device_mem_mb": (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None,
        "top_ops": top,
        "table": table,
        "trace": str(trace),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--problem", default="medium")
    parser.add_argument("--no-gate", action="store_true")
    args = parser.parse_args()
    device = resolve_device(args.device)
    state = machine_state(device, gate=not args.no_gate)
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = profile_epoch(args.problem, device)
    print(result["table"])
    print(f"wall {result['wall_ms']:.0f} ms; device busy {result['device_busy_ms']} ms; launches {result['kernel_launches']}")
    path = write_result(f"profile_{args.problem}", result, device=device, state=state, wall_clock_s=time.perf_counter() - start)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
