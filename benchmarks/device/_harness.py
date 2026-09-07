"""Measurement harness for the device suite.

Timing discipline: refuse to start on a loaded machine, record the machine
state in every result JSON, report medians of repeated timings, and
synchronise the device before and after every timed region so GPU timings
measure device work rather than kernel launches.
"""

from __future__ import annotations

import json
import os
import platform
import random
import resource
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch

import igl

LOAD_GATE = 6.0  # macOS ambient load runs 3.5-4 with zero compute


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs (benchmark scripts may seed globals)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def git_short_sha() -> str:
    """Return the label results are filed under: ``IGL_BENCH_LABEL`` if set, else the commit's short SHA.

    The override lets the suite measure an older library (``PYTHONPATH`` pointing
    at a worktree of that commit) from the current checkout and file the
    results under that commit.
    """
    label = os.environ.get("IGL_BENCH_LABEL")
    if label:
        return label
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "untracked"
    return out.decode().strip()


def resolve_device(name: str | None = None) -> torch.device:
    """Pick the benchmark device: the argument, else ``IGL_BENCH_DEVICE``, else ``cpu``."""
    chosen = name or os.environ.get("IGL_BENCH_DEVICE", "cpu")
    device = torch.device(chosen)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False")
    return device


def sync(device: torch.device | str) -> None:
    """Block until every queued kernel on ``device`` has finished (no-op on CPU)."""
    kind = torch.device(device).type
    if kind == "cuda":
        torch.cuda.synchronize()
    elif kind == "mps":
        torch.mps.synchronize()


def device_memory_mb(device: torch.device | str) -> float | None:
    """Peak allocated device memory in MB since the last reset, when the backend tracks it."""
    kind = torch.device(device).type
    if kind == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if kind == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    return None


def reset_device_memory(device: torch.device | str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB (ru_maxrss is bytes on macOS, KB on Linux)."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 2**20 if sys.platform == "darwin" else raw / 2**10


def _other_python_training_alive() -> bool:
    """True when another python process with a heavyweight entrypoint is running (ancestors excluded)."""
    try:
        out = subprocess.check_output(["ps", "-axo", "pid,ppid,command"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    parents: dict[int, int] = {}
    commands: dict[int, str] = {}
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        parents[int(parts[0])] = int(parts[1])
        commands[int(parts[0])] = parts[2]
    ancestors = {os.getpid()}
    node = os.getpid()
    while node in parents and parents[node] not in ancestors and parents[node] > 1:
        node = parents[node]
        ancestors.add(node)
    for pid, command in commands.items():
        if pid in ancestors:
            continue
        if "python" in command and any(marker in command for marker in ("train", "probe_", "benchmarks.", "pytest")):
            return True
    return False


def machine_state(device: torch.device | str = "cpu", *, gate: bool = True) -> dict[str, Any]:
    """Snapshot load/platform/device state; optionally refuse to run on a loaded machine."""
    load1, load5, load15 = os.getloadavg()
    kind = torch.device(device).type
    state: dict[str, Any] = {
        "load_avg": [round(load1, 2), round(load5, 2), round(load15, 2)],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "igl": getattr(igl, "__version__", "unknown"),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if kind == "cuda" else platform.processor() or platform.machine(),
    }
    if kind == "cuda":
        state["cuda"] = torch.version.cuda
        state["tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
    if gate:
        for attempt in range(6):
            load1 = os.getloadavg()[0]
            if load1 <= LOAD_GATE:
                break
            if attempt == 5:
                raise RuntimeError(f"machine-state gate: load average {load1:.1f} > {LOAD_GATE}; rerun on a quiet machine")
            time.sleep(30)
        state["load_avg"] = [round(load1, 2), *state["load_avg"][1:]]
        for attempt in range(3):
            if not _other_python_training_alive():
                break
            if attempt == 2:
                raise RuntimeError("machine-state gate: another python training/benchmark process is running")
            time.sleep(5)
    return state


@dataclass(slots=True)
class Timing:
    """Median-of-k timing of one callable, with the per-repeat samples kept."""

    median_s: float
    samples_s: list[float]
    result: Any = field(repr=False, default=None)


def time_median(fn: Callable[[], Any], *, device: torch.device | str = "cpu", repeats: int = 5, warmup: int = 1) -> Timing:
    """Run ``fn`` ``warmup + repeats`` times with device syncs; return the median wall-clock."""
    result = None
    for _ in range(warmup):
        result = fn()
    sync(device)
    samples: list[float] = []
    for _ in range(repeats):
        sync(device)
        start = time.perf_counter()
        result = fn()
        sync(device)
        samples.append(time.perf_counter() - start)
    return Timing(median_s=median(samples), samples_s=samples, result=result)


def run_dir(device: torch.device | str) -> Path:
    """Deterministic output directory ``results/benchmarks/device/<sha>/<device>/``."""
    base = Path.cwd() / "results" / "benchmarks" / "device" / git_short_sha() / torch.device(device).type
    base.mkdir(parents=True, exist_ok=True)
    return base


def write_result(
    experiment: str,
    payload: dict[str, Any],
    *,
    device: torch.device | str,
    state: dict[str, Any] | None = None,
    wall_clock_s: float | None = None,
) -> Path:
    """Write one experiment result JSON with the provenance envelope."""
    envelope: dict[str, Any] = {
        "experiment": experiment,
        "git_commit": git_short_sha(),
        "package_version": getattr(igl, "__version__", "unknown"),
        "machine_state": state or machine_state(device, gate=False),
        "wall_clock_s": wall_clock_s,
        **payload,
    }
    path = run_dir(device) / f"{experiment}.json"
    path.write_text(json.dumps(envelope, indent=2, default=_json_default))
    return path


def _json_default(value: object) -> object:
    if isinstance(value, np.floating | np.integer):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)
