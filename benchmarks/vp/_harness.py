"""Measurement harness for the VP solver suite.

Timing discipline follows the W3 card convention from the companion paper
repo: refuse to start when the 1-minute load average is high or another
Python training process is alive, record the machine state in every result
JSON, and report medians of repeated timings rather than single runs.
"""

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

LOAD_GATE = 6.0  # macOS ambient load runs 3.5-4 with zero compute; see W3 amendment


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs (benchmark scripts may seed globals)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def git_short_sha() -> str:
    """Return the current commit's short SHA, or ``"untracked"``."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "untracked"
    return out.decode().strip()


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB (ru_maxrss is bytes on macOS, KB on Linux)."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 2**20 if sys.platform == "darwin" else raw / 2**10


def _other_python_training_alive() -> bool:
    """True when another python process with a heavyweight entrypoint is running.

    The launcher chain of this very process (shell, uv, tee, ...) repeats the
    benchmark module name on its command lines, so the whole ancestor chain is
    excluded before matching.
    """
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
        pid = int(parts[0])
        parents[pid] = int(parts[1])
        commands[pid] = parts[2]
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


def machine_state(*, gate: bool = True) -> dict[str, Any]:
    """Snapshot load/platform state; optionally refuse to run on a loaded machine.

    Raises:
        RuntimeError: When ``gate`` is set and the 1-minute load average
            exceeds ``LOAD_GATE`` or another Python training process is alive.
    """
    load1, load5, load15 = os.getloadavg()
    state: dict[str, Any] = {
        "load_avg": [round(load1, 2), round(load5, 2), round(load15, 2)],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "igl": getattr(igl, "__version__", "unknown"),
        "mps_available": torch.backends.mps.is_available(),
    }
    if gate:
        if load1 > LOAD_GATE:
            raise RuntimeError(f"machine-state gate: load average {load1:.1f} > {LOAD_GATE}; rerun on a quiet machine")
        # Retry the process scan: IDE test-discovery pytest processes flicker
        # for a few seconds and must not fail a multi-hour benchmark launch.
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


def time_median(fn: Callable[[], Any], *, repeats: int = 5, warmup: int = 1) -> Timing:
    """Run ``fn`` ``warmup + repeats`` times; return the median wall-clock of the repeats.

    The last invocation's return value is kept on ``Timing.result`` so callers
    can validate the computed output without re-running.
    """
    result = None
    for _ in range(warmup):
        result = fn()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - start)
    return Timing(median_s=median(samples), samples_s=samples, result=result)


def run_dir(experiment: str) -> Path:
    """Deterministic output directory ``results/benchmarks/vp/<experiment>/<sha>/``."""
    base = Path.cwd() / "results" / "benchmarks" / "vp" / experiment / git_short_sha()
    base.mkdir(parents=True, exist_ok=True)
    return base


def write_result(
    experiment: str,
    payload: dict[str, Any],
    *,
    name: str = "result.json",
    state: dict[str, Any] | None = None,
    wall_clock_s: float | None = None,
) -> Path:
    """Write one experiment result JSON with the house provenance envelope."""
    envelope: dict[str, Any] = {
        "experiment": experiment,
        "git_commit": git_short_sha(),
        "package_version": getattr(igl, "__version__", "unknown"),
        "machine_state": state or machine_state(gate=False),
        "wall_clock_s": wall_clock_s,
        **payload,
    }
    path = run_dir(experiment) / name
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
