"""Result filing for the geometry suite, on top of the device harness.

Results go under ``results/benchmarks/geometry/<label>/<block>/<bed>/<arm>/seed<k>.json`` with
the same provenance envelope as the device suite (commit, package version, machine state).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from benchmarks.device._harness import (
    _json_default,
    git_short_sha,
    machine_state,
    resolve_device,
    set_seed,
    sync,
    time_median,
)

__all__ = [
    "machine_state",
    "memory_gate",
    "resolve_device",
    "result_path",
    "set_seed",
    "sync",
    "time_median",
    "write_block_result",
]

MIN_FREE_GB = 12.0


def free_memory_gb() -> float:
    """Free plus inactive memory in GB on macOS (``vm_stat``); on other systems ``/proc/meminfo``'s available memory."""
    import re
    import subprocess
    import sys

    if sys.platform == "darwin":
        out = subprocess.check_output(["vm_stat"], text=True)
        page = int(re.search(r"page size of (\d+)", out).group(1))  # type: ignore[union-attr]
        pages = 0
        for key in ("Pages free", "Pages inactive", "Pages speculative"):
            m = re.search(rf"{key}:\s+(\d+)", out)
            pages += int(m.group(1)) if m else 0
        return pages * page / 2**30
    try:
        text = Path("/proc/meminfo").read_text()
        m = re.search(r"MemAvailable:\s+(\d+) kB", text)
        return int(m.group(1)) / 2**20 if m else float("inf")
    except OSError:
        return float("inf")


def memory_gate(min_free_gb: float = MIN_FREE_GB) -> float:
    """Refuse to start when less than ``min_free_gb`` is free: two of this suite's jobs took the machine down once."""
    free = free_memory_gb()
    if free < min_free_gb:
        raise RuntimeError(f"memory gate: only {free:.1f} GB free (< {min_free_gb} GB); wait for other jobs to finish")
    return free


def result_path(block: str, bed: str, arm: str, seed: int, *, label: str | None = None) -> Path:
    """Deterministic path of one result file; the parent directories are created."""
    base = Path.cwd() / "results" / "benchmarks" / "geometry" / (label or git_short_sha()) / block / bed / arm
    base.mkdir(parents=True, exist_ok=True)
    return base / f"seed{seed}.json"


def write_block_result(
    block: str,
    bed: str,
    arm: str,
    seed: int,
    payload: dict[str, Any],
    *,
    device: torch.device | str = "cpu",
    state: dict[str, Any] | None = None,
    started_at: float | None = None,
    label: str | None = None,
) -> Path:
    """Write one result JSON with the provenance envelope and return its path."""
    envelope: dict[str, Any] = {
        "block": block,
        "bed": bed,
        "arm": arm,
        "seed": seed,
        "git_commit": git_short_sha(),
        "machine_state": state or machine_state(device, gate=False),
        "wall_clock_s": None if started_at is None else time.perf_counter() - started_at,
        **payload,
    }
    path = result_path(block, bed, arm, seed, label=label)
    path.write_text(json.dumps(envelope, indent=2, default=_json_default))
    return path
