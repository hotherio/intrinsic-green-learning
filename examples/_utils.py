"""Tiny helpers shared by the bundled examples.

Anything reusable outside the examples lives in the library proper. The
contents of this module are intentionally minimal — its role is to keep the
synthetic scripts terse, not to grow into a parallel library surface.
"""

import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Libraries shouldn't seed Python globals (that surprises users); examples
    can, because the whole point of an example is reproducibility.
    """
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


def make_run_dir(example_name: str) -> Path:
    """Pick a deterministic directory under ``results/<example>/<sha>/``."""
    base = Path.cwd() / "results" / example_name / git_short_sha()
    base.mkdir(parents=True, exist_ok=True)
    return base


def save_history(directory: Path, history: object, *, name: str = "history.json") -> None:
    """Dump training history (any dataclass-like) as JSON."""
    if hasattr(history, "__dict__"):
        payload = {k: list(v) if isinstance(v, list) else v for k, v in vars(history).items()}
    else:
        payload = {"history": str(history)}
    (directory / name).write_text(json.dumps(payload, indent=2))


def save_curve(directory: Path, curve: dict[int, float], *, name: str = "curve.csv") -> None:
    """Save the dimension/loss curve as a 2-column CSV."""
    lines = ["k,loss"] + [f"{k},{v:.6g}" for k, v in sorted(curve.items())]
    (directory / name).write_text("\n".join(lines) + "\n")
