"""Shared command-line conventions of the blocks: ``--smoke``, ``--seeds``, ``--device``, ``--label``, ``--out``."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.device._harness import git_short_sha


def parser(description: str) -> argparse.ArgumentParser:
    """Argument parser with the options every block accepts."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--smoke", action="store_true", help="tiny setting, under a minute on CPU")
    p.add_argument("--seeds", default="0,1,2", help="comma-separated seeds for learned-chart blocks")
    p.add_argument("--device", default=None, help="cpu, mps or cuda (default: IGL_BENCH_DEVICE or cpu)")
    p.add_argument("--label", default=None, help="results label (default: IGL_BENCH_LABEL or the short SHA)")
    p.add_argument("--no-gate", action="store_true", help="do not refuse to run on a loaded machine")
    return p


def gate(args: argparse.Namespace) -> None:
    """Apply the memory gate unless ``--no-gate`` (the load gate is applied by learned-chart blocks only)."""
    if not args.no_gate:
        from benchmarks.geometry._harness import memory_gate

        memory_gate()


def seeds_of(args: argparse.Namespace) -> list[int]:
    """Seeds requested on the command line, one in smoke mode."""
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    return seeds[:1] if args.smoke else seeds


def figure_dir(block: str, label: str | None = None) -> Path:
    """Directory of the block's figures, next to its results."""
    base = Path.cwd() / "results" / "benchmarks" / "geometry" / (label or git_short_sha()) / block / "figures"
    base.mkdir(parents=True, exist_ok=True)
    return base


__all__ = ["figure_dir", "gate", "parser", "seeds_of"]
