"""End-to-end examples on one device: wall time, memory, and the headline numbers as a regression guard.

Every bundled example is run as a subprocess with ``IGL_EXAMPLE_DEVICE`` set,
its stdout is kept, and the numbers the example prints (d_eff, accuracy, R²,
MSE, KL, hierarchy) are parsed so a speedup can never hide a regression.

Run with::

    IGL_BENCH_DEVICE=cuda python -m benchmarks.device.e2e [--only moons_xor,poisson_1d]
"""

from __future__ import annotations

import argparse
import os
import re
import resource
import subprocess
import sys
import time
from typing import Any

from benchmarks.device._harness import machine_state, resolve_device, write_result

EXAMPLES = (
    "examples.synthetic.moons_xor",
    "examples.synthetic.swiss_roll_recon",
    "examples.synthetic.torus_classification",
    "examples.synthetic.whitened_regression",
    "examples.synthetic.poisson_1d",
    "examples.synthetic.save_load",
)
_HEADLINES = (
    re.compile(r"d_eff\s*=\s*(?P<d_eff>\d+)"),
    re.compile(r"(?:val_acc|accuracy)\s*=\s*(?P<acc>[0-9.]+)"),
    re.compile(r"R²\s*=\s*(?P<r2>[0-9.]+)"),
    re.compile(r"MSE\s*=\s*(?P<mse>[0-9.eE+-]+)"),
    re.compile(r"hierarchy_holds:\s*(?P<hierarchy>True|False)"),
    re.compile(r"KL=(?P<kl>[0-9.]+)"),
    re.compile(r"round-trip max abs difference:\s*(?P<roundtrip>[0-9.eE+-]+)"),
)


def _headlines(stdout: str) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for line in stdout.splitlines():
        for pattern in _HEADLINES:
            for match in pattern.finditer(line):
                for key, value in match.groupdict().items():
                    if value is not None:
                        found.setdefault(key, []).append(value)
    return found


def run_example(module: str, device: str) -> dict[str, Any]:
    env = {**os.environ, "IGL_EXAMPLE_DEVICE": device, "PYTHONWARNINGS": "ignore"}
    rss_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    start = time.perf_counter()
    proc = subprocess.run([sys.executable, "-m", module], env=env, capture_output=True, text=True, check=False)
    wall = time.perf_counter() - start
    rss_after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    rss_mb = (rss_after if rss_after > rss_before else rss_before) / (2**20 if sys.platform == "darwin" else 2**10)
    return {
        "module": module,
        "returncode": proc.returncode,
        "wall_s": wall,
        "child_peak_rss_mb": rss_mb,
        "headlines": _headlines(proc.stdout),
        "stderr_tail": proc.stderr[-800:] if proc.returncode else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--only", default=None, help="comma-separated example short names")
    parser.add_argument("--no-gate", action="store_true")
    args = parser.parse_args()
    device = resolve_device(args.device)
    state = machine_state(device, gate=not args.no_gate)
    start = time.perf_counter()
    results: list[dict[str, Any]] = []
    for module in EXAMPLES:
        short = module.rsplit(".", 1)[-1]
        if args.only and short not in args.only.split(","):
            continue
        result = run_example(module, str(device))
        results.append(result)
        status = "ok" if result["returncode"] == 0 else f"FAILED rc={result['returncode']}"
        print(
            f"{device.type:4s} {short:22s} {result['wall_s']:7.1f} s  rss {result['child_peak_rss_mb']:6.0f} MB  "
            f"{status}  {result['headlines']}",
            flush=True,
        )
        if result["returncode"]:
            print(result["stderr_tail"])
    path = write_result("e2e", {"examples": results}, device=device, state=state, wall_clock_s=time.perf_counter() - start)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
