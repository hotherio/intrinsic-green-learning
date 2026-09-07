"""Assemble Markdown tables from the device-suite result JSONs.

Run with::

    python -m benchmarks.device.report [--commits <sha_before>,<sha_after>]

Reads ``results/benchmarks/device/<sha>/<device>/*.json`` and prints tables
comparing devices and commits; paste the output into REPORT.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path.cwd() / "results" / "benchmarks" / "device"


def load(sha: str, device: str, experiment: str) -> dict[str, Any] | None:
    path = ROOT / sha / device / f"{experiment}.json"
    return json.loads(path.read_text()) if path.exists() else None


def table(headers: list[str], rows: list[list[Any]]) -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.2f}" if abs(v) >= 0.01 else f"{v:.2e}"
        return str(v)

    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commits", required=True, help="comma-separated short SHAs, oldest first")
    parser.add_argument("--devices", default="cpu,mps,cuda")
    args = parser.parse_args()
    shas = args.commits.split(",")
    devices = args.devices.split(",")

    print("## End-to-end examples (wall seconds)\n")
    rows: list[list[Any]] = []
    for device in devices:
        per_sha = [load(sha, device, "e2e") for sha in shas]
        if not any(per_sha):
            continue
        names = {r["module"].rsplit(".", 1)[-1] for res in per_sha if res for r in res["examples"]}
        for name in sorted(names):
            row: list[Any] = [device, name]
            for res in per_sha:
                found = next((r for r in (res["examples"] if res else []) if r["module"].endswith(name)), None)
                row.append(f"{found['wall_s']:.1f}" if found else "-")
            first = next((r for res in per_sha if res for r in res["examples"] if r["module"].endswith(name)), None)
            row.append(first["headlines"] if first else "")
            rows.append(row)
    print(table(["device", "example", *shas, "headlines (first)"], rows))

    print("\n## Per-batch regions (ms per batch)\n")
    rows = []
    for device in devices:
        for sha in shas:
            res = load(sha, device, "regions")
            if not res:
                continue
            for problem, regions in res["problems"].items():
                rows.append(
                    [
                        device,
                        sha,
                        problem,
                        *[regions[k] for k in ("outer_fwd", "inner_encode", "inner_kernel", "solve", "loss_bwd_step", "total")],
                    ]
                )
    print(
        table(
            ["device", "commit", "problem", "outer_fwd", "inner_encode", "inner_kernel", "solve", "loss_bwd_step", "total"],
            rows,
        )
    )

    print("\n## Host synchronisations per epoch\n")
    rows = []
    for device in devices:
        for sha in shas:
            res = load(sha, device, "syncs")
            if not res:
                continue
            for name, wl in res["workloads"].items():
                if "error" in wl:
                    rows.append([device, sha, name, "-", "-", "-", "-", f"skipped: {wl['error'][:60]}"])
                    continue
                sites = wl.get("cuda_sync_sites", {})
                top = ", ".join(f"{site} ({n / res['epochs']:g})" for site, n in list(sites.items())[:3])
                rows.append(
                    [
                        device,
                        sha,
                        name,
                        wl["per_epoch"]["aten::_local_scalar_dense"],
                        wl["per_epoch"]["aten::_to_copy"],
                        wl["memcpy_dtoh"] / res["epochs"],
                        wl["cuda_sync_warnings"] / res["epochs"],
                        top,
                    ]
                )
    print(
        table(
            [
                "device",
                "commit",
                "workload",
                "scalar_dense/epoch",
                "to_copy/epoch",
                "memcpy DtoH/epoch",
                "cuda sync warnings/epoch",
                "sites (per epoch)",
            ],
            rows,
        )
    )

    print("\n## Components\n")
    for device in devices:
        for sha in shas:
            res = load(sha, device, "components")
            if not res:
                continue
            print(f"\n### {device} @ {sha}\n")
            print(
                table(
                    ["d", "fwd current", "fwd einsum", "fwd+bwd current", "fwd+bwd einsum", "rel err"],
                    [
                        [
                            r["d"],
                            r["fwd_ms_current"],
                            r["fwd_ms_einsum"],
                            r["fwdbwd_ms_current"],
                            r["fwdbwd_ms_einsum"],
                            r["max_rel_err_einsum"],
                        ]
                        for r in res["kernel"]
                    ],
                )
            )
            print()
            print(
                table(
                    ["n", "r", "c", "method", "ms", "rel pred err"],
                    [
                        [r["n"], r["r"], r["c"], r["method"], r.get("ms") or r.get("error"), r.get("max_rel_pred_err", "")]
                        for r in res["solver"]
                    ],
                )
            )
            print()
            print(
                table(
                    ["batch", "input_dim", "latent_dim", "loop ms", "vmap ms", "|Δloss|"],
                    [
                        [
                            r["batch"],
                            r["input_dim"],
                            r["latent_dim"],
                            r["fwdbwd_ms_loop"],
                            r["fwdbwd_ms_vmap"],
                            r["abs_loss_diff"],
                        ]
                        for r in res["jacobian"]
                    ],
                )
            )


if __name__ == "__main__":
    main()
