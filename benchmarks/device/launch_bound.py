"""What removes the launch overhead: CUDA graphs and ``torch.compile`` around one batch step.

The HEAD profile shows the batch step launch-bound on the H100 (about 250 kernels and
2.9 ms per batch at every problem size, the GPU busy 12% of the epoch). This script
rebuilds the trainer's batch step from the library's pieces (inner design matrix and
device readout solve under ``no_grad``, outer forward, loss, backward, fused AdamW) with
static buffers, then times it eagerly, replayed from a captured CUDA graph, and compiled
with ``torch.compile`` in its default and ``reduce-overhead`` modes. CUDA only.

    IGL_BENCH_DEVICE=cuda python -m benchmarks.device.launch_bound
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from typing import Any

import torch

import igl
from benchmarks.device._harness import machine_state, resolve_device, sync, write_result
from benchmarks.device.workloads import PROBLEMS, make_data, make_module
from igl.core.solver import full_precision_matmul, ridge_solve_device


def _timeit(fn: Callable[[], Any], device: torch.device, *, repeats: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    sync(device)
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def build_step(problem_name: str, device: torch.device) -> tuple[Callable[[], torch.Tensor], dict[str, torch.Tensor]]:
    """One Matryoshka VP batch step over static buffers; returns it and the buffers to refill per batch."""
    problem = PROBLEMS[problem_name]
    x, y, _, _ = make_data(problem, device)
    module = make_module(problem, device, output_dim=2)
    loss = igl.CrossEntropyLoss(n_classes=2)
    target_all = loss.target(y)
    params = [p for n, p in module.named_parameters() if n != "source_weights"]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4, fused=True, capturable=True)
    inner_idx = torch.arange(min(problem.inner_batch_size, problem.n), device=device)
    outer_idx = torch.zeros(problem.batch_size, dtype=torch.long, device=device)
    gate = torch.ones(problem.max_dim, device=device)
    l2 = 1e-3

    def step() -> torch.Tensor:
        with torch.no_grad():
            phi = module.design_matrix(x.index_select(0, inner_idx), gate_mask=gate)
            t = target_all.index_select(0, inner_idx) - module.bias
            with full_precision_matmul(device):
                w, bad = ridge_solve_device(phi, t, l2=l2)
            module.source_weights.copy_(torch.where(bad, module.source_weights, w))
        pred = module(x.index_select(0, outer_idx), gate_mask=gate)
        task = loss.loss(pred, target_all.index_select(0, outer_idx))
        task.backward()
        opt.step()
        opt.zero_grad(set_to_none=False)
        return task.detach()

    return step, {"outer_idx": outer_idx, "gate": gate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--problem", default="medium")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--no-gate", action="store_true")
    args = parser.parse_args()
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise SystemExit("this experiment is CUDA only")
    state = machine_state(device, gate=not args.no_gate)
    start = time.perf_counter()
    torch.manual_seed(0)
    n = PROBLEMS[args.problem].n
    batch = PROBLEMS[args.problem].batch_size

    def refill(buffers: dict[str, torch.Tensor]) -> None:
        buffers["outer_idx"].copy_(torch.randint(0, n, (batch,), device=device))
        k = int(torch.randint(1, buffers["gate"].numel() + 1, (1,)).item())
        buffers["gate"].zero_()
        buffers["gate"][:k] = 1.0

    results: dict[str, float | str] = {}

    # Eager.
    step, buffers = build_step(args.problem, device)

    def eager() -> None:
        refill(buffers)
        step()

    results["eager_ms"] = _timeit(eager, device, repeats=args.repeats)
    print(f"cuda {args.problem:7s} eager           {results['eager_ms']:7.3f} ms/batch", flush=True)

    # CUDA graph capture of the whole step (warm-up on a side stream as the docs prescribe).
    step, buffers = build_step(args.problem, device)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            refill(buffers)
            step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_loss = step()

    def replay() -> None:
        refill(buffers)
        graph.replay()

    results["cuda_graph_ms"] = _timeit(replay, device, repeats=args.repeats)
    print(
        f"cuda {args.problem:7s} cuda graph      {results['cuda_graph_ms']:7.3f} ms/batch  (loss {static_loss.item():.4f})",
        flush=True,
    )

    # torch.compile: fusion only, then fusion + cudagraph trees.
    for mode in ("default", "reduce-overhead"):
        step, buffers = build_step(args.problem, device)
        compiled = torch.compile(step, mode=mode)
        t0 = time.perf_counter()
        try:
            for _ in range(3):
                refill(buffers)
                torch.compiler.cudagraph_mark_step_begin()  # cudagraph trees: outputs of the previous step may be reused
                compiled()
            sync(device)
        except RuntimeError as exc:  # cudagraph trees reject a step that carries its own backward (torch 2.8)
            results[f"compile_{mode.replace('-', '_')}_error"] = str(exc).splitlines()[0][:200]
            print(f"cuda {args.problem:7s} compile {mode:15s} not measured: {str(exc).splitlines()[0][:90]}", flush=True)
            torch._dynamo.reset()  # pyright: ignore[reportPrivateUsage]
            continue
        compile_s = time.perf_counter() - t0

        def run(compiled: Callable[[], torch.Tensor] = compiled, buffers: dict[str, torch.Tensor] = buffers) -> None:
            refill(buffers)
            torch.compiler.cudagraph_mark_step_begin()
            compiled()

        key = f"compile_{mode.replace('-', '_')}_ms"
        results[key] = _timeit(run, device, repeats=args.repeats)
        results[f"{key}_first_calls_s"] = compile_s
        print(
            f"cuda {args.problem:7s} compile {mode:15s} {results[key]:7.3f} ms/batch  (first 3 calls {compile_s:.1f} s)",
            flush=True,
        )

    path = write_result(
        f"launch_bound_{args.problem}",
        {"problem": args.problem, **results},
        device=device,
        state=state,
        wall_clock_s=time.perf_counter() - start,
    )
    print(f"-> {path}")


if __name__ == "__main__":
    main()
