"""CUDA graph replay of the training batch step: the CUDA branch's answer to launch overhead.

After the synchronisation work the batch step on CUDA is launch-bound: about 250
kernels and 2.5–2.9 ms per batch at every problem size, the GPU idle most of the time.
Capturing the whole step once and replaying it turns those launches into one, which is
legal only because the step never touches the host (see the census in
``benchmarks/device/REPORT.md``).

The runner owns the static buffers the step reads (batch indices, inner-solve indices,
gate mask, and the learning rate as a device tensor the fused optimizer reads at replay,
so a scheduler step never forces a re-capture), runs the step eagerly for a few warm-up
batches, captures it on the next full batch, and replays it afterwards. Optionally the
step is first compiled with ``torch.compile`` so the captured graph holds fused kernels.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

__all__ = ["GraphRunner"]


class GraphRunner:
    """Static buffers plus capture-and-replay for one training step.

    Args:
        device: The CUDA device.
        lr: Initial learning rate; kept in :attr:`lr` as a device tensor.
        batch_size: Rows per full batch (the graph covers full batches only).
        inner_n: Rows of the inner-solve subset, or ``None`` when the subset is the
            whole training set (no index buffer needed).
        d_max: Latent width, the gate mask's length.
        use_graph: Capture and replay the step from a CUDA graph.
        use_compile: Compile the step with ``torch.compile`` (default mode, static
            shapes) before capture, or on its own when ``use_graph`` is ``False``.
        warmup: Eager runs before the capture (allocator and library handles warm).
    """

    def __init__(
        self,
        *,
        device: torch.device,
        lr: float,
        batch_size: int,
        inner_n: int | None,
        d_max: int,
        use_graph: bool = True,
        use_compile: bool = False,
        warmup: int = 2,
    ) -> None:
        # ``full`` with a Python scalar launches a fill kernel: no host-to-device copy.
        self.lr = torch.full((), float(lr), dtype=torch.float32, device=device)
        self.idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.inner_idx = None if inner_n is None else torch.zeros(inner_n, dtype=torch.long, device=device)
        self.mask = torch.zeros(d_max, device=device)
        self.use_graph = use_graph
        self.use_compile = use_compile
        self.disabled = False
        self.captures = 0
        self.replays = 0
        self.eager_runs = 0
        self._warmup = warmup
        self._graph: torch.cuda.CUDAGraph | None = None
        self._step: Callable[[], None] | None = None

    def bind(self, step: Callable[[], None]) -> None:
        """Attach the step closure (it must read only this runner's buffers and fit-constant tensors)."""
        self._step = step
        if self.use_compile:
            self._step = torch.compile(step, mode="default", dynamic=False)  # pyright: ignore[reportUnknownMemberType]

    def sync_lr(self, optimizer: torch.optim.Optimizer) -> None:
        """Copy a scheduler's new learning rate into the tensor the recorded step reads, and hand it back to the optimizer.

        Schedulers assign a fresh value to ``param_group["lr"]``; the captured
        kernels keep reading :attr:`lr`, so the value is copied over and the
        group is pointed back at the shared tensor. The optimizer and its
        scheduler are built with a plain float rate first, so the scheduler's
        base rates never alias this tensor.
        """
        for group in optimizer.param_groups:
            value = group["lr"]
            if value is self.lr:
                continue
            if isinstance(value, torch.Tensor):
                self.lr.copy_(value)
            else:
                self.lr.fill_(float(value))  # a scalar argument to a kernel, not a host copy
            group["lr"] = self.lr

    def run(self, *, idx: torch.Tensor, mask: torch.Tensor, inner_idx: torch.Tensor | None) -> None:
        """Fill the buffers and run the step: eagerly while warming up, then from the graph."""
        if self._step is None:
            raise RuntimeError("GraphRunner.run called before bind()")
        self.idx.copy_(idx)
        self.mask.copy_(mask)
        if self.inner_idx is not None and inner_idx is not None:
            self.inner_idx.copy_(inner_idx)
        if not self.use_graph:
            self._step()
            self.eager_runs += 1
            return
        if self._graph is not None:
            self._graph.replay()
            self.replays += 1
            return
        if self.eager_runs < self._warmup:
            self._step()
            self.eager_runs += 1
            return
        graph = self._capture(self._step)
        self._graph = graph
        self.captures += 1
        graph.replay()
        self.replays += 1

    def _capture(self, step: Callable[[], None]) -> Any:  # pragma: no cover  # CUDA only
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        return graph
