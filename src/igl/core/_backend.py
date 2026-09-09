"""Device execution backends: one branch per device type (CPU, MPS, CUDA).

Everything in the training loop that has to be done differently per device
lives here, behind one small interface, so the trainer, the solver and the
kernel never test ``device.type`` themselves:

- ``CpuBackend`` keeps every CPU number bit-identical to the reference
  implementation (the SPD wrapper reproduces published EEG results, and the
  reproducibility tests pin its RNG-consumption profile): the least-squares
  solve, the loop Jacobian, CPU snapshots and the optimizer are exactly the
  pre-backend code paths.
- ``MpsBackend`` keeps every tensor on the device: the readout is solved by a
  Cholesky factorisation with one step of iterative refinement (MPS has no
  ``lstsq``, no ``eigh`` and no float64), failure is signalled by a device
  flag, snapshots stay on the device.
- ``CudaBackend`` is the MPS branch plus TF32 matmuls (the Gram matrix and the
  refinement stay in full precision), a fused AdamW, and factorisations that
  never check their ``info`` on the host. It performs no host-device
  synchronisation per batch; the trainer reads one stacked scalar tensor per
  epoch through :meth:`Backend.host_scalars`.

Select one with :func:`select_backend`; tests inject a counting double through
``MatryoshkaTrainer(backend=...)``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator, Iterable, Sequence
from typing import Literal, Protocol, cast

import torch
from torch import nn
from torch.optim import AdamW

from igl.core.solver import direct_solve_weights, ridge_solve_device
from igl.types import PrefixForward

BackendName = Literal["cpu", "mps", "cuda"]
KernelPath = Literal["reference", "fast"]


class Backend(Protocol):
    """The per-device branch of the training loop.

    Attributes:
        name: The device type this backend serves.
        kernel_path: Which formulation the Green kernel uses on this device.
    """

    name: BackendName
    kernel_path: KernelPath

    def ridge_solve(self, phi: torch.Tensor, y: torch.Tensor, *, l2: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve the Tikhonov least-squares readout; return ``(w, bad)``.

        ``bad`` is a 0-d boolean tensor (on ``w``'s device) that is true when
        the solve failed and ``w`` was replaced by zeros.
        """
        ...  # pragma: no cover  # Protocol method body

    def loss_accumulator(self, device: torch.device) -> torch.Tensor:
        """A 0-d tensor the epoch loss is summed into on the device."""
        ...  # pragma: no cover  # Protocol method body

    def host_scalars(self, values: Sequence[torch.Tensor | float]) -> list[float]:
        """Bring scalars to the host in ONE transfer; Python floats pass through."""
        ...  # pragma: no cover  # Protocol method body

    def snapshot(self, module: PrefixForward) -> dict[str, torch.Tensor]:
        """Copy the module's state for early-stopping restore."""
        ...  # pragma: no cover  # Protocol method body

    def restore(self, module: PrefixForward, snapshot: dict[str, torch.Tensor]) -> None:
        """Restore a :meth:`snapshot`."""
        ...  # pragma: no cover  # Protocol method body

    def make_optimizer(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float | torch.Tensor,
        weight_decay: float | None,
        capturable: bool = False,
    ) -> AdamW:
        """Build the AdamW optimizer for this device (``capturable`` and a tensor ``lr`` serve CUDA graph replay)."""
        ...  # pragma: no cover  # Protocol method body

    def precision(self) -> contextlib.AbstractContextManager[None]:
        """Context manager holding the matmul precision policy for a fit."""
        ...  # pragma: no cover  # Protocol method body

    def refresh_weights(self, phi: torch.Tensor, target: torch.Tensor, *, l2: float, current: torch.Tensor) -> torch.Tensor:
        """The end-of-epoch readout refresh: new weights, or ``current`` when the solve cannot be trusted."""
        ...  # pragma: no cover  # Protocol method body

    def inner_subset(self, n_samples: int, inner_n: int, device: torch.device) -> torch.Tensor | None:
        """Row indices of the inner-solve subset, or ``None`` for every row in natural order."""
        ...  # pragma: no cover  # Protocol method body


class _DeviceBackend:
    """Shared implementation of the on-device (MPS, CUDA) branches."""

    name: BackendName
    kernel_path: KernelPath = "fast"
    _accumulator_dtype: torch.dtype = torch.float64
    _check_errors: bool = False

    def ridge_solve(self, phi: torch.Tensor, y: torch.Tensor, *, l2: float) -> tuple[torch.Tensor, torch.Tensor]:
        return ridge_solve_device(phi, y, l2=l2, check_errors=self._check_errors)

    def refresh_weights(self, phi: torch.Tensor, target: torch.Tensor, *, l2: float, current: torch.Tensor) -> torch.Tensor:
        # No host guard: the solve neutralises non-finite inputs itself and
        # reports through the flag; a failed refresh keeps the last good readout.
        weights, bad = self.ridge_solve(phi, target, l2=l2)
        return torch.where(bad, current.to(weights.device), weights)

    def inner_subset(self, n_samples: int, inner_n: int, device: torch.device) -> torch.Tensor | None:
        """Skip the permutation when the subset is the whole set: the ridge solve does not depend on row order.

        A permutation of ``n`` rows is a device sort per batch plus a gather of
        every row, and it only shuffles what gets summed; the device branches
        take the rows as they are. The CPU branch keeps the permutation so its
        random-number consumption, and every bit of its arithmetic, stay as
        they were.
        """
        if inner_n >= n_samples:
            return None
        return torch.randperm(n_samples, device=device)[:inner_n]

    def loss_accumulator(self, device: torch.device) -> torch.Tensor:
        return torch.zeros((), dtype=self._accumulator_dtype, device=device)

    def host_scalars(self, values: Sequence[torch.Tensor | float]) -> list[float]:
        tensors = [v for v in values if isinstance(v, torch.Tensor)]
        transferred: list[float] = []
        if tensors:
            dtype = torch.float32 if self.name == "mps" else torch.float64
            stacked = torch.stack([t.detach().reshape(()).to(dtype) for t in tensors])
            transferred = cast(list[float], stacked.tolist())  # pyright: ignore[reportUnknownMemberType]
        out: list[float] = []
        for v in values:
            out.append(transferred.pop(0) if isinstance(v, torch.Tensor) else float(v))
        return out

    def snapshot(self, module: PrefixForward) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in module.state_dict().items()}

    def restore(self, module: PrefixForward, snapshot: dict[str, torch.Tensor]) -> None:
        module.load_state_dict(snapshot)

    def make_optimizer(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float | torch.Tensor,
        weight_decay: float | None,
        capturable: bool = False,
    ) -> AdamW:
        return AdamW(params, lr=lr, weight_decay=weight_decay) if weight_decay is not None else AdamW(params, lr=lr)

    def precision(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()


class CpuBackend:
    """The reference branch: bit-identical to the pre-backend code.

    ``threads`` caps torch's intra-op thread count for the duration of a fit
    (:meth:`precision`). Off by default: fewer threads are faster on small
    problems (an M4 Max runs the medium benchmark batch 13% faster with 6
    threads than with its default 12), but the thread count changes the order
    of BLAS reductions, so results are then no longer bit-identical to the
    default's.
    """

    name: BackendName = "cpu"
    kernel_path: KernelPath = "reference"

    def ridge_solve(self, phi: torch.Tensor, y: torch.Tensor, *, l2: float) -> tuple[torch.Tensor, torch.Tensor]:
        # direct_solve_weights keeps the reference semantics: it warns per call and
        # returns zeros on failure; the trainer's non-finite-loss check is the
        # divergence signal on this branch, so the flag is always clear.
        return direct_solve_weights(phi, y, l2=l2), torch.zeros((), dtype=torch.bool)

    def refresh_weights(self, phi: torch.Tensor, target: torch.Tensor, *, l2: float, current: torch.Tensor) -> torch.Tensor:
        # Verbatim reference guard: a diverged encoder means non-finite phi, and the
        # lstsq backend would crash inside LAPACK; keep the last good readout and
        # let the next epoch's non-finite-loss check raise cleanly.
        if torch.isfinite(phi).all() and torch.isfinite(target).all():
            return direct_solve_weights(phi, target, l2=l2)
        return current

    def inner_subset(self, n_samples: int, inner_n: int, device: torch.device) -> torch.Tensor | None:
        return torch.randperm(n_samples, device=device)[:inner_n]

    def loss_accumulator(self, device: torch.device) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float64, device=device)

    def host_scalars(self, values: Sequence[torch.Tensor | float]) -> list[float]:
        return [float(v.item()) if isinstance(v, torch.Tensor) else float(v) for v in values]

    def snapshot(self, module: PrefixForward) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

    def restore(self, module: PrefixForward, snapshot: dict[str, torch.Tensor]) -> None:
        module.load_state_dict(snapshot)

    def make_optimizer(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float | torch.Tensor,
        weight_decay: float | None,
        capturable: bool = False,
    ) -> AdamW:
        return AdamW(params, lr=lr, weight_decay=weight_decay) if weight_decay is not None else AdamW(params, lr=lr)

    def __init__(self, *, threads: int | None = None) -> None:
        self.threads = threads

    def precision(self) -> contextlib.AbstractContextManager[None]:
        if self.threads is None:
            return contextlib.nullcontext()
        return _thread_cap(self.threads)


@contextlib.contextmanager
def _thread_cap(threads: int) -> Generator[None]:
    """Set torch's intra-op thread count for the block and restore the previous value."""
    previous = torch.get_num_threads()
    torch.set_num_threads(max(1, threads))
    try:
        yield
    finally:
        torch.set_num_threads(previous)


class MpsBackend(_DeviceBackend):
    """Apple MPS: on-device Cholesky solve, float32 accumulation (MPS has no float64).

    The readout stays on the device on purpose: factoring the ``R × R`` system
    in float64 on the CPU is 3× faster in isolation, but the copy drains the
    asynchronous Metal queue every batch and the fit gets 10–20% slower
    (measured, ``benchmarks/device/REPORT.md``); the hybrid is kept for the
    one-off public solve only.
    """

    name: BackendName = "mps"
    _accumulator_dtype = torch.float32


class CudaBackend(_DeviceBackend):
    """CUDA: no host synchronisation per batch, TF32 matmuls, fused AdamW."""

    name: BackendName = "cuda"
    _check_errors = False

    def __init__(self, *, tf32: bool = True) -> None:
        self.tf32 = tf32

    def make_optimizer(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float | torch.Tensor,
        weight_decay: float | None,
        capturable: bool = False,
    ) -> AdamW:  # pragma: no cover  # CUDA only
        # ``capturable`` keeps the step counters on the device so the optimizer
        # step can be recorded into a CUDA graph (see ``igl.core._graph``).
        if weight_decay is not None:
            return AdamW(params, lr=lr, weight_decay=weight_decay, fused=True, capturable=capturable)
        return AdamW(params, lr=lr, fused=True, capturable=capturable)

    def precision(self) -> contextlib.AbstractContextManager[None]:
        return _tf32(enabled=self.tf32)


@contextlib.contextmanager
def _tf32(*, enabled: bool) -> Generator[None]:  # pragma: no cover  # CUDA only
    """Temporarily set the CUDA TF32 flags (matmul and cuDNN); restored on exit."""
    matmul_before = torch.backends.cuda.matmul.allow_tf32
    cudnn_before = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_before
        torch.backends.cudnn.allow_tf32 = cudnn_before


def select_backend(device: torch.device | str, *, tf32: bool = True, cpu_threads: int | None = None) -> Backend:
    """The backend for ``device``: ``CpuBackend``, ``MpsBackend`` or ``CudaBackend``.

    Args:
        device: The device the module lives on.
        tf32: Whether the CUDA backend enables TF32 matmuls during a fit.
        cpu_threads: Intra-op thread cap for the CPU backend during a fit
            (``None`` leaves torch's setting; results then stay bit-identical).

    Returns:
        A :class:`Backend`.
    """
    kind = torch.device(device).type
    if kind == "cuda":  # pragma: no cover  # CUDA only
        return CudaBackend(tf32=tf32)
    if kind == "mps":
        return MpsBackend()
    return cast(Backend, CpuBackend(threads=cpu_threads))


__all__ = ["Backend", "BackendName", "CpuBackend", "CudaBackend", "KernelPath", "MpsBackend", "select_backend"]
