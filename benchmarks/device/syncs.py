"""Host-device synchronisation census per epoch, for every estimator and loss strategy.

Counts, from ``torch.profiler``, the operators that force the host to wait
for the device (``aten::_local_scalar_dense`` behind ``.item()`` and Python
truthiness, ``aten::_to_copy`` between devices) during one training epoch,
and on CUDA the warnings of ``torch.cuda.set_sync_debug_mode("warn")``.

Run with::

    IGL_BENCH_DEVICE=cuda python -m benchmarks.device.syncs
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import time
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

import igl
from benchmarks.device._harness import machine_state, resolve_device, sync, write_result
from benchmarks.device.workloads import PROBLEMS, make_config, make_data, make_module
from igl.spd.linalg import MatrixMethod

SYNC_OPS = ("aten::_local_scalar_dense", "aten::item", "aten::_to_copy", "aten::copy_")


def _fit_once(fit: Callable[[], Any], device: torch.device) -> dict[str, Any]:
    """Run ``fit`` under the profiler; return sync-op counts and, on CUDA, sync-debug warnings."""
    warn_count = 0
    sites: dict[str, int] = {}
    cuda_sync_cm: contextlib.AbstractContextManager[Any] = contextlib.nullcontext()
    if device.type == "cuda":
        torch.cuda.set_sync_debug_mode("warn")

        @contextlib.contextmanager
        def _count_warnings() -> Iterator[None]:
            nonlocal warn_count
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                yield
            for w in caught:
                if "synchroniz" not in str(w.message).lower():
                    continue
                warn_count += 1
                site = f"{'/'.join(w.filename.rsplit('/', 3)[-3:])}:{w.lineno}"
                sites[site] = sites.get(site, 0) + 1

        cuda_sync_cm = _count_warnings()
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    sync(device)
    with cuda_sync_cm, profile(activities=activities) as prof:
        fit()
        sync(device)
    if device.type == "cuda":
        torch.cuda.set_sync_debug_mode("default")
    counts = {name: 0 for name in SYNC_OPS}
    memcpy_dtoh = 0
    for event in prof.key_averages():
        if event.key in counts:
            counts[event.key] += event.count
        if "Memcpy DtoH" in event.key or "memcpy_dtoh" in event.key.lower():
            memcpy_dtoh += event.count
    return {
        "ops": counts,
        "memcpy_dtoh": memcpy_dtoh,
        "cuda_sync_warnings": warn_count,
        "cuda_sync_sites": dict(sorted(sites.items(), key=lambda kv: -kv[1])),
    }


def workloads(device: torch.device, *, epochs: int) -> dict[str, Callable[[], Any]]:
    problem = PROBLEMS["small"]
    x, y, x_val, y_val = make_data(problem, device)
    cfg = make_config(problem, epochs=epochs)
    n_epochs = epochs

    def classifier() -> None:
        module = make_module(problem, device, output_dim=2)
        igl.MatryoshkaTrainer(loss=igl.CrossEntropyLoss(n_classes=2), config=cfg).fit(module, x, y, x_val=x_val, y_val=y_val)

    def regressor() -> None:
        module = make_module(problem, device, output_dim=problem.input_dim)
        igl.MatryoshkaTrainer(loss=igl.MSELoss(), config=cfg).fit(module, x, x, x_val=x_val, y_val=x_val)

    def spectral() -> None:
        from igl.spectral import ConstantNullSpace, FourierCosineBasis, SpectralKernel

        kernel = SpectralKernel(
            latent_dim=problem.max_dim,
            bases=FourierCosineBasis(n_modes=8),
            n_anchors=problem.n_anchors,
            null_space=ConstantNullSpace(),
        )
        torch.manual_seed(0)
        module = igl.IGLModule(input_dim=problem.input_dim, max_dim=problem.max_dim, output_dim=2, kernel=kernel).to(device)
        igl.MatryoshkaTrainer(loss=igl.CrossEntropyLoss(n_classes=2), config=cfg).fit(module, x, y, x_val=x_val, y_val=y_val)

    def orthogonality() -> None:
        from igl.spd import OrthogonalityPenalty

        module = make_module(problem, device, output_dim=problem.input_dim)
        igl.MatryoshkaTrainer(loss=igl.MSELoss(), config=cfg).fit(
            module, x, x, x_val=x_val, y_val=x_val, extra_losses=[OrthogonalityPenalty(weight=0.1, every=1)]
        )

    def airm(method: MatrixMethod) -> None:
        from igl.data import make_spd_dataset
        from igl.spd import AIRMLoss, LogEigVectorizer

        covs, _ = make_spd_dataset(problem.n, d=4, seed=0)
        vec = torch.as_tensor(LogEigVectorizer().fit(covs.numpy()).transform(covs.numpy()), dtype=torch.float32).to(device)
        torch.manual_seed(0)
        module = igl.IGLModule(
            input_dim=vec.shape[1],
            max_dim=problem.max_dim,
            output_dim=vec.shape[1],
            n_anchors=problem.n_anchors,
            n_scales=problem.n_scales,
        ).to(device)
        igl.MatryoshkaTrainer(loss=AIRMLoss(latent_dim=4, matrix_method=method), config=cfg).fit(module, vec, vec)

    def autoencoder() -> None:
        # Estimator level: includes the sklearn wrapper's own data movement and post-fit reads.
        igl.IGLAutoencoder(
            max_dim=problem.max_dim,
            n_anchors=problem.n_anchors,
            n_scales=problem.n_scales,
            config=igl.IGLConfig(max_dim=problem.max_dim, matryoshka=cfg),
            device=str(device),
            random_state=0,
        ).fit(x.cpu().numpy())

    def distiller() -> None:
        igl.IGLDistiller(
            max_dim=problem.max_dim,
            config=igl.IGLConfig(max_dim=problem.max_dim, matryoshka=cfg),
            device=str(device),
            random_state=0,
        ).fit(x.cpu().numpy())

    del n_epochs
    return {
        "classifier_ce": classifier,
        "regressor_mse": regressor,
        "spectral_cosine": spectral,
        "orthogonality": orthogonality,
        "airm": functools.partial(airm, "eigh"),
        "airm_iterative": functools.partial(airm, "iterative"),
        "autoencoder_estimator": autoencoder,
        "distiller_estimator": distiller,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--only", default=None, help="comma-separated workload names")
    parser.add_argument("--no-gate", action="store_true")
    args = parser.parse_args()
    device = resolve_device(args.device)
    state = machine_state(device, gate=not args.no_gate)
    results: dict[str, Any] = {}
    start = time.perf_counter()
    batches_per_epoch = -(-PROBLEMS["small"].n // PROBLEMS["small"].batch_size)
    for name, fit in workloads(device, epochs=args.epochs).items():
        if args.only and name not in args.only.split(","):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            counts = _fit_once(fit, device)
        per_epoch = {k: v / args.epochs for k, v in counts["ops"].items()}
        results[name] = {**counts, "per_epoch": per_epoch, "batches_per_epoch": batches_per_epoch}
        print(
            f"{device.type:4s} {name:16s} per epoch: scalar_dense={per_epoch['aten::_local_scalar_dense']:6.1f} "
            f"to_copy={per_epoch['aten::_to_copy']:6.1f} memcpy_dtoh={counts['memcpy_dtoh'] / args.epochs:6.1f} "
            f"cuda_sync_warnings={counts['cuda_sync_warnings'] / args.epochs:6.1f}  ({batches_per_epoch} batches/epoch)",
            flush=True,
        )
        for site, n in list(counts["cuda_sync_sites"].items())[:8]:
            print(f"{'':21s} {n / args.epochs:6.1f}/epoch  {site}", flush=True)
    path = write_result(
        "syncs",
        {"epochs": args.epochs, "workloads": results},
        device=device,
        state=state,
        wall_clock_s=time.perf_counter() - start,
    )
    print(f"-> {path}")


if __name__ == "__main__":
    main()
