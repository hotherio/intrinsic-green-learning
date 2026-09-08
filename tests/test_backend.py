"""Tests for the per-device execution backends (:mod:`igl.core._backend`)."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pytest
import torch

import igl
from igl import IGLConvergenceError, IGLModule, MatryoshkaConfig, MatryoshkaTrainer, MSELoss, direct_solve_weights
from igl.core._backend import CpuBackend, MpsBackend
from igl.core.solver import ridge_solve_device
from igl.device import select_backend

_DEVICES = ["cpu", *(["mps"] if torch.backends.mps.is_available() else [])]


def _design(n: int, r: int, c: int, *, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    module = IGLModule(input_dim=16, max_dim=8, output_dim=c, n_anchors=r, n_scales=4)
    with torch.no_grad():
        phi = module.design_matrix(torch.randn(n, 16))
    return phi, torch.randn(n, c)


def test_select_backend_by_device_type() -> None:
    assert isinstance(select_backend("cpu"), CpuBackend)
    assert select_backend(torch.device("cpu")).name == "cpu"
    if torch.backends.mps.is_available():
        assert isinstance(select_backend("mps"), MpsBackend)


@pytest.mark.parametrize(("n", "r", "c"), [(1024, 32, 2), (2048, 64, 64)])
def test_device_solve_matches_lstsq_predictions(n: int, r: int, c: int) -> None:
    """Cholesky + one refinement step reproduces the QR solve's predictions to 1e-4."""
    phi, y = _design(n, r, c)
    reference = direct_solve_weights(phi, y, l2=1e-3)
    weights, bad = ridge_solve_device(phi, y, l2=1e-3)
    assert not bool(bad)
    rel = ((phi @ weights - phi @ reference).abs().max() / (phi @ reference).abs().max()).item()
    assert rel < 1e-4, rel


def test_device_solve_flags_non_finite_input_without_raising() -> None:
    phi, y = _design(256, 16, 3)
    phi[3, 4] = float("nan")
    weights, bad = ridge_solve_device(phi, y, l2=1e-3)
    assert bool(bad)
    assert torch.equal(weights, torch.zeros_like(weights))


def test_device_solve_accepts_one_dimensional_targets() -> None:
    phi, y = _design(128, 8, 1)
    weights, bad = ridge_solve_device(phi, y[:, 0], l2=1e-3)
    assert weights.shape == (8, 1)
    assert not bool(bad)


def test_cpu_backend_flag_is_always_clear() -> None:
    phi, y = _design(64, 8, 2)
    weights, bad = CpuBackend().ridge_solve(phi, y, l2=1e-3)
    assert weights.shape == (8, 2)
    assert not bool(bad)


def test_device_backend_refresh_keeps_current_weights_when_the_solve_fails() -> None:
    phi, y = _design(64, 8, 2)
    current = torch.full((8, 2), 7.0)
    phi[0, 0] = float("inf")
    kept = MpsBackend().refresh_weights(phi, y, l2=1e-3, current=current)
    assert torch.equal(kept, current)
    clean_phi, clean_y = _design(64, 8, 2)
    healthy = MpsBackend().refresh_weights(clean_phi, clean_y, l2=1e-3, current=current)
    assert not torch.equal(healthy, current)


def test_host_scalars_passes_floats_through_and_reads_tensors_once() -> None:
    backend = MpsBackend()
    out = backend.host_scalars([torch.tensor(2.5), 1.0, torch.tensor(3, dtype=torch.int64), float("inf")])
    assert out == [2.5, 1.0, 3.0, float("inf")]
    assert CpuBackend().host_scalars([torch.tensor(1.5), 2.0]) == [1.5, 2.0]


class _Counting(CpuBackend):
    """CPU backend that counts host transfers, for the sync-property tests."""

    def __init__(self) -> None:
        super().__init__()
        self.transfers = 0

    def host_scalars(self, values: Sequence[torch.Tensor | float]) -> list[float]:
        self.transfers += 1
        return super().host_scalars(values)


def _fit(backend: object | None, *, seed: int = 0, epochs: int = 3) -> IGLModule:
    torch.manual_seed(seed)
    x = torch.randn(96, 6)
    y = x[:, :2] + 0.1 * torch.randn(96, 2)
    torch.manual_seed(seed)
    module = IGLModule(input_dim=6, max_dim=3, output_dim=2, n_anchors=8, n_scales=2)
    cfg = MatryoshkaConfig(epochs=epochs, batch_size=32, inner_batch_size=64, early_stop_patience=None, verbose=False)
    MatryoshkaTrainer(loss=MSELoss(), config=cfg, backend=backend).fit(module, x, y)  # type: ignore[arg-type]
    return module


def test_injected_cpu_backend_is_bit_identical_to_the_default() -> None:
    counting = _Counting()
    a = _fit(None)
    b = _fit(counting)
    assert torch.equal(a.source_weights, b.source_weights)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters(), strict=True))
    assert counting.transfers == 3  # exactly one host transfer per epoch


def test_device_branch_calls_item_only_for_the_sampler(mocker: object) -> None:
    """On the device branch the package's only `.item()` per batch is the CPU sampler's draw.

    Only calls made from this package's own frames are counted: torch's
    optimizer keeps its step counters on the CPU and reads them with `.item()`
    regardless of the device, which is not a device synchronisation.
    """
    import sys

    original = torch.Tensor.item
    calls: list[str] = []

    def spy(self: torch.Tensor) -> object:
        caller = sys._getframe(1).f_code.co_filename  # noqa: SLF001
        if "/igl/" in caller:
            calls.append(caller.rsplit("/", 1)[-1])
        return original(self)

    getattr(mocker, "patch").object(torch.Tensor, "item", new=spy)  # noqa: B009
    epochs, n, batch = 2, 96, 32
    _fit(_DeviceStyle(), epochs=epochs)
    batches = -(-n // batch)
    assert calls == ["sampler.py"] * (epochs * batches), calls


class _DeviceStyle(MpsBackend):
    """The on-device branch exercised on CPU tensors (MPS has no CPU-only CI)."""

    name = "cpu"  # type: ignore[assignment]


def test_device_branch_trains_on_cpu_tensors() -> None:
    module = _fit(_DeviceStyle(), epochs=5)
    x = torch.randn(32, 6)
    with torch.no_grad():
        out = module(x)
    assert torch.isfinite(out).all()


def test_readout_failure_raises_unless_batches_may_be_skipped() -> None:
    class _Failing(MpsBackend):
        name = "cpu"  # type: ignore[assignment]

        def ridge_solve(self, phi: torch.Tensor, y: torch.Tensor, *, l2: float) -> tuple[torch.Tensor, torch.Tensor]:
            weights, _ = super().ridge_solve(phi, y, l2=l2)
            return torch.zeros_like(weights), torch.tensor(data=True)

    torch.manual_seed(0)
    x = torch.randn(64, 4)
    module = IGLModule(input_dim=4, max_dim=2, output_dim=1, n_anchors=6, n_scales=2)
    cfg = MatryoshkaConfig(epochs=1, batch_size=32, inner_batch_size=64, early_stop_patience=None, verbose=False)
    with pytest.raises(IGLConvergenceError, match="readout solve failed"):
        MatryoshkaTrainer(loss=MSELoss(), config=cfg, backend=_Failing()).fit(module, x, x[:, :1])
    cfg_skip = MatryoshkaConfig(
        epochs=1, batch_size=32, inner_batch_size=64, early_stop_patience=None, verbose=False, skip_failing_batches=True
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        MatryoshkaTrainer(loss=MSELoss(), config=cfg_skip, backend=_Failing()).fit(module, x, x[:, :1])
    assert any("readout solve failed" in str(w.message) for w in caught)


@pytest.mark.parametrize("device", _DEVICES)
def test_estimator_fits_and_predicts_on_every_available_device(device: str) -> None:
    from igl.data import embed_in_high_dim, make_moons

    x_2d, y = make_moons(200, noise=0.1, seed=0)
    x = embed_in_high_dim(x_2d, target_dim=8, seed=1).numpy()
    cfg = igl.IGLConfig(matryoshka=MatryoshkaConfig(epochs=5, batch_size=64, inner_batch_size=160, early_stop_patience=None))
    clf = igl.IGLClassifier(max_dim=4, n_anchors=12, n_scales=2, random_state=0, config=cfg, device=device).fit(x, y.numpy())
    assert clf.score(x, y.numpy()) > 0.6


@pytest.mark.parametrize("device", _DEVICES)
def test_reconstruction_estimators_fit_on_every_available_device(device: str) -> None:
    """The distiller whitens targets on the device; the autoencoder reads its curve there."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((160, 6)).astype(np.float32)
    cfg = igl.IGLConfig(
        max_dim=3, matryoshka=MatryoshkaConfig(epochs=2, batch_size=64, inner_batch_size=160, early_stop_patience=None)
    )
    distiller = igl.IGLDistiller(max_dim=3, config=cfg, random_state=0, device=device).fit(x)
    assert distiller.reconstruct(x).shape == x.shape
    auto = igl.IGLAutoencoder(max_dim=3, n_anchors=8, n_scales=2, config=cfg, random_state=0, device=device).fit(x)
    assert auto.transform(x).shape == (160, 3)


def test_cuda_backend_precision_context_toggles_tf32_and_restores() -> None:
    """The TF32 flags are plain torch globals, so the context is testable without a GPU."""
    from igl.device import CudaBackend

    before = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with CudaBackend(tf32=True).precision():
            assert torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cuda.matmul.allow_tf32
        with CudaBackend(tf32=False).precision():
            assert not torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = before


def test_gate_mask_table_matches_the_per_batch_masks() -> None:
    from igl.core.trainer import _gate_masks

    masks = _gate_masks(5, torch.device("cpu"))
    assert masks.shape == (6, 5)
    for k in range(6):
        expected = torch.zeros(5)
        expected[:k] = 1.0
        assert torch.equal(masks[k], expected)


def test_inner_subset_is_a_permutation_on_cpu_and_skipped_on_devices_when_whole() -> None:
    from igl.device import CpuBackend, MpsBackend

    torch.manual_seed(0)
    cpu_idx = CpuBackend().inner_subset(10, 10, torch.device("cpu"))
    assert cpu_idx is not None and sorted(cpu_idx.tolist()) == list(range(10))
    assert MpsBackend().inner_subset(10, 10, torch.device("cpu")) is None
    partial = MpsBackend().inner_subset(10, 4, torch.device("cpu"))
    assert partial is not None and partial.shape == (4,) and len(set(partial.tolist())) == 4


def test_hybrid_solve_matches_the_reference_and_flags_bad_input() -> None:
    from igl.core.solver import direct_solve_weights, ridge_solve_hybrid

    torch.manual_seed(0)
    module = IGLModule(input_dim=6, max_dim=3, output_dim=2, n_anchors=16, n_scales=3)
    x = torch.randn(512, 6)
    with torch.no_grad():
        phi = module.design_matrix(x)
    y = torch.randn(512, 2)
    w_ref = direct_solve_weights(phi, y, l2=1e-3)
    w, bad = ridge_solve_hybrid(phi, y, l2=1e-3)
    assert not bool(bad)
    pred_ref = phi @ w_ref
    assert float((phi @ w - pred_ref).abs().max() / pred_ref.abs().max()) < 1e-4
    w1, bad1 = ridge_solve_hybrid(phi, y[:, 0], l2=1e-3)
    assert w1.shape == (phi.shape[1], 1) and not bool(bad1)
    bad_phi = phi.clone()
    bad_phi[3, 2] = float("nan")
    w_bad, flag = ridge_solve_hybrid(bad_phi, y, l2=1e-3)
    assert bool(flag) and torch.isfinite(w_bad).all() and torch.equal(w_bad, torch.zeros_like(w_bad))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS only")
def test_mps_backend_solves_through_the_hybrid_path() -> None:
    from igl.core.solver import direct_solve_weights
    from igl.device import MpsBackend

    torch.manual_seed(0)
    module = IGLModule(input_dim=6, max_dim=3, output_dim=2, n_anchors=16, n_scales=3)
    x = torch.randn(512, 6)
    with torch.no_grad():
        phi = module.design_matrix(x)
    y = torch.randn(512, 2)
    w_ref = direct_solve_weights(phi, y, l2=1e-3)
    w, bad = MpsBackend().ridge_solve(phi.to("mps"), y.to("mps"), l2=1e-3)
    assert w.device.type == "mps" and bad.device.type == "mps" and not bool(bad)
    pred_ref = phi @ w_ref
    assert float((phi @ w.cpu() - pred_ref).abs().max() / pred_ref.abs().max()) < 1e-4


def test_cpu_thread_cap_applies_during_the_fit_and_restores() -> None:
    from igl.device import CpuBackend, select_backend

    before = torch.get_num_threads()
    backend = select_backend("cpu", cpu_threads=max(1, before - 1))
    assert isinstance(backend, CpuBackend) and backend.threads == max(1, before - 1)
    with backend.precision():
        assert torch.get_num_threads() == max(1, before - 1)
    assert torch.get_num_threads() == before
    with CpuBackend().precision():
        assert torch.get_num_threads() == before


def test_batch_size_defaults_per_device_and_round_trips() -> None:
    cfg = MatryoshkaConfig()
    assert cfg.batch_size is None
    assert cfg.batch_size_for("cpu") == 256 and cfg.batch_size_for("mps") == 256 and cfg.batch_size_for("cuda") == 1024
    assert MatryoshkaConfig(batch_size=64).batch_size_for("cuda") == 64
    data = igl.IGLConfig(matryoshka=MatryoshkaConfig(cpu_threads=6)).to_dict()
    back = igl.IGLConfig.from_dict(data)
    assert back.matryoshka.batch_size is None and back.matryoshka.cpu_threads == 6
    back2 = igl.IGLConfig.from_dict(igl.IGLConfig(matryoshka=MatryoshkaConfig(batch_size=128)).to_dict())
    assert back2.matryoshka.batch_size == 128
