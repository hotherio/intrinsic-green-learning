"""Tests for the per-device execution backends (:mod:`igl.core._backend`)."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

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
    assert counting.transfers >= 3  # once per epoch for the readout failure count


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
