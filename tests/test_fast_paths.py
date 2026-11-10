"""The expanded kernel path, the float64 hybrid solve, the opt-in CPU Cholesky and the all-prefix penalty."""

from __future__ import annotations

import pytest
import torch

import igl
from igl.core import solver
from igl.core._backend import CpuBackend, MpsBackend, select_backend
from igl.core.kernel import GreenKernel, _QuadraticLogKernel  # pyright: ignore[reportPrivateUsage]
from igl.exceptions import IGLConfigError
from igl.kernels._registry import get_operator, register_operator
from igl.spd.orthogonality import conformality_loss, conformality_loss_all_prefixes


def test_quadratic_log_kernel_gradients_are_exact() -> None:
    torch.manual_seed(0)
    z = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    a = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    g = torch.rand(3, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(_QuadraticLogKernel.apply, (z, a, g), eps=1e-6, atol=1e-6)  # pyright: ignore[reportUnknownMemberType]


@pytest.mark.parametrize("operator", ["gaussian", ("gaussian", "laplacian"), ("gaussian", "helmholtz")])
@pytest.mark.parametrize("per_scale", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_expanded_path_equals_the_reference_in_float64(operator: str | tuple[str, ...], per_scale: bool, masked: bool) -> None:
    torch.manual_seed(0)
    kernel = GreenKernel(6, n_anchors=9, n_scales=4, operator=operator, per_scale_columns=per_scale).double()
    mask = torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.float64) if masked else None
    z = torch.randn(50, 6, dtype=torch.float64)
    z = z * mask if mask is not None else z
    values: dict[str, torch.Tensor] = {}
    grads: dict[str, torch.Tensor] = {}
    for path in ("reference", "expanded"):
        kernel.zero_grad()
        z_in = z.clone().requires_grad_(True)
        phi = kernel(z_in, gate_mask=mask, path=path)
        (phi**2).sum().backward()  # pyright: ignore[reportUnknownMemberType]
        values[path] = phi.detach()
        assert z_in.grad is not None
        grads[path] = torch.cat([z_in.grad.flatten()] + [p.grad.flatten() for p in kernel.parameters() if p.grad is not None])
    assert torch.allclose(values["expanded"], values["reference"], atol=1e-12, rtol=0)
    assert torch.allclose(grads["expanded"], grads["reference"], atol=1e-10, rtol=0)


def test_expanded_path_float32_error_and_shift_invariance() -> None:
    torch.manual_seed(0)
    kernel = GreenKernel(32, n_anchors=64)
    z = torch.randn(512, 32) * 0.5
    with torch.no_grad():
        kernel.anchor_positions.copy_(z[:64] + 0.05 * torch.randn(64, 32))
        reference = kernel(z, path="reference")
        assert float((kernel(z, path="expanded") - reference).abs().max()) < 2e-4
        # A global offset is removed by the centring: the error does not grow with it.
        kernel.anchor_positions.add_(100.0)
        assert float((kernel(z + 100.0, path="expanded") - reference).abs().max()) < 1e-3
        assert float(kernel(z, path="expanded").max()) <= 1.0 + 1e-6  # the clamp: no value above the blend of ones


def test_auto_path_is_the_reference_on_cpu_and_chunking_follows_the_path() -> None:
    torch.manual_seed(0)
    kernel = GreenKernel(6, n_anchors=9)
    z = torch.randn(40, 6)
    with torch.no_grad():
        assert torch.equal(kernel(z), kernel(z, path="reference"))
        small = GreenKernel(6, n_anchors=9, chunk_elements=9 * 4 * 7)  # 7 rows per expanded chunk, 4 per reference chunk
        small.load_state_dict(kernel.state_dict())
        assert torch.allclose(small(z, path="expanded"), kernel(z, path="expanded"), atol=1e-6)
        assert torch.equal(small(z, path="reference"), kernel(z, path="reference"))
    assert kernel._all_expanded("expanded") and not kernel._all_expanded("fast")  # pyright: ignore[reportPrivateUsage]

    # Off CPU, "auto" takes the expanded path only from R·d = 4096 up (launch-bound below).
    class _OnDevice:
        device = torch.device("cuda")

    assert kernel._resolve_path(_OnDevice(), None) == "fast"  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage, reportArgumentType]
    assert GreenKernel(32, n_anchors=128)._resolve_path(_OnDevice(), None) == "expanded"  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage, reportArgumentType]
    mixed = GreenKernel(6, n_anchors=9, operator=("gaussian", "laplacian"))
    assert not mixed._all_expanded("expanded")  # pyright: ignore[reportPrivateUsage]


def test_kernel_path_is_validated_and_flows_from_the_config() -> None:
    with pytest.raises(IGLConfigError):
        GreenKernel(3, kernel_path="quick")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    with pytest.raises(IGLConfigError):
        igl.KernelConfig(kernel_path="quick")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    config = igl.IGLConfig(max_dim=4, kernel=igl.KernelConfig(kernel_path="expanded", per_scale_columns=True))
    module = igl.IGLModule(8, 4, 2, config=config)
    assert isinstance(module.green, GreenKernel) and module.green.kernel_path == "expanded"
    back = igl.IGLConfig.from_dict(config.to_dict())
    assert back.kernel.kernel_path == "expanded" and back.kernel.per_scale_columns is True


def test_quadratic_operators_must_be_separable() -> None:
    assert get_operator("gaussian").quadratic and not get_operator("laplacian").quadratic
    with pytest.raises(IGLConfigError):
        register_operator("not-separable-quadratic", get_operator("gaussian").fn, quadratic=True)


@pytest.mark.parametrize(("beta", "axis_normalize"), [(1.0, True), (0.0, True), (0.5, False)])
def test_all_prefix_penalty_equals_the_loop(beta: float, axis_normalize: bool) -> None:
    torch.manual_seed(0)
    jac = torch.randn(16, 12, 40, dtype=torch.float64)
    g = (jac @ jac.mT).requires_grad_(True)
    blocks = [conformality_loss(g[:, :k, :k], beta=beta, axis_normalize=axis_normalize) for k in range(2, 13)]
    loop = torch.stack(blocks).mean()
    closed = conformality_loss_all_prefixes(g, beta=beta, axis_normalize=axis_normalize)
    assert torch.allclose(closed, loop, atol=1e-13, rtol=0)
    (grad_loop,) = torch.autograd.grad(loop, g)  # pyright: ignore[reportUnknownMemberType]
    (grad_closed,) = torch.autograd.grad(closed, g)  # pyright: ignore[reportUnknownMemberType]
    assert torch.allclose(grad_closed, grad_loop, atol=1e-13, rtol=0)


def _rank_starved(n: int = 4096, r: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """A wide Φ of rank 19 with column norms over eight decades: the float32 Gram is not positive definite at ridge 1e-3."""
    torch.manual_seed(0)
    z = torch.randn(n, 19)
    phi = (z @ torch.randn(19, r)) * torch.logspace(-6, 1.8, r)
    phi = phi + 1e-7 * phi.abs().mean() * torch.randn_like(phi)
    return phi, torch.nn.functional.one_hot(z[:, :10].argmax(dim=1), 10).float()


@pytest.mark.parametrize("route", ["float64-gram", "float32-gram"])
def test_hybrid_solve_routes_agree_with_a_float64_least_squares(route: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if route == "float32-gram":
        monkeypatch.setattr(solver, "_F64_GRAM_MAX_FLOPS", 0)
    torch.manual_seed(0)
    phi = torch.rand(2048, 48)
    y = torch.randn(2048, 3)
    w, bad = solver.ridge_solve_hybrid(phi, y, l2=1e-3)
    p64, y64 = phi.double(), y.double()
    lam = 1e-3 * float(p64.norm(dim=0).mean()) ** 2
    exact = torch.linalg.solve(p64.T @ p64 + lam * torch.eye(48, dtype=torch.float64), p64.T @ y64)
    assert not bool(bad)
    tolerance = 1e-5 if route == "float64-gram" else 5e-3
    assert float((w.double() - exact).norm() / exact.norm()) < tolerance


def test_hybrid_solve_survives_a_gram_the_float32_factor_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    phi, y = _rank_starved()
    monkeypatch.setattr(solver, "_F64_GRAM_MAX_FLOPS", 0)  # force the float32 Gram first: the fallback must rescue it
    w, bad = solver.ridge_solve_hybrid(phi, y, l2=1e-3)
    assert not bool(bad)
    accuracy = float(((phi @ w).argmax(dim=1) == y.argmax(dim=1)).float().mean())
    assert accuracy > 0.9


def test_mps_backend_switches_to_the_hybrid_solve_at_the_column_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def device(phi: torch.Tensor, y: torch.Tensor, **_: object) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append("device")
        return torch.zeros(phi.shape[1], y.shape[1]), torch.tensor(False)

    def hybrid(phi: torch.Tensor, y: torch.Tensor, **_: object) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append("hybrid")
        return torch.zeros(phi.shape[1], y.shape[1]), torch.tensor(False)

    monkeypatch.setattr("igl.core._backend.ridge_solve_device", device)
    monkeypatch.setattr("igl.core._backend.ridge_solve_hybrid", hybrid)
    backend = MpsBackend(hybrid_min_columns=32)
    backend.ridge_solve(torch.zeros(8, 31), torch.zeros(8, 2), l2=1e-3)
    backend.ridge_solve(torch.zeros(8, 32), torch.zeros(8, 2), l2=1e-3)
    assert calls == ["device", "hybrid"]


def test_cpu_cholesky_solver_is_opt_in_and_close_to_the_reference() -> None:
    assert igl.MatryoshkaConfig().cpu_solver is igl.CpuSolver.LSTSQ
    reference = select_backend("cpu")
    assert isinstance(reference, CpuBackend) and not reference.cholesky
    fast = select_backend("cpu", cpu_cholesky=True)
    torch.manual_seed(0)
    phi, y = torch.rand(1024, 32), torch.randn(1024, 2)
    w_ref, _ = reference.ridge_solve(phi, y, l2=1e-3)
    w_fast, bad = fast.ridge_solve(phi, y, l2=1e-3)
    assert not bool(bad) and float((w_fast - w_ref).norm() / w_ref.norm()) < 1e-3
    current = torch.ones(32, 2)
    broken = phi.clone()
    broken[0, 0] = float("nan")
    assert torch.equal(fast.refresh_weights(broken, y, l2=1e-3, current=current), current)
    back = igl.IGLConfig.from_dict(igl.IGLConfig(max_dim=4, matryoshka=igl.MatryoshkaConfig(cpu_solver="cholesky")).to_dict())
    assert back.matryoshka.cpu_solver is igl.CpuSolver.CHOLESKY


def test_cpu_fast_settings_train_to_the_same_accuracy() -> None:
    torch.manual_seed(0)
    x = torch.randn(1200, 10)
    y = (x[:, 0] * x[:, 1] > 0).long()
    scores: dict[str, float] = {}
    for name, kernel_path, cpu_solver in (("reference", "auto", "lstsq"), ("fast", "expanded", "cholesky")):
        torch.manual_seed(1)
        config = igl.IGLConfig(
            max_dim=4,
            kernel=igl.KernelConfig(n_anchors=32, kernel_path=kernel_path),  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
            matryoshka=igl.MatryoshkaConfig(epochs=30, early_stop_patience=None, cpu_solver=cpu_solver),  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        )
        module = igl.IGLModule(10, 4, 2, config=config)
        trainer = igl.MatryoshkaTrainer(loss=igl.CrossEntropyLoss(n_classes=2), config=config.matryoshka)
        history = trainer.fit(module, x[:1000], y[:1000], x_val=x[1000:], y_val=y[1000:])
        scores[name] = history.val_metric[-1]
    assert scores["reference"] > 0.8
    assert abs(scores["fast"] - scores["reference"]) < 0.05
