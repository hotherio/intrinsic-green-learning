"""Component micro-benchmarks: kernel formulations, ridge solvers, Jacobians.

Every alternative is timed (median of repeats, device-synced) and checked
against a reference so speed never comes without an accuracy number.

Run with::

    IGL_BENCH_DEVICE=mps python -m benchmarks.device.components
"""

from __future__ import annotations

import argparse
import time
import warnings
from typing import Any

import torch

import igl
from benchmarks.device._harness import machine_state, resolve_device, time_median, write_result
from igl.core.solver import direct_solve_weights


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30))


def kernel_bench(device: torch.device, *, n: int = 4096, r: int = 64, k: int = 4) -> list[dict[str, Any]]:
    """Current log-space Gaussian kernel vs the separable einsum formulation, forward and forward+backward."""
    rows: list[dict[str, Any]] = []
    for d in (8, 16):
        torch.manual_seed(0)
        green = igl.GreenKernel(latent_dim=d, n_anchors=r, n_scales=k).to(device)
        z = torch.randn(n, d, device=device)
        mu, log_sigma, log_gamma, importance = green.anchor_positions, green.log_sigma, green.log_gamma, green.rank_importance

        def einsum_path(
            zz: torch.Tensor = z,
            *,
            log_sigma: torch.Tensor = log_sigma,
            mu: torch.Tensor = mu,
            log_gamma: torch.Tensor = log_gamma,
            importance: torch.Tensor = importance,
        ) -> torch.Tensor:
            weights = 1.0 / (2 * torch.exp(log_sigma) ** 2 + 1e-8)
            d2 = (zz[:, None, :] - mu[None]) ** 2
            logk = -torch.einsum("nrd,kd->nrk", d2, weights)
            return (torch.exp(logk) * torch.softmax(log_gamma, 0)).sum(-1) * torch.sigmoid(importance)[None]

        def current(green: igl.GreenKernel = green, z: torch.Tensor = z) -> torch.Tensor:
            return green(z)

        zg = z.clone().requires_grad_(True)

        def current_bwd(green: igl.GreenKernel = green, zg: torch.Tensor = zg) -> None:
            green(zg).sum().backward()

        def einsum_bwd(zg: torch.Tensor = zg) -> None:
            einsum_path(zg).sum().backward()

        with torch.no_grad():
            reference = current()
            err = _rel_err(einsum_path(), reference)
            t_cur = time_median(current, device=device, repeats=20, warmup=3).median_s
            t_ein = time_median(einsum_path, device=device, repeats=20, warmup=3).median_s
        tb_cur = time_median(current_bwd, device=device, repeats=10, warmup=2).median_s
        tb_ein = time_median(einsum_bwd, device=device, repeats=10, warmup=2).median_s
        rows.append(
            {
                "d": d,
                "fwd_ms_current": t_cur * 1e3,
                "fwd_ms_einsum": t_ein * 1e3,
                "fwdbwd_ms_current": tb_cur * 1e3,
                "fwdbwd_ms_einsum": tb_ein * 1e3,
                "max_rel_err_einsum": err,
            }
        )
    return rows


def solver_bench(device: torch.device) -> list[dict[str, Any]]:
    """Package solve (lstsq on the stacked system) vs on-device Cholesky with one refinement step."""
    rows: list[dict[str, Any]] = []
    for n, r, c in ((4096, 64, 2), (4096, 64, 512), (4096, 256, 2000)):
        torch.manual_seed(0)
        module = igl.IGLModule(input_dim=16, max_dim=8, output_dim=c, n_anchors=r, n_scales=4)
        with torch.no_grad():
            phi_cpu = module.design_matrix(torch.randn(n, 16))
        y_cpu = torch.randn(n, c)
        l2 = 1e-3
        p64, t64 = phi_cpu.double(), y_cpu.double()
        lam64 = l2 * p64.norm(dim=0).mean() ** 2
        w_ref = torch.linalg.solve(p64.T @ p64 + lam64 * torch.eye(r, dtype=torch.float64), p64.T @ t64)
        pred_ref = p64 @ w_ref
        phi, y = phi_cpu.to(device), y_cpu.to(device)
        lam = (l2 * phi.norm(dim=0).mean() ** 2).detach()
        eye = torch.eye(r, device=device)

        def package(phi: torch.Tensor = phi, y: torch.Tensor = y, l2: float = l2) -> torch.Tensor:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return direct_solve_weights(phi, y, l2=l2)

        def cholesky_refined(
            phi: torch.Tensor = phi, y: torch.Tensor = y, lam: torch.Tensor = lam, eye: torch.Tensor = eye
        ) -> torch.Tensor:
            gram = phi.T @ phi + lam * eye
            rhs = phi.T @ y
            chol, _info = torch.linalg.cholesky_ex(gram)
            w = torch.cholesky_solve(rhs, chol)
            residual = rhs - (phi.T @ (phi @ w) + lam * w)
            return w + torch.cholesky_solve(residual, chol)

        methods: dict[str, Any] = {"package_lstsq": package, "cholesky_refined": cholesky_refined}
        if device.type == "cuda":

            def gels(
                phi: torch.Tensor = phi,
                y: torch.Tensor = y,
                lam: torch.Tensor = lam,
                eye: torch.Tensor = eye,
                r: int = r,
                c: int = c,
            ) -> torch.Tensor:
                aug = torch.cat([phi, lam.sqrt() * eye])
                yaug = torch.cat([y, torch.zeros(r, c, device=device)])
                return torch.linalg.lstsq(aug, yaug, driver="gels").solution

            methods["lstsq_gels"] = gels
        for name, fn in methods.items():
            try:
                timing = time_median(fn, device=device, repeats=10, warmup=2)
                w = timing.result.to("cpu").double()
                err = float(((p64 @ w - pred_ref).abs().max() / pred_ref.abs().max()).item())
                rows.append({"n": n, "r": r, "c": c, "method": name, "ms": timing.median_s * 1e3, "max_rel_pred_err": err})
            except Exception as exc:  # noqa: BLE001  # a benchmark reports failures, never hides them
                rows.append({"n": n, "r": r, "c": c, "method": name, "ms": None, "error": str(exc).splitlines()[0][:120]})
    return rows


def jacobian_bench(device: torch.device) -> list[dict[str, Any]]:
    """Per-dimension autograd loop vs vmap(jacrev) for the orthogonality penalty."""
    from torch.func import functional_call, jacrev, vmap

    from igl.spd.orthogonality import jacobian, orthogonality_loss, pullback_metric

    rows: list[dict[str, Any]] = []
    for b, d_in, d_out in ((256, 10, 4), (256, 2080, 16)):
        torch.manual_seed(0)
        encoder = igl.MLPEncoder(d_in, d_out, hidden=256, depth=2).to(device)
        x = torch.randn(b, d_in, device=device)

        def loop(encoder: igl.MLPEncoder = encoder, x: torch.Tensor = x, d_out: int = d_out) -> torch.Tensor:
            return orthogonality_loss(pullback_metric(jacobian(encoder, x, output_dim=d_out)))

        def vmapped(encoder: igl.MLPEncoder = encoder, x: torch.Tensor = x) -> torch.Tensor:
            params = dict(encoder.named_parameters())

            def single(xi: torch.Tensor) -> torch.Tensor:
                return functional_call(encoder, params, (xi.unsqueeze(0),)).squeeze(0)

            return orthogonality_loss(pullback_metric(vmap(jacrev(single))(x)))

        def loop_bwd() -> None:
            loop().backward()

        def vmap_bwd() -> None:
            vmapped().backward()

        err = float((loop() - vmapped()).abs().item())
        t_loop = time_median(loop_bwd, device=device, repeats=5, warmup=1).median_s
        t_vmap = time_median(vmap_bwd, device=device, repeats=5, warmup=1).median_s
        rows.append(
            {
                "batch": b,
                "input_dim": d_in,
                "latent_dim": d_out,
                "fwdbwd_ms_loop": t_loop * 1e3,
                "fwdbwd_ms_vmap": t_vmap * 1e3,
                "abs_loss_diff": err,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-gate", action="store_true")
    args = parser.parse_args()
    device = resolve_device(args.device)
    state = machine_state(device, gate=not args.no_gate)
    start = time.perf_counter()
    payload = {"kernel": kernel_bench(device), "solver": solver_bench(device), "jacobian": jacobian_bench(device)}
    for section, rows in payload.items():
        print(f"== {section} ({device.type}) ==")
        for row in rows:
            print("  " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    path = write_result("components", payload, device=device, state=state, wall_clock_s=time.perf_counter() - start)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
