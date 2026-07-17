"""The inner-solver zoo for the VP benchmark suite.

Every solver minimizes the same Tikhonov least-squares objective as the
package's ``igl.direct_solve_weights``:

    w* = argmin_w ||phi @ w - y||^2 + l2_eff * ||w||^2,
    l2_eff = l2 * mean-column-norm(phi)^2,

so solutions are directly comparable across arms. Iterative solvers report
iteration counts and achieved residuals; all compute is fp32 on the tensors'
device (benchmarks pin to CPU by default, matching the package's solve).

The zoo exists to measure, not to ship: winners are upstreamed to the
package as a separate effort.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from igl import direct_solve_weights

__all__ = [
    "SolveResult",
    "adamw_to_tol",
    "block_cg",
    "cholesky_normal",
    "direct_lstsq",
    "effective_l2",
    "lbfgs_convex",
    "nystrom_pcg",
    "sgd_to_tol",
]


@dataclass(slots=True)
class SolveResult:
    """Solution plus solver diagnostics.

    Attributes:
        w: The solved weights ``[p, D]``.
        iterations: Iteration/epoch count (0 for direct methods).
        residual: Relative normal-equation residual ``||A w - b||_F / ||b||_F``.
        converged: Whether the iterative tolerance was reached (True for direct).
    """

    w: torch.Tensor
    iterations: int
    residual: float
    converged: bool


def effective_l2(phi: torch.Tensor, l2: float) -> float:
    """The package's relative ridge: ``l2 * mean-column-norm(phi)^2`` (solver.py semantics)."""
    col_scale = phi.norm(dim=0).mean().clamp_min(1e-6)
    return float(l2 * col_scale**2)


def _normal_residual(phi: torch.Tensor, y: torch.Tensor, w: torch.Tensor, lam: float) -> float:
    b = phi.T @ y
    r = phi.T @ (phi @ w) + lam * w - b
    return float(r.norm() / b.norm().clamp_min(1e-30))


def direct_lstsq(phi: torch.Tensor, y: torch.Tensor, *, l2: float = 1e-3) -> SolveResult:
    """Reference arm: the package's stacked-QR ``direct_solve_weights``."""
    w = direct_solve_weights(phi, y, l2=l2, on_nonfinite="raise")
    return SolveResult(w=w, iterations=0, residual=_normal_residual(phi, y, w, effective_l2(phi, l2)), converged=True)


def cholesky_normal(phi: torch.Tensor, y: torch.Tensor, *, l2: float = 1e-3) -> SolveResult:
    """Cholesky on the normal equations: forms the ``p x p`` Gram (the memory-cliff arm)."""
    lam = effective_l2(phi, l2)
    gram = phi.T @ phi
    gram.diagonal().add_(lam)
    factor = torch.linalg.cholesky(gram)
    w = torch.cholesky_solve(phi.T @ y, factor)
    return SolveResult(w=w, iterations=0, residual=_normal_residual(phi, y, w, lam), converged=True)


def block_cg(
    phi: torch.Tensor,
    y: torch.Tensor,
    *,
    l2: float = 1e-3,
    tol: float = 1e-6,
    max_iter: int | None = None,
    x0: torch.Tensor | None = None,
    matrix_free: bool = False,
    preconditioner: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> SolveResult:
    """CG on the normal equations, all right-hand sides advanced in lockstep.

    Each RHS column runs its own CG recursion (per-column alpha/beta), fully
    vectorized; converged columns are frozen. ``x0`` warm-starts (the
    outer-loop reuse pattern); ``matrix_free`` applies ``phi.T @ (phi @ v)``
    without forming the Gram matrix, so peak memory stays ``O(p * D)``.
    """
    lam = effective_l2(phi, l2)
    if matrix_free:

        def apply_a(v: torch.Tensor) -> torch.Tensor:
            return phi.T @ (phi @ v) + lam * v
    else:
        gram = phi.T @ phi
        gram.diagonal().add_(lam)

        def apply_a(v: torch.Tensor) -> torch.Tensor:
            return gram @ v

    b = phi.T @ y
    w = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - apply_a(w)
    z = r if preconditioner is None else preconditioner(r)
    p = z.clone()
    rz = (r * z).sum(dim=0)
    b_norm = b.norm(dim=0).clamp_min(1e-30)
    limit = max_iter if max_iter is not None else 20 * phi.shape[1]
    iterations = 0
    while iterations < limit:
        active = r.norm(dim=0) / b_norm > tol
        if not bool(active.any()):
            break
        iterations += 1
        ap = apply_a(p)
        alpha = rz / (p * ap).sum(dim=0).clamp_min(1e-30)
        alpha = torch.where(active, alpha, torch.zeros_like(alpha))
        w += alpha * p
        r -= alpha * ap
        z = r if preconditioner is None else preconditioner(r)
        rz_new = (r * z).sum(dim=0)
        beta = rz_new / rz.clamp_min(1e-30)
        rz = rz_new
        p = z + beta * p
    converged = bool((r.norm(dim=0) / b_norm <= tol).all())
    return SolveResult(w=w, iterations=iterations, residual=_normal_residual(phi, y, w, lam), converged=converged)


def nystrom_pcg(
    phi: torch.Tensor,
    y: torch.Tensor,
    *,
    l2: float = 1e-3,
    tol: float = 1e-6,
    sketch_rank: int | None = None,
    max_iter: int | None = None,
    seed: int = 0,
) -> SolveResult:
    """Nystrom-preconditioned CG (Frangella-Tropp-Udell style), matrix-free.

    Builds a randomized rank-``sketch_rank`` Nystrom approximation
    ``A ~ U diag(lam_i) U^T`` of ``A = phi^T phi + l2_eff I`` and
    preconditions CG with ``P^{-1} = U diag((lam_r + mu)/(lam_i + mu)) U^T
    + (I - U U^T)`` where ``mu = l2_eff`` and ``lam_r`` the smallest kept
    eigenvalue. Iteration counts become nearly independent of the condition
    number once the sketch captures the heavy spectrum.
    """
    p = phi.shape[1]
    rank = sketch_rank if sketch_rank is not None else max(min(p // 10, 800), 16)
    lam = effective_l2(phi, l2)
    generator = torch.Generator().manual_seed(seed)
    omega = torch.randn(p, rank, generator=generator, dtype=phi.dtype)
    omega, _ = torch.linalg.qr(omega)
    a_omega = phi.T @ (phi @ omega) + lam * omega
    nu = 1e-7 * a_omega.norm()
    a_omega = a_omega + nu * omega
    core = omega.T @ a_omega
    factor = torch.linalg.cholesky(0.5 * (core + core.T))
    half = torch.linalg.solve_triangular(factor, a_omega.T, upper=False).T
    u, s, _ = torch.linalg.svd(half, full_matrices=False)
    eigs = (s**2 - nu).clamp_min(0.0)
    mu = lam
    floor = eigs[-1] + mu

    def preconditioner(r: torch.Tensor) -> torch.Tensor:
        coeffs = floor / (eigs + mu)
        ur = u.T @ r
        return u @ (coeffs.unsqueeze(1) * ur) + (r - u @ ur)

    return block_cg(phi, y, l2=l2, tol=tol, max_iter=max_iter, matrix_free=True, preconditioner=preconditioner)


def _gd_step_size(phi: torch.Tensor, lam: float, *, power_iters: int = 30) -> float:
    """1/L step for GD on the normal equations, L estimated by power iteration."""
    v = torch.randn(phi.shape[1], 1, dtype=phi.dtype)
    v /= v.norm()
    l_max = 1.0
    for _ in range(power_iters):
        v = phi.T @ (phi @ v) + lam * v
        l_max = float(v.norm())
        v /= max(l_max, 1e-30)
    return 1.0 / l_max


def sgd_to_tol(
    phi: torch.Tensor,
    y: torch.Tensor,
    *,
    l2: float = 1e-3,
    tol: float = 1e-6,
    batch_size: int | None = None,
    max_epochs: int = 20_000,
    seed: int = 0,
) -> SolveResult:
    """(S)GD on the quadratic to a relative-gradient tolerance.

    ``batch_size=None`` runs full-batch gradient descent with a 1/L step
    (L from power iteration): the O(kappa)-epoch arm of the discussion's
    claim. A finite batch size runs **SVRG** (epoch-anchored variance
    reduction) rather than plain minibatch SGD: constant-step SGD stalls in
    its noise ball and cannot meet a tight tolerance at all, so SVRG is the
    strongest steelman of "stochastic inner solver to convergence" — if even
    SVRG loses to CG, the claim holds a fortiori.
    """
    lam = effective_l2(phi, l2)
    n = phi.shape[0]
    full_step = _gd_step_size(phi, lam)
    w = torch.zeros(phi.shape[1], y.shape[1], dtype=phi.dtype)
    b = phi.T @ y
    b_norm = float(b.norm().clamp_min(1e-30))
    generator = torch.Generator().manual_seed(seed)
    epochs = 0
    while epochs < max_epochs:
        epochs += 1
        if batch_size is None:
            grad = phi.T @ (phi @ w) + lam * w - b
            w -= full_step * grad
        else:
            anchor = w.clone()
            anchor_grad = phi.T @ (phi @ anchor) + lam * anchor - b
            step = full_step / 4.0
            perm = torch.randperm(n, generator=generator)
            for start in range(0, n, batch_size):
                rows = perm[start : start + batch_size]
                phi_b = phi[rows]
                scale = n / phi_b.shape[0]
                correction = scale * (phi_b.T @ (phi_b @ (w - anchor)))
                grad = correction + lam * (w - anchor) + anchor_grad
                w -= step * grad
        grad_norm = float((phi.T @ (phi @ w) + lam * w - b).norm()) / b_norm
        if grad_norm <= tol:
            break
    converged = float((phi.T @ (phi @ w) + lam * w - b).norm()) / b_norm <= tol
    return SolveResult(w=w, iterations=epochs, residual=_normal_residual(phi, y, w, lam), converged=converged)


def adamw_to_tol(
    phi: torch.Tensor,
    y: torch.Tensor,
    *,
    l2: float = 1e-3,
    tol: float = 1e-6,
    lr: float = 1e-2,
    max_epochs: int = 20_000,
) -> SolveResult:
    """Full-batch AdamW on the quadratic — the deep-learning-shelf arm.

    Exists to demonstrate the optimizer claim empirically: a blind adaptive
    optimizer cannot beat CG on a known constant Hessian.
    """
    lam = effective_l2(phi, l2)
    w = torch.zeros(phi.shape[1], y.shape[1], dtype=phi.dtype, requires_grad=True)
    optimizer = torch.optim.AdamW([w], lr=lr, weight_decay=0.0)
    b = phi.T @ y
    b_norm = float(b.norm().clamp_min(1e-30))
    epochs = 0
    while epochs < max_epochs:
        epochs += 1
        optimizer.zero_grad(set_to_none=True)
        loss = 0.5 * ((phi @ w - y) ** 2).sum() + 0.5 * lam * (w**2).sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            grad_norm = float((phi.T @ (phi @ w) + lam * w - b).norm())
        if grad_norm / b_norm <= tol:
            break
    w_final = w.detach()
    converged = float((phi.T @ (phi @ w_final) + lam * w_final - b).norm()) / b_norm <= tol
    return SolveResult(w=w_final, iterations=epochs, residual=_normal_residual(phi, y, w_final, lam), converged=converged)


def lbfgs_convex(
    objective: Callable[[torch.Tensor], torch.Tensor],
    w0: torch.Tensor,
    *,
    tol: float = 1e-8,
    max_iter: int = 500,
    history_size: int = 20,
) -> tuple[torch.Tensor, int]:
    """Minimize a smooth convex objective in ``w`` with torch L-BFGS.

    The generalized-VP inner solver (E7): the objective is the true
    cross-entropy through a frozen readout, convex in ``w``. Returns the
    solution and the number of closure evaluations.
    """
    w = w0.detach().clone().contiguous().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [w], max_iter=max_iter, history_size=history_size, tolerance_grad=tol, line_search_fn="strong_wolfe"
    )
    evals = 0

    def closure() -> torch.Tensor:
        nonlocal evals
        evals += 1
        optimizer.zero_grad(set_to_none=True)
        loss = objective(w)
        loss.backward()
        return loss

    optimizer.step(closure)  # pyright: ignore[reportArgumentType]
    return w.detach(), evals
