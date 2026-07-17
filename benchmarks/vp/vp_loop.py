"""Instrumented variable-projection trainer for the benchmark suite.

A compact reimplementation of the package trainer's epoch loop (permutation
slicing, uniform truncation sampling, fresh inner sub-batch, bias-subtracted
targets, detached ``w`` in the graph ``phi @ w + bias``, AdamW outer, grad
clip — mirroring ``igl/core/trainer.py``) with the three extension points
the package trainer deliberately lacks:

- ``inner_solver``: any zoo member (direct, CG at a tolerance, SVRG, ...),
  with per-``k`` warm starting across outer steps;
- ``outer_mode``: minibatch AdamW (the house recipe), full-batch AdamW,
  full-batch L-BFGS on the reduced functional, or minibatch AdamW with
  epoch-level safeguarded Anderson/RNA extrapolation;
- ``probe_solvers``: per outer step, the encoder gradient is also computed
  under each probe solver at the *same* parameters and batch, and the
  relative error against the direct-solve gradient is recorded — the
  teacher-forced envelope-bias measurement of E1.

This module measures; it does not ship. The package trainer stays the
canonical implementation.
"""

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from igl import IGLModule, direct_solve_weights, normalize_phi

InnerSolver = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], tuple[torch.Tensor, int]]
"""``(phi, target, x0) -> (w, iterations)`` — x0 is a warm start or None."""

OuterMode = Literal[
    "minibatch-adam",
    "fullbatch-adam",
    "fullbatch-lbfgs",
    "minibatch-adam-aa",
    "fullbatch-adam-aa",
    "hybrid-adam-lbfgs",
]


def direct_inner(
    phi: torch.Tensor, target: torch.Tensor, x0: torch.Tensor | None = None, *, l2: float
) -> tuple[torch.Tensor, int]:
    """The package's exact inner solve as an :data:`InnerSolver`."""
    del x0
    return direct_solve_weights(phi, target, l2=l2), 0


@dataclass(slots=True)
class VPLoopConfig:
    epochs: int = 300
    batch_size: int = 256
    inner_batch_size: int = 4096
    encoder_lr: float = 1e-3
    source_l2: float = 1e-3
    grad_clip: float = 1.0
    fixed_k: int | None = None  # None = uniform 1..max_dim, matching the house sampler
    aa_window: int = 5
    aa_reg: float = 1e-8
    aa_ema: float | None = None  # EMA factor for the extrapolated iterates (RNA on averaged iterates)
    hybrid_warmup_epochs: int = 100  # minibatch-Adam epochs before the L-BFGS polish phase
    lbfgs_max_iter: int = 10
    seed: int = 0


@dataclass(slots=True)
class VPLoopResult:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    inner_iterations: list[int] = field(default_factory=list)
    grad_probe_errors: dict[str, list[float]] = field(default_factory=dict)
    grad_probe_cosines: dict[str, list[float]] = field(default_factory=dict)
    grad_probe_w_errors: dict[str, list[float]] = field(default_factory=dict)
    aa_proposals: int = 0
    aa_accepted: int = 0
    epoch_time_s: list[float] = field(default_factory=list)  # cumulative wall-clock at each epoch end
    wall_clock_s: float = 0.0


def _flat(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


class _WarmStarts:
    """Last solved w per truncation k, reused as the next warm start."""

    def __init__(self) -> None:
        self._cache: dict[int, torch.Tensor] = {}

    def get(self, k: int) -> torch.Tensor | None:
        return self._cache.get(k)

    def put(self, k: int, w: torch.Tensor) -> None:
        self._cache[k] = w.detach().clone()


class VPLoop:
    """Train an :class:`igl.IGLModule` with pluggable inner/outer machinery.

    Args:
        loss: A package loss strategy (``MSELoss``, ``CrossEntropyLoss``,
            ``WhitenedMSELoss``); its ``target()`` flows through the inner
            solve exactly as in the package trainer.
        config: Loop hyperparameters.
        inner_solver: The inner solver used for the *actual* update path;
            ``None`` means the package's direct solve.
        outer_mode: Outer optimizer variant.
        probe_solvers: Named solvers whose envelope gradient is compared to
            the direct-solve gradient at identical parameters each step.
    """

    def __init__(
        self,
        *,
        loss: Any,
        config: VPLoopConfig,
        inner_solver: InnerSolver | None = None,
        outer_mode: OuterMode = "minibatch-adam",
        probe_solvers: dict[str, InnerSolver] | None = None,
    ) -> None:
        self.loss = loss
        self.config = config
        self.inner_solver = inner_solver
        self.outer_mode = outer_mode
        self.probe_solvers = probe_solvers or {}
        self._lbfgs_start = 0.0

    # -- inner solve -----------------------------------------------------

    def _solve(self, phi: torch.Tensor, target: torch.Tensor, warm: torch.Tensor | None) -> tuple[torch.Tensor, int]:
        if self.inner_solver is None:
            return direct_solve_weights(phi, target, l2=self.config.source_l2), 0
        return self.inner_solver(phi, target, warm)

    def _design(self, module: IGLModule, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = module.encoder(x) * mask.unsqueeze(0)
        return normalize_phi(module.green(z, gate_mask=mask), module.normalize)

    def _inner_batch(
        self, module: IGLModule, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = x.shape[0]
        lstsq_n = min(self.config.inner_batch_size, n)
        idx = torch.randperm(n, generator=generator)[:lstsq_n]
        with torch.no_grad():
            phi = self._design(module, x[idx], mask)
            target = self.loss.target(y[idx]) - module.bias.detach()
        return phi, target

    # -- evaluation ------------------------------------------------------

    def _refresh_and_validate(
        self,
        module: IGLModule,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        generator: torch.Generator,
    ) -> float:
        module.eval()
        mask = torch.ones(module.encoder(x_train[:1]).shape[1])
        phi, target = self._inner_batch(module, x_train, y_train, mask, generator)
        with torch.no_grad():
            w, _ = self._solve(phi, target, None)
            if torch.isfinite(w).all():
                module.set_source_weights(w)
            phi_val = self._design(module, x_val, mask)
            output = phi_val @ module.source_weights + module.bias
            val = float(self.loss.loss(output, self.loss.target(y_val)).item())
        module.train()
        return val

    # -- the loop --------------------------------------------------------

    def fit(  # noqa: PLR0915 (the epoch loop reads as one unit; splitting obscures it)
        self,
        module: IGLModule,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> VPLoopResult:
        config = self.config
        torch.manual_seed(config.seed)
        generator = torch.Generator().manual_seed(config.seed)
        result = VPLoopResult()
        for name in self.probe_solvers:
            result.grad_probe_errors[name] = []
            result.grad_probe_cosines[name] = []
            result.grad_probe_w_errors[name] = []
        params = list(module.encoder.parameters()) + list(module.green.parameters()) + [module.bias]
        d_max = int(module.encoder(x_train[:1]).shape[1])
        warm = _WarmStarts()
        n = x_train.shape[0]
        start = time.perf_counter()

        if self.outer_mode == "fullbatch-lbfgs":
            self._fit_lbfgs(module, params, x_train, y_train, x_val, y_val, d_max, generator, result, start=start)
            result.wall_clock_s = time.perf_counter() - start
            return result

        optimizer = torch.optim.AdamW(params, lr=config.encoder_lr)
        batch_size = n if self.outer_mode.startswith("fullbatch") else config.batch_size
        # Hybrid explore-then-polish: minibatch Adam picks the basin, L-BFGS
        # finishes the smooth endgame (basin selection is what stochasticity
        # buys; convergence rate is what curvature buys).
        hybrid = self.outer_mode == "hybrid-adam-lbfgs"
        adam_epochs = min(config.hybrid_warmup_epochs, config.epochs) if hybrid else config.epochs
        aa_active = self.outer_mode.endswith("-aa")
        aa_history: deque[torch.Tensor] = deque(maxlen=config.aa_window + 1)
        aa_ema_state: torch.Tensor | None = None
        best_val = float("inf")

        for _epoch in range(adam_epochs):
            perm = torch.randperm(n, generator=generator)
            epoch_loss, n_batches = 0.0, 0
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                x_batch, y_batch = x_train[idx], y_train[idx]
                k = config.fixed_k or int(torch.randint(1, d_max + 1, (1,), generator=generator).item())
                mask = torch.zeros(d_max)
                mask[:k] = 1.0
                phi_grad = self._design(module, x_batch, mask)
                phi_in, target_in = self._inner_batch(module, x_train, y_train, mask, generator)
                w, iters = self._solve(phi_in, target_in, warm.get(k))
                warm.put(k, w)
                result.inner_iterations.append(iters)
                self._probe_gradients(module, params, phi_grad, phi_in, target_in, y_batch, w, result)
                optimizer.zero_grad()
                output = phi_grad @ w.detach() + module.bias
                loss = self.loss.loss(output, self.loss.target(y_batch))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            result.train_loss.append(epoch_loss / max(n_batches, 1))
            val = self._refresh_and_validate(module, x_train, y_train, x_val, y_val, generator)
            result.val_loss.append(val)
            result.epoch_time_s.append(time.perf_counter() - start)
            best_val = min(best_val, val)

            if aa_active:
                iterate = _flat([p.detach().clone() for p in params])
                if config.aa_ema is not None:
                    aa_ema_state = (
                        iterate if aa_ema_state is None else config.aa_ema * aa_ema_state + (1 - config.aa_ema) * iterate
                    )
                    aa_history.append(aa_ema_state.clone())
                else:
                    aa_history.append(iterate)
                proposal = self._rna_proposal(aa_history)
                if proposal is not None:
                    result.aa_proposals += 1
                    backup = _flat([p.detach().clone() for p in params])
                    self._load_flat(params, proposal)
                    val_prop = self._refresh_and_validate(module, x_train, y_train, x_val, y_val, generator)
                    if val_prop < best_val:
                        result.aa_accepted += 1
                        result.val_loss[-1] = val_prop
                        best_val = val_prop
                        aa_history.clear()
                        aa_ema_state = _flat([p.detach().clone() for p in params])
                        aa_history.append(aa_ema_state.clone())
                    else:
                        self._load_flat(params, backup)
        if self.outer_mode == "hybrid-adam-lbfgs" and config.epochs > adam_epochs:
            polish = VPLoopConfig(**{f.name: getattr(config, f.name) for f in config.__dataclass_fields__.values()})  # type: ignore[arg-type]
            polish.epochs = config.epochs - adam_epochs
            self.config = polish
            self._fit_lbfgs(module, params, x_train, y_train, x_val, y_val, d_max, generator, result, start=start)
            self.config = config
        result.wall_clock_s = time.perf_counter() - start
        return result

    def _fit_lbfgs(
        self,
        module: IGLModule,
        params: list[torch.Tensor],
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        d_max: int,
        generator: torch.Generator,
        result: VPLoopResult,
        *,
        start: float | None = None,
    ) -> None:
        """Full-batch L-BFGS on the reduced functional (direct inner solve in the closure)."""
        config = self.config
        self._lbfgs_start = time.perf_counter() if start is None else start
        optimizer = torch.optim.LBFGS(
            params, lr=1.0, max_iter=config.lbfgs_max_iter, history_size=10, line_search_fn="strong_wolfe"
        )
        mask = torch.ones(d_max)

        for _ in range(config.epochs):

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                phi_in, target_in = self._inner_batch(module, x_train, y_train, mask, generator)
                w, _ = self._solve(phi_in, target_in, None)
                phi = self._design(module, x_train, mask)
                output = phi @ w.detach() + module.bias
                loss = self.loss.loss(output, self.loss.target(y_train))
                loss.backward()
                return loss

            loss = optimizer.step(closure)  # pyright: ignore[reportArgumentType]
            result.train_loss.append(float(loss.item()))  # pyright: ignore[reportAttributeAccessIssue]
            result.val_loss.append(self._refresh_and_validate(module, x_train, y_train, x_val, y_val, generator))
            result.epoch_time_s.append(time.perf_counter() - self._lbfgs_start)

    def _probe_gradients(
        self,
        module: IGLModule,
        params: list[torch.Tensor],
        phi_grad: torch.Tensor,
        phi_in: torch.Tensor,
        target_in: torch.Tensor,
        y_batch: torch.Tensor,
        w_reference_source: torch.Tensor,
        result: VPLoopResult,
    ) -> None:
        """Teacher-forced envelope-bias measurement (E1).

        At the current parameters, compute the encoder gradient with the
        exact inner solve and with each probe solver on the *same* inner
        batch; record relative L2 error and cosine similarity.
        """
        if not self.probe_solvers:
            return
        target_batch = self.loss.target(y_batch)

        def encoder_grad(w: torch.Tensor) -> torch.Tensor:
            output = phi_grad @ w.detach() + module.bias
            loss = self.loss.loss(output, target_batch)
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            return _flat([g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params, strict=True)])

        w_exact = direct_solve_weights(phi_in, target_in, l2=self.config.source_l2)
        reference = encoder_grad(w_exact)
        w_exact_norm = float(w_exact.norm().clamp_min(1e-30))
        for name, solver in self.probe_solvers.items():
            w_probe, _ = solver(phi_in, target_in, None)
            result.grad_probe_w_errors[name].append(float((w_probe - w_exact).norm()) / w_exact_norm)
            grad = encoder_grad(w_probe)
            denom = float(reference.norm().clamp_min(1e-30))
            result.grad_probe_errors[name].append(float((grad - reference).norm()) / denom)
            cosine = float(torch.dot(grad, reference) / (grad.norm().clamp_min(1e-30) * reference.norm().clamp_min(1e-30)))
            result.grad_probe_cosines[name].append(cosine)

    # -- Anderson / RNA --------------------------------------------------

    def _rna_proposal(self, history: deque[torch.Tensor]) -> torch.Tensor | None:
        """Regularized nonlinear acceleration over the epoch-end iterates."""
        if len(history) < 3:
            return None
        thetas = list(history)
        residuals = torch.stack([b - a for a, b in zip(thetas[:-1], thetas[1:], strict=True)])  # [m, P]
        m = residuals.shape[0]
        gram = residuals @ residuals.T
        gram = gram / gram.norm().clamp_min(1e-30)
        rhs = torch.ones(m, dtype=gram.dtype)
        try:
            coeffs = torch.linalg.solve(gram + self.config.aa_reg * torch.eye(m), rhs)
        except RuntimeError:
            return None
        coeffs = coeffs / coeffs.sum().clamp_min(1e-30)
        return torch.stack(thetas[1:]).T @ coeffs

    @staticmethod
    def _load_flat(params: list[torch.Tensor], flat: torch.Tensor) -> None:
        offset = 0
        with torch.no_grad():
            for p in params:
                count = p.numel()
                p.copy_(flat[offset : offset + count].view_as(p))
                offset += count
