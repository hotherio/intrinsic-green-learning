"""E7 — generalized VP: a true cross-entropy inner problem.

Danskin's theorem does not need the inner problem to be quadratic — only
minimized. So "iterative inner solver to convergence + envelope outer
gradient" is precisely the recipe for a generalized VP whose inner loss is
the *actual* cross-entropy of the linear readout (convex in w), rather
than the one-hot least-squares surrogate the package's
``CrossEntropyLoss`` uses today.

Pre-registered prediction (P10, synthetic leg):

- P10a  At matched outer budgets, generalized VP's final validation
        cross-entropy is <= the one-hot LS arm's on both classification
        testbeds (it optimizes the true objective).
- P10b  The accuracy gap between the two arms is small (the surrogate is
        a second-order approximation): |acc_gen - acc_ls| <= 2 points.

The GPT-2 leg (CE through a frozen verified head vs the Fisher-whitened
quadratic) is deferred to a follow-up wave; this card establishes the
mechanism on synthetic testbeds first.

Run with::

    python -m benchmarks.vp.e7_generalized_vp [--smoke]
"""

import argparse
import json
import time
from typing import Any

import torch
from torch.nn.functional import cross_entropy

from benchmarks.vp._harness import machine_state, set_seed, write_result
from benchmarks.vp.data import moons_classification, split
from benchmarks.vp.solvers import effective_l2, lbfgs_convex
from igl import IGLModule, direct_solve_weights, normalize_phi

SEEDS = [42, 123, 456]
L2 = 1e-3


def four_blobs(n: int = 2000, *, seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    """Four gaussian blobs in 8-d with unequal class margins (CE-vs-LS separator)."""
    generator = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[2.0, 0.0], [-2.0, 0.0], [0.0, 1.2], [0.0, -1.2]])
    labels = torch.randint(0, 4, (n,), generator=generator)
    x_2d = centers[labels] + 0.6 * torch.randn(n, 2, generator=generator)
    lift = torch.randn(2, 8, generator=generator)
    x = x_2d @ lift + 0.05 * torch.randn(n, 8, generator=generator)
    return x, labels


def _design(module: IGLModule, x: torch.Tensor) -> torch.Tensor:
    z = module.encoder(x)
    return normalize_phi(module.green(z, gate_mask=None), module.normalize)


def train(
    module: IGLModule,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    n_classes: int,
    inner: str,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    """Fixed-k VP loop with either one-hot-LS or true-CE (L-BFGS) inner solves."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    params = list(module.encoder.parameters()) + list(module.green.parameters()) + [module.bias]
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    n = x_train.shape[0]
    onehot = torch.eye(n_classes)
    w_prev: torch.Tensor | None = None
    inner_evals: list[int] = []
    for _ in range(epochs):
        perm = torch.randperm(n, generator=generator)
        for i in range(0, n, 256):
            idx = perm[i : i + 256]
            with torch.no_grad():
                sub = torch.randperm(n, generator=generator)[:1024]
                phi_in = _design(module, x_train[sub])
                labels_in = y_train[sub]
                if inner == "onehot_ls":
                    target = onehot[labels_in] - module.bias.detach()
                    w = direct_solve_weights(phi_in, target, l2=L2)
                else:
                    lam = effective_l2(phi_in, L2)
                    bias = module.bias.detach()

                    def objective(
                        w_var: torch.Tensor,
                        phi_in: torch.Tensor = phi_in,
                        labels: torch.Tensor = labels_in,
                        lam: float = lam,
                        bias: torch.Tensor = bias,
                    ) -> torch.Tensor:
                        logits = phi_in @ w_var + bias
                        return cross_entropy(logits, labels) + 0.5 * lam * (w_var**2).sum() / phi_in.shape[0]

                    w0 = w_prev if w_prev is not None else torch.zeros(phi_in.shape[1], n_classes)
                    with torch.enable_grad():
                        w, evals = lbfgs_convex(objective, w0, tol=1e-8, max_iter=200)
                    w_prev = w
                    inner_evals.append(evals)
            optimizer.zero_grad()
            phi = _design(module, x_train[idx])
            logits = phi @ w.detach() + module.bias
            loss = cross_entropy(logits, y_train[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
    module.eval()
    with torch.no_grad():
        phi_in = _design(module, x_train)
        if inner == "onehot_ls":
            w = direct_solve_weights(phi_in, onehot[y_train] - module.bias.detach(), l2=L2)
        else:
            lam = effective_l2(phi_in, L2)
            bias = module.bias.detach()

            def final_objective(w_var: torch.Tensor) -> torch.Tensor:
                logits = phi_in @ w_var + bias
                return cross_entropy(logits, y_train) + 0.5 * lam * (w_var**2).sum() / phi_in.shape[0]

            with torch.enable_grad():
                w, _ = lbfgs_convex(
                    final_objective,
                    w_prev if w_prev is not None else torch.zeros(phi_in.shape[1], n_classes),
                    tol=1e-9,
                    max_iter=500,
                )
        logits_val = _design(module, x_val) @ w + module.bias
        val_ce = float(cross_entropy(logits_val, y_val).item())
        val_acc = float((logits_val.argmax(dim=1) == y_val).float().mean().item())
    return {
        "val_ce": val_ce,
        "val_acc": val_acc,
        "median_inner_evals": (sorted(inner_evals)[len(inner_evals) // 2] if inner_evals else 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    state = machine_state(gate=not args.smoke)
    start = time.time()
    epochs = 8 if args.smoke else 120
    n = 600 if args.smoke else 2000
    x_m, y_m = moons_classification(n)
    x_b, y_b = four_blobs(n)
    beds = {
        "moons": (split(x_m, y_m), 2),
        "four_blobs": (split(x_b, y_b), 4),
    }
    results: dict[str, Any] = {}
    p10a, gaps = True, []
    for name, ((x_train, y_train, x_val, y_val), n_classes) in beds.items():
        per_seed: list[dict[str, Any]] = []
        for seed in SEEDS:
            row: dict[str, Any] = {"seed": seed}
            for inner in ("onehot_ls", "true_ce"):
                set_seed(seed)
                module = IGLModule(input_dim=x_train.shape[1], max_dim=8, output_dim=n_classes, n_anchors=48, n_scales=4)
                row[inner] = train(
                    module, x_train, y_train, x_val, y_val, n_classes=n_classes, inner=inner, epochs=epochs, seed=seed
                )
            row["ce_improvement"] = row["onehot_ls"]["val_ce"] - row["true_ce"]["val_ce"]
            row["acc_gap_points"] = 100.0 * abs(row["true_ce"]["val_acc"] - row["onehot_ls"]["val_acc"])
            p10a = p10a and bool(row["true_ce"]["val_ce"] <= row["onehot_ls"]["val_ce"] * 1.001)
            gaps.append(row["acc_gap_points"])
            per_seed.append(row)
            print(
                f"{name:11s} seed={seed} CE ls={row['onehot_ls']['val_ce']:.4f} gen={row['true_ce']['val_ce']:.4f} "
                f"acc ls={row['onehot_ls']['val_acc']:.3f} gen={row['true_ce']['val_acc']:.3f}",
                flush=True,
            )
        results[name] = per_seed
    payload = {
        "config": {"seeds": SEEDS, "epochs": epochs, "smoke": args.smoke},
        "beds": results,
        "verdicts": {
            "P10a_generalized_ce_never_worse": p10a,
            "P10b_max_acc_gap_points": max(gaps) if gaps else None,
            "P10b_pass": bool(gaps) and max(gaps) <= 2.0,
        },
    }
    path = write_result("e7_generalized_vp", payload, state=state, wall_clock_s=time.time() - start)
    print(f"\nwrote {path}")
    print(json.dumps(payload["verdicts"], indent=2))


if __name__ == "__main__":
    main()
