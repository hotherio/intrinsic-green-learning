"""Whitened distillation — the metric decides what a tight budget captures.

A frozen random "teacher head" maps 24-D states to logits over 48 classes.
We compress the states through a Matryoshka bottleneck twice, with the same
module and budget, changing only the training target:

- plain :class:`igl.MSELoss` — reconstruction; capacity follows variance;
- :class:`igl.whitening.WhitenedMSELoss` with the head's Fisher pullback —
  plain least squares becomes the second-order expansion of the downstream
  KL, so capacity follows what the head actually reads.

The read-out compares, per bottleneck width, the KL between the teacher's
distribution on true and reconstructed states. The Fisher-whitened chart
should dominate at small ``k``: the variance-heavy directions it drops are
exactly the ones the softmax read-out prices at zero.

Run with::

    python -m examples.synthetic.whitened_regression
"""

import json

import torch
import torch.nn.functional as F  # noqa: N812

from examples._utils import make_run_dir, set_seed
from igl import IGLDistiller, IGLModule, MatryoshkaConfig, MatryoshkaTrainer, MSELoss
from igl.whitening import TargetWhitener, WhitenedMSELoss, fisher_pullback

EXAMPLE_NAME = "whitened_regression"


def _downstream_kl(w: torch.Tensor, y_true: torch.Tensor, y_hat: torch.Tensor) -> float:
    p = F.softmax(y_true @ w.T, dim=-1)
    log_q = F.log_softmax(y_hat @ w.T, dim=-1)
    return float(F.kl_div(log_q, p, reduction="batchmean"))


def main() -> None:
    set_seed(42)
    n_samples, state_dim, vocab = 2000, 24, 48
    max_dim = 12

    # States with anisotropic variance: the top variance directions carry
    # little of what the head reads (the paper's dissociation, in miniature).
    scales = torch.linspace(3.0, 0.1, state_dim)
    states = torch.randn(n_samples, state_dim) * scales
    w = torch.randn(vocab, state_dim) / state_dim**0.5
    w[:, :4] *= 0.05  # the head barely reads the four highest-variance directions

    config = MatryoshkaConfig(epochs=200, batch_size=128, early_stop_patience=None)
    results: dict[str, dict[int, float]] = {}

    for name in ("plain_mse", "fisher_whitened"):
        set_seed(42)
        module = IGLModule(input_dim=state_dim, max_dim=max_dim, output_dim=state_dim, n_anchors=32, n_scales=3)
        if name == "plain_mse":
            loss = MSELoss()
            unwhiten = None
        else:
            whitener = TargetWhitener(fisher_pullback(w, states, n_sub=n_samples)).fit(states)
            loss = WhitenedMSELoss(whitener)
            unwhiten = whitener.inverse_transform
        MatryoshkaTrainer(loss=loss, config=config).fit(module, states, states)

        curve: dict[int, float] = {}
        with torch.no_grad():
            for k in (2, 4, 8, max_dim):
                mask = torch.zeros(max_dim)
                mask[:k] = 1.0
                y_hat = module(states, gate_mask=mask)
                if unwhiten is not None:
                    y_hat = unwhiten(y_hat)
                curve[k] = _downstream_kl(w, states, y_hat)
        results[name] = curve
        print(f"{name}: " + "  ".join(f"k={k}: KL={v:.4f}" for k, v in curve.items()))

    # The same pipeline as a three-line estimator: metric in, charts out.
    set_seed(42)
    distiller = IGLDistiller(
        max_dim=max_dim,
        metric=fisher_pullback(w, states, n_sub=n_samples),
        config=None,
        random_state=42,
    )
    distiller.fit(states.numpy())
    kl_estimator = _downstream_kl(w, states, torch.from_numpy(distiller.reconstruct(states.numpy(), k=4)))
    print(f"IGLDistiller (k=4): KL={kl_estimator:.4f}  effective_dimension_={distiller.effective_dimension_}")

    run_dir = make_run_dir(EXAMPLE_NAME)
    (run_dir / "downstream_kl.json").write_text(json.dumps(results, indent=2))
    print(f"results written to {run_dir}")

    tight = min(results["plain_mse"])
    if results["fisher_whitened"][tight] < results["plain_mse"][tight]:
        print(f"at k={tight}, the Fisher-whitened chart preserves the head's read-out better — the metric is the lever")


if __name__ == "__main__":
    main()
