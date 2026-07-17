"""Required dimensions of a tiny GPT-2's final layer, head in the loop.

Downloads ``sshleifer/tiny-gpt2`` (a few MB, randomly-initialized weights,
so the numbers are illustrative), extracts final-layer activations with
disjoint fit/eval splits, and sweeps a PCA projection arm against the
model's own verified head: for each k, states are projected to k
dimensions, reconstructed, and read by the frozen head; the report gives
the smallest k within each perplexity tolerance of the intact model.

Requires the ``[nlp]`` extra and network access for the model download.

Run with::

    python -m examples.nlp.requirement_tiny_gpt2
"""

import json

import torch

from examples._utils import make_run_dir, set_seed
from igl.nlp import extract_activations, requirement_dimension, resolve_head

EXAMPLE_NAME = "requirement_tiny_gpt2"
MODEL = "sshleifer/tiny-gpt2"


def main() -> None:
    from transformers import AutoModelForCausalLM

    set_seed(42)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    head = resolve_head(model)
    print(f"{MODEL}: head verified, post_norm={head.post_norm}")

    vocab = int(model.config.vocab_size)
    ids = torch.randint(0, vocab, (4096,), generator=torch.Generator().manual_seed(0))
    last_layer = int(model.config.num_hidden_layers) - 1
    acts = extract_activations(model, ids, layer=last_layer, ctx=128, n_eval_chunks=8, n_fit=2000)

    mu = acts.fit.mean(0)
    _, _, v = torch.linalg.svd(acts.fit - mu, full_matrices=False)

    def pca(states: torch.Tensor, k: int) -> torch.Tensor:
        basis = v[:k]
        return (states - mu) @ basis.T @ basis + mu

    width = acts.fit.shape[1]
    grid = sorted({k for k in (1, 2, 4, 8, width) if k <= width})
    report = requirement_dimension({"pca": pca}, head, acts.eval_batches, grid=grid)

    print(f"intact perplexity: {report.intact_perplexity:.2f}")
    for k, ppl in report.curves["pca"].items():
        print(f"  k={k:>3}: {ppl:.2f}")
    for tol, k_req in report.required["pca"].items():
        print(f"  within {tol:.2f}x: k={k_req}")

    run_dir = make_run_dir(EXAMPLE_NAME)
    payload = {
        "intact_perplexity": report.intact_perplexity,
        "curves": report.curves,
        "required": {arm: {str(t): k for t, k in tols.items()} for arm, tols in report.required.items()},
    }
    (run_dir / "requirement.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {run_dir / 'requirement.json'}")


if __name__ == "__main__":
    main()
