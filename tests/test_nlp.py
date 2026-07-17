"""Tests for :mod:`igl.nlp` against a tiny HuggingFace model.

Skipped when the ``[nlp]`` extra is not installed or the tiny model cannot
be downloaded (offline CI).
"""

import pytest
import torch

from igl import IGLConfigError

transformers = pytest.importorskip("transformers")

_TINY = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def model() -> object:
    try:
        return transformers.AutoModelForCausalLM.from_pretrained(_TINY, dtype=torch.float32).eval()
    except OSError as error:  # offline or hub outage
        pytest.skip(f"cannot download {_TINY}: {error}")


@pytest.fixture(scope="module")
def token_stream(model: object) -> torch.Tensor:
    vocab = int(model.config.vocab_size)  # pyright: ignore[reportAttributeAccessIssue]
    generator = torch.Generator().manual_seed(0)
    return torch.randint(0, vocab, (600,), generator=generator)


def test_resolve_head_matches_model_logits(model: object) -> None:
    from igl.nlp import resolve_head

    head = resolve_head(model)
    input_ids = torch.arange(1, 12).unsqueeze(0)
    with torch.no_grad():
        output = model(input_ids, output_hidden_states=True)  # pyright: ignore[reportCallIssue]
        assert torch.allclose(head(output.hidden_states[-1]), output.logits.float(), atol=1e-2)


def test_resolve_head_rejects_headless_object() -> None:
    from igl.nlp import resolve_head

    with pytest.raises(IGLConfigError, match="no unembedding"):
        resolve_head(torch.nn.Linear(4, 4))


def test_continue_from_passes_identity_gate_everywhere(model: object) -> None:
    from igl.nlp import ContinueFrom

    n_blocks = ContinueFrom(model, 0).n_blocks
    for layer in range(n_blocks):
        closure = ContinueFrom(model, layer)
        assert closure.layer == layer


def test_continue_from_rejects_out_of_range_layer(model: object) -> None:
    from igl.nlp import ContinueFrom

    with pytest.raises(IGLConfigError, match="out of range"):
        ContinueFrom(model, 99)


def test_extract_activations_splits_disjointly(model: object, token_stream: torch.Tensor) -> None:
    from igl.nlp import extract_activations

    acts = extract_activations(model, token_stream, layer=0, ctx=64, n_eval_chunks=3, n_fit=100)
    assert len(acts.eval_batches) == 3
    states, targets = acts.eval_batches[0]
    assert states.shape == (63, states.shape[1])
    assert targets.shape == (63,)
    assert torch.equal(targets, token_stream[1:64])
    assert acts.fit.shape == (100, states.shape[1])


def test_extract_activations_needs_enough_tokens(model: object) -> None:
    from igl.nlp import extract_activations

    with pytest.raises(IGLConfigError, match="disjoint"):
        extract_activations(model, torch.zeros(100, dtype=torch.long), layer=0, ctx=64, n_eval_chunks=3)


def test_requirement_dimension_full_rank_arm_matches_intact(model: object, token_stream: torch.Tensor) -> None:
    from igl.nlp import extract_activations, requirement_dimension, resolve_head

    head = resolve_head(model)
    last_layer = int(model.config.num_hidden_layers) - 1  # pyright: ignore[reportAttributeAccessIssue]
    acts = extract_activations(model, token_stream, layer=last_layer, ctx=64, n_eval_chunks=3, n_fit=100)
    mu = acts.fit.mean(0)
    _, _, v = torch.linalg.svd(acts.fit - mu, full_matrices=False)

    def pca(states: torch.Tensor, k: int) -> torch.Tensor:
        basis = v[:k]
        return (states - mu) @ basis.T @ basis + mu

    width = acts.fit.shape[1]
    report = requirement_dimension({"pca": pca}, head, acts.eval_batches, grid=[1, width])
    assert report.curves["pca"][width] == pytest.approx(report.intact_perplexity, rel=1e-4)
    required = report.required["pca"][1.05]
    assert required is not None and required <= width


def test_requirement_dimension_validates_grid(model: object) -> None:
    from igl.nlp import requirement_dimension, resolve_head

    head = resolve_head(model)
    with pytest.raises(IGLConfigError, match="grid"):
        requirement_dimension({}, head, [], grid=[])
    with pytest.raises(IGLConfigError, match="batches is empty"):
        requirement_dimension({}, head, [], grid=[1, 2])
