"""Activation extraction with a fit/eval disjointness guarantee.

:func:`extract_activations` slices a token stream into fixed-length
context chunks, runs the model once per chunk, and splits the resulting
hidden states into an evaluation prefix (kept as per-chunk
``(states, targets)`` pairs for head-in-the-loop measurement) and a fit
pool (subsampled positions from the remaining chunks, for fitting charts
or baselines). The two come from disjoint chunks by construction, so a
chart can never be fit on the tokens it is later scored on.
"""

from dataclasses import dataclass
from typing import cast

import torch

from igl.exceptions import IGLConfigError
from igl.nlp._gate import require_transformers
from igl.nlp._hf import CausalLM

require_transformers()

__all__ = ["ActivationSet", "extract_activations"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ActivationSet:
    """Extracted hidden states, split into disjoint fit and eval sets.

    Attributes:
        fit: Subsampled states ``[n_fit, C]`` from the post-eval chunks.
        eval_batches: One ``(states [ctx-1, C], targets [ctx-1])`` pair per
            evaluation chunk; targets are the next-token ids, ready for
            :func:`igl.nlp.requirement_dimension`.
        layer: The zero-based block index the states come out of.
        ctx: The context length used for chunking.
    """

    fit: torch.Tensor
    eval_batches: list[tuple[torch.Tensor, torch.Tensor]]
    layer: int
    ctx: int


def extract_activations(
    model: object,
    input_ids: torch.Tensor,
    *,
    layer: int,
    ctx: int = 512,
    n_eval_chunks: int = 20,
    n_fit: int = 20_000,
    seed: int = 42,
) -> ActivationSet:
    """Extract layer activations with disjoint fit and eval splits.

    The token stream is cut into ``ctx``-length chunks. The first
    ``n_eval_chunks`` become the evaluation set; the fit pool is sampled
    (without replacement, seeded) from all positions of the remaining
    chunks, so fit and eval tokens never overlap.

    Args:
        model: A HuggingFace causal LM with hidden-state outputs.
        input_ids: A 1-D token-id tensor.
        layer: Zero-based block index; states are the residual stream
            after block ``layer`` (``hidden_states[layer + 1]``).
        ctx: Context length per forward.
        n_eval_chunks: Number of leading chunks reserved for evaluation.
        n_fit: Number of fit positions to sample; capped at the pool size.
        seed: Seed of the fit subsampling.

    Returns:
        An :class:`ActivationSet` on CPU, fp32.

    Raises:
        IGLConfigError: When ``input_ids`` is not 1-D or holds fewer than
            ``(n_eval_chunks + 1) * ctx`` tokens (no fit chunk would remain).
    """
    if input_ids.ndim != 1:
        raise IGLConfigError(f"input_ids must be 1-D, got shape {tuple(input_ids.shape)}")
    n_chunks = len(input_ids) // ctx
    if n_chunks <= n_eval_chunks:
        raise IGLConfigError(
            f"need more than {n_eval_chunks} chunks of {ctx} tokens for disjoint fit and eval splits, "
            f"got {n_chunks} ({len(input_ids)} tokens)"
        )
    lm = cast("CausalLM", model)
    device = next(cast("torch.nn.Module", model).parameters()).device
    eval_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    fit_pool: list[torch.Tensor] = []
    with torch.no_grad():
        for c in range(n_chunks):
            chunk = input_ids[c * ctx : (c + 1) * ctx].to(device).unsqueeze(0)
            states = lm(chunk, output_hidden_states=True).hidden_states[layer + 1][0].float().cpu()
            if c < n_eval_chunks:
                eval_batches.append((states[:-1], chunk[0, 1:].cpu()))
            else:
                fit_pool.append(states)
    pool = torch.cat(fit_pool)
    generator = torch.Generator().manual_seed(seed)
    keep = torch.randperm(len(pool), generator=generator)[: min(n_fit, len(pool))]
    return ActivationSet(fit=pool[keep], eval_batches=eval_batches, layer=layer, ctx=ctx)
