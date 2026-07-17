"""Head-convention resolution for HuggingFace causal language models.

Whether the unembedding applies to the last hidden state directly
(post-norm, e.g. GPT-2's ``hidden_states[-1]`` already includes ``ln_f``)
or only after the model's final normalization differs per family, and
getting it wrong silently corrupts every head-side measurement. The
resolver probes both conventions against the model's own logits and
returns a verified closure instead of guessing.
"""

from dataclasses import dataclass
from typing import cast

import torch

from igl.exceptions import IGLConfigError
from igl.nlp._gate import require_transformers
from igl.nlp._hf import HEAD_PATHS, NORM_PATHS, CausalLM, first_attr, resolve_attr

require_transformers()

__all__ = ["HeadClosure", "resolve_head"]


@dataclass(frozen=True, slots=True, kw_only=True)
class HeadClosure:
    """A verified hidden-states-to-logits closure.

    Attributes:
        post_norm: Whether the final hidden state is already normalized
            (the unembedding applies to it directly).
        lm_head: The unembedding module.
        norm: The final normalization module, when ``post_norm`` is false.
    """

    post_norm: bool
    lm_head: torch.nn.Module
    norm: torch.nn.Module | None

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        """Map states ``[..., C]`` to fp32 logits, casting through the head's dtype."""
        h = states.to(next(self.lm_head.parameters()).dtype)
        if not self.post_norm:
            assert self.norm is not None
            h = self.norm(h)
        return cast("torch.Tensor", self.lm_head(h)).float()


def resolve_head(model: object, *, probe_tokens: int = 8, atol: float = 1e-2) -> HeadClosure:
    """Probe and verify how the unembedding reads the final hidden state.

    Runs a short forward with hidden states, then checks which convention
    reproduces the model's own logits: the head on the raw last hidden
    state, or the head on the final norm of it (candidate norms:
    ``model.norm``, ``transformer.ln_f``, ``gpt_neox.final_layer_norm``).

    Args:
        model: A HuggingFace causal LM exposing ``lm_head`` (or
            ``embed_out``) and ``output_hidden_states``.
        probe_tokens: Length of the verification forward.
        atol: Absolute tolerance of the logits comparison, in fp32.

    Returns:
        A verified :class:`HeadClosure`.

    Raises:
        IGLConfigError: When no unembedding module is found, or neither
            convention reproduces the model's logits.
    """
    lm_head = cast("torch.nn.Module | None", first_attr(model, HEAD_PATHS))
    if lm_head is None:
        raise IGLConfigError(f"no unembedding module found (tried {', '.join(HEAD_PATHS)})")
    device = next(lm_head.parameters()).device
    input_ids = torch.arange(1, probe_tokens + 1, device=device).unsqueeze(0)
    with torch.no_grad():
        output = cast("CausalLM", model)(input_ids, output_hidden_states=True)
        hidden = output.hidden_states[-1]
        logits = output.logits.float()
        if torch.allclose(cast("torch.Tensor", lm_head(hidden)).float(), logits, atol=atol):
            return HeadClosure(post_norm=True, lm_head=lm_head, norm=None)
        for dotted in NORM_PATHS:
            norm = resolve_attr(model, dotted)
            if norm is None:
                continue
            candidate = cast("torch.nn.Module", norm)
            if torch.allclose(cast("torch.Tensor", lm_head(candidate(hidden))).float(), logits, atol=atol):
                return HeadClosure(post_norm=False, lm_head=lm_head, norm=candidate)
    raise IGLConfigError("neither head convention reproduces the model's logits; refusing to guess")
