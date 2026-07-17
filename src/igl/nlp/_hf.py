"""Shared HuggingFace-model plumbing for :mod:`igl.nlp`.

Attribute-path tables and structural protocols used by the head resolver
and the continuation closure. Kept in one place so both probe the same
candidate architectures.
"""

from collections.abc import Sequence
from typing import Protocol

import torch

NORM_PATHS = ("model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm")
HEAD_PATHS = ("lm_head", "embed_out")
BLOCK_PATHS = ("transformer.h", "model.layers", "gpt_neox.layers")
ROTARY_PATHS = ("model.rotary_emb", "gpt_neox.rotary_emb")


class LMOutput(Protocol):
    """The slice of a causal-LM forward output that :mod:`igl.nlp` reads."""

    hidden_states: tuple[torch.Tensor, ...]
    logits: torch.Tensor


class CausalLM(Protocol):
    """A causal LM callable with hidden-state outputs."""

    def __call__(self, input_ids: torch.Tensor, *, output_hidden_states: bool) -> LMOutput: ...


def resolve_attr(root: object, dotted: str) -> object | None:
    """Resolve a dotted attribute path (``"transformer.ln_f"``), or ``None``."""
    node: object | None = root
    for part in dotted.split("."):
        node = getattr(node, part, None)
        if node is None:
            return None
    return node


def first_attr(root: object, paths: Sequence[str]) -> object | None:
    """Return the first resolvable path in ``paths``, or ``None``."""
    return next((m for p in paths if (m := resolve_attr(root, p)) is not None), None)
