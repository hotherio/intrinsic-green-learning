"""Verified patch-and-continue closures for causal language models.

:class:`ContinueFrom` runs the transformer blocks after a chosen layer on
supplied hidden states, then the final norm and unembedding, producing the
logits the model itself would have produced from those states. The closure
is only handed out after an identity gate: continuing from the model's own
intermediate states must reproduce the model's own logits, so any silent
architecture mismatch (missing rotary embeddings, wrong block list, wrong
norm) fails loudly at construction instead of corrupting measurements.
"""

from collections.abc import Callable, Sequence
from typing import cast

import torch

from igl.exceptions import IGLConfigError
from igl.nlp._gate import require_transformers
from igl.nlp._hf import BLOCK_PATHS, HEAD_PATHS, NORM_PATHS, ROTARY_PATHS, CausalLM, first_attr

require_transformers()

__all__ = ["ContinueFrom"]


class ContinueFrom:
    """Run blocks ``layer+1 .. end`` plus final norm and head on given states.

    Args:
        model: A HuggingFace causal LM (GPT-2, LLaMA/Qwen, GPT-NeoX
            families: block lists at ``transformer.h``, ``model.layers``,
            or ``gpt_neox.layers``).
        layer: Zero-based block index the states come out of; the closure
            applies blocks ``layer+1`` onward. Must satisfy
            ``0 <= layer < n_blocks``.
        probe_tokens: Length of the identity-gate forward.
        atol: Absolute logits tolerance of the identity gate, in fp32.

    Raises:
        IGLConfigError: When the architecture is not recognized, ``layer``
            is out of range, or the identity gate fails (continuing from
            the model's own states does not reproduce its logits).
    """

    def __init__(self, model: object, layer: int, *, probe_tokens: int = 16, atol: float = 1e-2) -> None:
        blocks = first_attr(model, BLOCK_PATHS)
        norm = first_attr(model, NORM_PATHS)
        head = first_attr(model, HEAD_PATHS)
        if blocks is None or norm is None or head is None:
            raise IGLConfigError(
                "unrecognized architecture: could not locate blocks, final norm, and unembedding "
                f"(tried blocks at {', '.join(BLOCK_PATHS)})"
            )
        self._blocks = cast("Sequence[torch.nn.Module]", blocks)
        self._norm = cast("torch.nn.Module", norm)
        self._head = cast("torch.nn.Module", head)
        self._rotary = cast(
            "Callable[[torch.Tensor, torch.Tensor], object] | None",
            first_attr(model, ROTARY_PATHS),
        )
        self.n_blocks = len(self._blocks)
        if not 0 <= layer < self.n_blocks:
            raise IGLConfigError(f"layer {layer} out of range for a model with {self.n_blocks} blocks")
        self.layer = layer
        self._device = next(self._head.parameters()).device
        self._apply_norm = True
        self._verify(model, probe_tokens=probe_tokens, atol=atol)

    def _block_kwargs(self, n_tokens: int, width: int) -> dict[str, object]:
        """Position embeddings for rotary architectures, computed once per call."""
        if self._rotary is None:
            return {}
        positions = torch.arange(n_tokens, device=self._device).unsqueeze(0)
        dummy = torch.zeros(1, n_tokens, width, device=self._device)
        return {"position_embeddings": self._rotary(dummy, positions)}

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        """Continue from ``states`` ``[T, C]`` to fp32 logits ``[T, V]``."""
        x = states.to(self._device).to(next(self._head.parameters()).dtype).unsqueeze(0)
        kwargs = self._block_kwargs(states.shape[0], states.shape[1])
        for block in self._blocks[self.layer + 1 :]:
            try:
                out = block(x, **kwargs)
            except TypeError:
                out = block(x)
            x = cast("torch.Tensor", out[0] if isinstance(out, tuple) else out)
        if self._apply_norm:
            x = self._norm(x)
        return cast("torch.Tensor", self._head(x))[0].float()

    def _verify(self, model: object, *, probe_tokens: int, atol: float) -> None:
        """Identity gate, probing the final-norm convention rather than assuming it.

        Continuing from the last block's stored hidden state must not re-apply
        the final norm on post-norm families (GPT-2 bakes ``ln_f`` into
        ``hidden_states[-1]``), so both conventions are tried against the
        model's own logits.
        """
        input_ids = torch.arange(1, probe_tokens + 1, device=self._device).unsqueeze(0)
        with torch.no_grad():
            output = cast("CausalLM", model)(input_ids, output_hidden_states=True)
            states = output.hidden_states[self.layer + 1][0]
            reference = output.logits[0].float()
            deviations: list[float] = []
            for apply_norm in (True, False):
                self._apply_norm = apply_norm
                logits = self(states)
                if torch.allclose(logits, reference, atol=atol):
                    return
                deviations.append((logits - reference).abs().max().item())
        raise IGLConfigError(
            f"identity gate failed at layer {self.layer}: continuing from the model's own states does not "
            f"reproduce its logits under either norm convention (max abs deviations "
            f"{deviations[0]:.3e} with final norm, {deviations[1]:.3e} without, atol {atol:g})"
        )
