"""Required-dimension measurement with the frozen head in the loop.

Given one or more projection arms (anything that maps hidden states to a
k-dimensional reconstruction of them), :func:`requirement_dimension`
measures the perplexity the model's own head produces when reading each
reconstruction, sweeps k over a grid, and reports the smallest k that
stays within each tolerance of the intact perplexity. The head is the
consumer: nothing is retrained, so the number measures how many
dimensions the downstream computation actually reads.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch.nn.functional import cross_entropy

from igl.exceptions import IGLConfigError
from igl.nlp._gate import require_transformers

require_transformers()

__all__ = ["Projector", "RequirementReport", "requirement_dimension"]

Projector = Callable[[torch.Tensor, int], torch.Tensor]
"""A projection arm: ``(states [N, C], k) -> reconstructed states [N, C]``."""

_DEFAULT_TOLERANCES = (1.05, 1.10, 1.20)


@dataclass(frozen=True, slots=True, kw_only=True)
class RequirementReport:
    """Result of a required-dimension sweep.

    Attributes:
        intact_perplexity: Perplexity of the head reading the raw states.
        curves: Per arm, the perplexity at each swept k.
        required: Per arm, the smallest swept k whose perplexity is within
            ``tolerance * intact_perplexity``, or ``None`` when no swept k
            qualifies (the requirement exceeds the grid).
    """

    intact_perplexity: float
    curves: dict[str, dict[int, float]]
    required: dict[str, dict[float, int | None]]


def _perplexity(
    head: Callable[[torch.Tensor], torch.Tensor],
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> float:
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for states, targets in batches:
            h = states if transform is None else transform(states)
            logits = head(h)
            total_nll += cross_entropy(logits, targets.to(logits.device), reduction="sum").item()
            total_tokens += targets.numel()
    return float(torch.exp(torch.tensor(total_nll / total_tokens)).item())


def requirement_dimension(
    projectors: Mapping[str, Projector],
    head: Callable[[torch.Tensor], torch.Tensor],
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    grid: Sequence[int],
    tolerances: Sequence[float] = _DEFAULT_TOLERANCES,
) -> RequirementReport:
    """Sweep projection arms over k and report per-tolerance requirements.

    Args:
        projectors: Named arms mapping ``(states, k)`` to reconstructed
            states of the same width (e.g. an IGL chart's
            project-then-reconstruct, or a PCA baseline).
        head: The frozen consumer mapping states to logits: an
            :class:`igl.nlp.HeadClosure` for the final layer, or an
            :class:`igl.nlp.ContinueFrom` for patch-and-continue profiles.
        batches: ``(states [N, C], targets [N])`` evaluation pairs; targets
            are next-token ids. Materialized once and reused per (arm, k).
        grid: The k values to sweep, in increasing order.
        tolerances: Multiplicative perplexity tolerances; for each, the
            report records the smallest swept k within
            ``tolerance * intact_perplexity``.

    Returns:
        A :class:`RequirementReport` with the intact perplexity, one
        perplexity curve per arm, and per-tolerance required dimensions.

    Raises:
        IGLConfigError: When ``grid`` is empty, not increasing, or
            ``batches`` is empty.
    """
    if not grid or list(grid) != sorted(set(grid)):
        raise IGLConfigError("grid must be a non-empty, strictly increasing sequence of k values")
    materialized = list(batches)
    if not materialized:
        raise IGLConfigError("batches is empty: at least one (states, targets) pair is required")
    intact = _perplexity(head, materialized)
    curves: dict[str, dict[int, float]] = {}
    required: dict[str, dict[float, int | None]] = {}
    for name, project in projectors.items():
        curve = {int(k): _perplexity(head, materialized, transform=lambda h, k=k, p=project: p(h, k)) for k in grid}
        curves[name] = curve
        required[name] = {float(tol): next((k for k in grid if curve[int(k)] <= tol * intact), None) for tol in tolerances}
    return RequirementReport(intact_perplexity=intact, curves=curves, required=required)
