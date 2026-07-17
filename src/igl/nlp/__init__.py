"""Language-model probing helpers. Requires the ``[nlp]`` extra.

Verified plumbing for head-in-the-loop measurements on HuggingFace causal
LMs: :func:`resolve_head` probes the head convention instead of assuming
it, :class:`ContinueFrom` builds identity-gated patch-and-continue
closures, :func:`extract_activations` produces disjoint fit/eval splits,
and :func:`requirement_dimension` sweeps projection arms against the
frozen head. Importing any submodule raises :class:`igl.IGLDependencyError`
when ``transformers`` is missing, so the failure is loud and actionable.
"""

from igl.nlp._gate import require_transformers
from igl.nlp.activations import ActivationSet, extract_activations
from igl.nlp.continuation import ContinueFrom
from igl.nlp.head import HeadClosure, resolve_head
from igl.nlp.requirement import Projector, RequirementReport, requirement_dimension

__all__ = [
    "ActivationSet",
    "ContinueFrom",
    "HeadClosure",
    "Projector",
    "RequirementReport",
    "extract_activations",
    "require_transformers",
    "requirement_dimension",
    "resolve_head",
]
