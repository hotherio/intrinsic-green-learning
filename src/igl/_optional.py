"""Helpers for gating modules behind optional dependency extras.

Each gated submodule (``igl.viz``, ``igl.data.eeg``, …) calls
:func:`require_extra` at import time. The check raises
:class:`igl.IGLDependencyError` with an actionable install hint when any
listed module cannot be imported, so that ``import igl.viz`` fails fast with a
helpful message instead of failing deep inside a function call.
"""

from collections.abc import Sequence
from importlib.util import find_spec

from igl.exceptions import IGLDependencyError


def require_extra(feature: str, extra: str, modules: Sequence[str]) -> None:
    """Raise :class:`IGLDependencyError` if any ``modules`` is missing.

    Args:
        feature: Human-readable name of the gated feature (e.g. ``"plotting"``).
        extra: The pip ``[extra]`` group that installs the missing modules.
        modules: Module names to check via :func:`importlib.util.find_spec`.

    Raises:
        IGLDependencyError: If at least one of ``modules`` cannot be resolved.
    """
    missing = [m for m in modules if find_spec(m) is None]
    if missing:
        raise IGLDependencyError(feature=feature, extra=extra, missing=missing)


__all__ = ["require_extra"]
