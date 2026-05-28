"""Internal helper for matplotlib gating.

``igl.viz`` requires the ``[viz]`` extra. Submodules under ``igl.viz`` call
:func:`require_matplotlib` at import time so a missing ``matplotlib``
surfaces as :class:`igl.IGLDependencyError` immediately rather than deep
inside a plotting routine.
"""

from igl._optional import require_extra


def require_matplotlib() -> None:
    """Raise :class:`IGLDependencyError` if ``matplotlib`` is not importable."""
    require_extra(feature="plotting", extra="viz", modules=["matplotlib"])


__all__ = ["require_matplotlib"]
