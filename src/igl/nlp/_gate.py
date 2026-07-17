"""Optional-dependency gate for :mod:`igl.nlp`."""

from igl._optional import require_extra


def require_transformers() -> None:
    """Raise :class:`igl.IGLDependencyError` unless the ``nlp`` extra is installed."""
    require_extra(feature="nlp helpers", extra="nlp", modules=["transformers"])
