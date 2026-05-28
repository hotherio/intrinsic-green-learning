"""Elbow-detection helpers for dimension curves.

Two implementations:

- :func:`detect_elbow_log_ratio` (default, always available) — the same
  log-scale ratio detector exposed at the top level as :func:`igl.detect_elbow`.
- :func:`detect_elbow_kneedle` — wraps the optional ``kneed`` package and
  raises :class:`igl.IGLDependencyError` if it isn't installed. Available
  via the ``[elbow]`` extra: ``pip install intrinsic-green-learning[elbow]``.
"""

from collections.abc import Mapping

from igl.exceptions import IGLConfigError
from igl.matryoshka.dimension_curve import detect_elbow as _detect_elbow_log_ratio

DimensionCurveMapping = Mapping[int, float]


def detect_elbow_log_ratio(curve: DimensionCurveMapping, *, ratio: float = 2.0) -> int:
    """Log-scale first-derivative elbow detector. Default, scipy-free."""
    return _detect_elbow_log_ratio(curve, ratio=ratio)


def detect_elbow_kneedle(curve: DimensionCurveMapping, *, sensitivity: float = 1.0) -> int:
    """Kneedle elbow detector. Requires the ``[elbow]`` extra.

    Args:
        curve: ``{k: loss}`` dict.
        sensitivity: Sensitivity parameter forwarded to ``KneeLocator``.

    Raises:
        IGLDependencyError: If the ``kneed`` package is not installed.
        IGLConfigError: If the curve is empty.
    """
    from igl._optional import require_extra  # noqa: PLC0415

    require_extra(feature="kneedle elbow detection", extra="elbow", modules=["kneed"])
    if not curve:
        raise IGLConfigError("curve must contain at least one entry")

    from kneed import KneeLocator  # noqa: PLC0415  # pyright: ignore[reportMissingImports, reportUnknownVariableType]

    ks = sorted(curve)
    losses = [curve[k] for k in ks]
    locator = KneeLocator(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        ks,
        losses,
        S=sensitivity,
        curve="convex",
        direction="decreasing",
    )
    knee = locator.knee  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if knee is None:
        return ks[0]
    return int(knee)  # pyright: ignore[reportUnknownArgumentType]


__all__ = ["detect_elbow_kneedle", "detect_elbow_log_ratio"]
