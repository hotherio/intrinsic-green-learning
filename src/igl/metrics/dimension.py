"""Dimension-curve metrics: effective dimension extraction and cross-task comparison.

This module surfaces small utilities around the post-fit ``dimension_curve_``
that every sklearn-wrapped IGL estimator carries. The canonical use case is
verifying the empirical hierarchy

    d_eff(cls) ≤ d_eff(reg) ≤ d_eff(recon)

on the same underlying data.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from igl.matryoshka.dimension_curve import detect_elbow
from igl.types import DimensionCurve


def d_eff_from_curve(curve: DimensionCurve, *, ratio: float = 2.0) -> int:
    """Discovered effective dimension. Alias of :func:`igl.detect_elbow`."""
    return detect_elbow(curve, ratio=ratio)


@dataclass(frozen=True, slots=True)
class DimensionComparison:
    """Result of :func:`compare_d_eff`.

    Attributes:
        d_effs: Mapping from task name to its discovered effective dimension.
        hierarchy_holds: ``True`` iff iterating the task names in insertion
            order yields non-decreasing ``d_eff`` values — i.e. the
            ``cls ≤ reg ≤ recon`` ordering is satisfied. The check uses the
            insertion order of ``d_effs``.
    """

    d_effs: Mapping[str, int]
    hierarchy_holds: bool


def compare_d_eff(
    *,
    ratio: float = 2.0,
    **curves: DimensionCurve,
) -> DimensionComparison:
    """Compute and compare ``d_eff`` for multiple task curves.

    Pass each task's dimension curve as a keyword argument; the key is the
    task name. The function returns a :class:`DimensionComparison` containing
    each task's discovered effective dimension and a boolean indicating
    whether the values appear in non-decreasing order (the canonical IGL
    hierarchy).

    Example::

        from igl import compare_d_eff
        report = compare_d_eff(
            cls=classifier.dimension_curve_,
            reg=regressor.dimension_curve_,
            recon=autoencoder.dimension_curve_,
        )
        assert report.hierarchy_holds  # 1 <= 2 <= 2 on swiss roll

    Args:
        ratio: Forwarded to :func:`igl.detect_elbow` for each curve.
        **curves: Task name → dimension curve.

    Returns:
        A :class:`DimensionComparison`.
    """
    d_effs: dict[str, int] = {name: detect_elbow(curve, ratio=ratio) for name, curve in curves.items()}
    values = list(d_effs.values())
    hierarchy_holds = all(values[i] <= values[i + 1] for i in range(len(values) - 1))
    return DimensionComparison(d_effs=d_effs, hierarchy_holds=hierarchy_holds)


__all__ = ["DimensionComparison", "compare_d_eff", "d_eff_from_curve"]
