"""Matplotlib helper for plotting Matryoshka dimension curves.

Requires the ``[viz]`` extra; importing this module raises
:class:`igl.IGLDependencyError` if ``matplotlib`` is missing.
"""

from typing import TYPE_CHECKING

from igl.viz._matplotlib import require_matplotlib

require_matplotlib()

import matplotlib.pyplot as plt  # noqa: E402

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from igl.types import DimensionCurve


def plot_dimension_curve(
    curve: "DimensionCurve",
    *,
    ax: "Axes | None" = None,
    elbow: int | None = None,
    title: str | None = None,
    label: str | None = None,
    log_y: bool = True,
) -> "Axes":
    """Plot a Matryoshka dimension curve.

    Args:
        curve: ``{k: loss}`` mapping (the ``dimension_curve_`` attribute of a
            fitted IGL estimator, or the return value of
            :func:`igl.eval_dimension_curve`).
        ax: Optional axes to draw on. Created via ``plt.subplots()`` when
            absent.
        elbow: If provided, draw a vertical marker line at this ``k``.
        title: Optional axes title.
        label: Optional line label (for use with multi-curve overlays).
        log_y: Use a log-scale Y axis (the elbow is usually clearer on log).

    Returns:
        The axes the curve was drawn on.
    """
    # matplotlib's stubs are partial — the runtime types are stable, so the
    # whole function is one big stub-tolerance zone.
    if ax is None:
        _fig, ax = plt.subplots(figsize=(6, 4))  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    ks = sorted(curve)
    values = [curve[k] for k in ks]
    ax.plot(ks, values, marker="o", label=label)  # pyright: ignore[reportUnknownMemberType]

    if elbow is not None:
        ax.axvline(elbow, linestyle="--", alpha=0.6, label=f"d_eff = {elbow}")  # pyright: ignore[reportUnknownMemberType]

    ax.set_xlabel("truncation level k")  # pyright: ignore[reportUnknownMemberType]
    ax.set_ylabel("curve score (lower is better)")  # pyright: ignore[reportUnknownMemberType]
    if log_y:
        ax.set_yscale("log")  # pyright: ignore[reportUnknownMemberType]
    if title:
        ax.set_title(title)  # pyright: ignore[reportUnknownMemberType]
    if label is not None or elbow is not None:
        ax.legend()  # pyright: ignore[reportUnknownMemberType]
    ax.grid(visible=True, alpha=0.3)  # pyright: ignore[reportUnknownMemberType]
    return ax


__all__ = ["plot_dimension_curve"]
