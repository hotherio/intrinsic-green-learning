"""Tests for the ``igl.viz`` module.

The module is gated behind the ``[viz]`` extra. We exercise both paths:

- When ``matplotlib`` is missing (simulated via monkeypatch), calling
  :func:`igl.viz.require_matplotlib` raises :class:`igl.IGLDependencyError`.
- When ``matplotlib`` is available, :func:`igl.viz.plot_dimension_curve`
  returns a configured ``Axes``.
"""

from __future__ import annotations

import importlib

import pytest

from igl import IGLDependencyError, _optional
from igl.viz import _matplotlib as mpl_helper


def _is_matplotlib_available() -> bool:
    return importlib.util.find_spec("matplotlib") is not None


pytestmark_mpl = pytest.mark.skipif(
    not _is_matplotlib_available(),
    reason="matplotlib not installed; install the [viz] extra to run viz tests.",
)


def test_require_matplotlib_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_optional, "find_spec", lambda _name: None)
    with pytest.raises(IGLDependencyError, match="plotting"):
        mpl_helper.require_matplotlib()


@pytestmark_mpl
def test_plot_dimension_curve_returns_axes_with_label() -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from igl.viz import plot_dimension_curve  # noqa: PLC0415

    fig, ax = plt.subplots()
    out = plot_dimension_curve(
        {1: 0.5, 2: 0.1, 3: 0.05, 4: 0.05},
        ax=ax,
        elbow=3,
        title="t",
        label="cls",
        log_y=True,
    )
    assert out.get_title() == "t"
    plt.close(fig)


@pytestmark_mpl
def test_plot_dimension_curve_creates_axes_when_none() -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from igl.viz import plot_dimension_curve  # noqa: PLC0415

    ax = plot_dimension_curve({1: 1.0, 2: 0.5, 3: 0.4}, log_y=False)
    assert ax is not None
    plt.close(ax.figure)


def test_require_matplotlib_passes_when_available() -> None:
    """No exception expected when matplotlib is installed in the dev env."""
    mpl_helper.require_matplotlib()
