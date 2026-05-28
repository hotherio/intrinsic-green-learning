"""Optional plotting helpers. Requires the ``[viz]`` extra.

Importing any submodule of ``igl.viz`` raises :class:`igl.IGLDependencyError`
when ``matplotlib`` is missing, so the failure is loud and actionable rather
than a deep ``ImportError`` inside a plotting call.
"""

from igl.viz._matplotlib import require_matplotlib
from igl.viz.dimension_curve import plot_dimension_curve

__all__ = ["plot_dimension_curve", "require_matplotlib"]
