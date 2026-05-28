"""Tests for the metrics module."""

import sys

import pytest

from igl import IGLConfigError, IGLDependencyError, compare_d_eff, d_eff_from_curve
from igl.metrics.dimension import DimensionComparison
from igl.metrics.elbow import detect_elbow_kneedle, detect_elbow_log_ratio


def test_d_eff_from_curve_is_an_alias() -> None:
    curve = {1: 1.0, 2: 0.5, 3: 0.4}
    assert d_eff_from_curve(curve) == 2


def test_d_eff_from_curve_passes_ratio() -> None:
    """A larger ratio is more permissive (counts more drops as substantial)."""
    curve = {1: 1.0, 2: 0.1, 3: 0.05, 4: 0.04}
    # ratio=2.0: only the first big drop counts → elbow at k=2.
    assert d_eff_from_curve(curve, ratio=2.0) == 2
    # ratio=5.0: the second drop also clears the cutoff → elbow at k=3.
    assert d_eff_from_curve(curve, ratio=5.0) == 3


def test_compare_d_eff_canonical_hierarchy_holds() -> None:
    cls_curve = {1: 0.5, 2: 0.05, 3: 0.05}
    reg_curve = {1: 0.5, 2: 0.2, 3: 0.05, 4: 0.05}
    recon_curve = {1: 0.5, 2: 0.4, 3: 0.2, 4: 0.05, 5: 0.05}

    report = compare_d_eff(cls=cls_curve, reg=reg_curve, recon=recon_curve)
    assert isinstance(report, DimensionComparison)
    assert report.hierarchy_holds is True
    # Values must be non-decreasing in insertion order.
    values = list(report.d_effs.values())
    assert values == sorted(values)


def test_compare_d_eff_detects_violation() -> None:
    # cls dominates reg here — violates the expected ordering.
    cls_curve = {1: 1.0, 2: 0.5, 3: 0.2, 4: 0.05, 5: 0.05}
    reg_curve = {1: 0.1, 2: 0.05, 3: 0.05}
    report = compare_d_eff(cls=cls_curve, reg=reg_curve)
    assert report.hierarchy_holds is False


def test_detect_elbow_log_ratio_default() -> None:
    curve = {1: 1.0, 2: 0.05, 3: 0.04}
    assert detect_elbow_log_ratio(curve) == 2


def test_detect_elbow_kneedle_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """If ``kneed`` isn't installed, the detector raises IGLDependencyError."""
    from igl import _optional  # noqa: PLC0415

    monkeypatch.setattr(_optional, "find_spec", lambda _name: None)
    with pytest.raises(IGLDependencyError, match="kneedle"):
        detect_elbow_kneedle({1: 1.0, 2: 0.5})


def test_detect_elbow_kneedle_rejects_empty_curve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even with the extra fake-installed, an empty curve fails early."""
    from importlib.machinery import ModuleSpec  # noqa: PLC0415

    from igl import _optional  # noqa: PLC0415

    monkeypatch.setattr(_optional, "find_spec", lambda name: ModuleSpec(name=name, loader=None))

    # Provide a minimal stub `kneed` module so the import inside the helper
    # succeeds without the dependency.
    class _StubLocator:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.knee = None

    stub = type(sys)("kneed")
    stub.KneeLocator = _StubLocator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kneed", stub)
    with pytest.raises(IGLConfigError, match="curve must contain"):
        detect_elbow_kneedle({})


def test_detect_elbow_kneedle_runs_with_stub_kneed(monkeypatch: pytest.MonkeyPatch) -> None:
    """When kneed.KneeLocator returns a knee, we return its integer value."""
    from importlib.machinery import ModuleSpec  # noqa: PLC0415

    from igl import _optional  # noqa: PLC0415

    monkeypatch.setattr(_optional, "find_spec", lambda name: ModuleSpec(name=name, loader=None))

    class _StubLocator:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.knee = 2

    stub = type(sys)("kneed")
    stub.KneeLocator = _StubLocator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kneed", stub)
    assert detect_elbow_kneedle({1: 1.0, 2: 0.1, 3: 0.05}) == 2


def test_detect_elbow_kneedle_falls_back_when_no_knee_found(monkeypatch: pytest.MonkeyPatch) -> None:
    from importlib.machinery import ModuleSpec  # noqa: PLC0415

    from igl import _optional  # noqa: PLC0415

    monkeypatch.setattr(_optional, "find_spec", lambda name: ModuleSpec(name=name, loader=None))

    class _StubLocator:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.knee = None

    stub = type(sys)("kneed")
    stub.KneeLocator = _StubLocator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kneed", stub)
    assert detect_elbow_kneedle({1: 1.0, 2: 0.5, 3: 0.4}) == 1


def test_dimension_comparison_is_frozen() -> None:
    report = DimensionComparison(d_effs={"cls": 1, "reg": 2}, hierarchy_holds=True)
    with pytest.raises(AttributeError):
        report.hierarchy_holds = False  # type: ignore[misc]
