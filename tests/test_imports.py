"""Surface tests: the top-level ``igl`` package exposes the documented symbols."""

import igl


def test_version_is_a_string() -> None:
    assert isinstance(igl.__version__, str)
    assert igl.__version__  # not empty


def test_version_is_in___all__() -> None:
    assert "__version__" in igl.__all__


def test_exception_symbols_in___all__() -> None:
    for name in (
        "IGLConfigError",
        "IGLConvergenceError",
        "IGLDependencyError",
        "IGLError",
        "IGLNotFittedError",
    ):
        assert name in igl.__all__, f"{name!r} missing from igl.__all__"


def test_all_exports_resolve() -> None:
    """Every name in ``__all__`` must actually be importable from ``igl``."""
    for name in igl.__all__:
        assert hasattr(igl, name), f"igl.__all__ lists {name!r} but the attribute is missing"
