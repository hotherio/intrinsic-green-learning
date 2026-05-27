"""Tests for the optional-dependencies gating helper."""

import sys
from importlib.machinery import ModuleSpec

import pytest

from igl._optional import require_extra
from igl.exceptions import IGLDependencyError


def test_require_extra_passes_when_modules_resolve() -> None:
    # ``sys`` is always importable.
    require_extra(feature="stdlib smoke check", extra="all", modules=["sys"])


def test_require_extra_raises_when_a_module_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_find_spec(name: str) -> ModuleSpec | None:
        if name == "definitely_not_a_real_module":
            return None
        return ModuleSpec(name=name, loader=None)

    monkeypatch.setattr("igl._optional.find_spec", fake_find_spec)

    with pytest.raises(IGLDependencyError) as exc_info:
        require_extra(feature="fake feature", extra="fake", modules=["sys", "definitely_not_a_real_module"])

    assert exc_info.value.missing == ("definitely_not_a_real_module",)
    assert "fake feature" in str(exc_info.value)


def test_require_extra_reports_all_missing_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("igl._optional.find_spec", lambda _name: None)
    with pytest.raises(IGLDependencyError) as exc_info:
        require_extra(feature="multi", extra="all", modules=["a", "b", "c"])
    assert exc_info.value.missing == ("a", "b", "c")


def test_require_extra_does_not_pollute_sys_modules() -> None:
    # The helper uses ``find_spec``, not ``import_module``, so it must not
    # leave imported modules in ``sys.modules`` as a side effect.
    snapshot = set(sys.modules)
    require_extra(feature="purity", extra="all", modules=["json"])
    assert set(sys.modules) - snapshot == set()
