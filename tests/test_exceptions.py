"""Tests for :mod:`igl.exceptions`."""

import pytest

from igl import (
    IGLConfigError,
    IGLConvergenceError,
    IGLDependencyError,
    IGLError,
    IGLNotFittedError,
)


def test_all_subclasses_inherit_from_iglerror() -> None:
    for subclass in (IGLConfigError, IGLConvergenceError, IGLDependencyError, IGLNotFittedError):
        assert issubclass(subclass, IGLError)


def test_iglerror_is_a_plain_exception_subclass() -> None:
    assert issubclass(IGLError, Exception)


def test_convergence_error_carries_structured_attributes() -> None:
    epoch = 42
    last_loss = 1.5e-3
    with pytest.raises(IGLConvergenceError) as exc_info:
        raise IGLConvergenceError(epoch=epoch, last_loss=last_loss)
    err = exc_info.value
    assert err.epoch == epoch
    assert err.last_loss == pytest.approx(last_loss)
    assert str(epoch) in str(err)
    assert "0.0015" in str(err)


def test_dependency_error_carries_feature_extra_and_missing() -> None:
    with pytest.raises(IGLDependencyError) as exc_info:
        raise IGLDependencyError(feature="plotting", extra="viz", missing=["matplotlib"])
    err = exc_info.value
    assert err.feature == "plotting"
    assert err.extra == "viz"
    assert err.missing == ("matplotlib",)
    message = str(err)
    assert "plotting" in message
    assert "matplotlib" in message
    assert "intrinsic-green-learning[viz]" in message


def test_dependency_error_lists_multiple_missing_modules() -> None:
    err = IGLDependencyError(feature="eeg", extra="eeg", missing=["mne", "moabb"])
    assert err.missing == ("mne", "moabb")
    assert "mne" in str(err)
    assert "moabb" in str(err)


def test_simple_subclasses_have_no_extra_attributes() -> None:
    """:class:`IGLConfigError` and :class:`IGLNotFittedError` are bare markers."""
    for cls in (IGLConfigError, IGLNotFittedError):
        err = cls("boom")
        assert str(err) == "boom"
