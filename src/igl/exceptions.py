"""Exception hierarchy raised by :mod:`igl`.

A single base class :class:`IGLError` with one level of subclasses; consumers
can catch the base or a specific subclass without chasing deep inheritance trees.
"""

from collections.abc import Sequence


class IGLError(Exception):
    """Base class for all ``igl``-raised exceptions."""


class IGLConfigError(IGLError):
    """Raised when a configuration value is invalid or inconsistent."""


class IGLConvergenceError(IGLError):
    """Raised when training fails to converge (e.g. loss becomes NaN/inf).

    Attributes:
        epoch: The epoch at which divergence was detected.
        last_loss: The most recent finite loss value seen before failure.
    """

    epoch: int
    last_loss: float

    def __init__(self, *, epoch: int, last_loss: float, message: str | None = None) -> None:
        super().__init__(message or f"training diverged at epoch {epoch} (last finite loss: {last_loss:.6g})")
        self.epoch = epoch
        self.last_loss = last_loss


class IGLDependencyError(IGLError):
    """Raised when an optional feature is used without its extras installed.

    Attributes:
        feature: A human-readable name for the requested feature.
        extra: The ``[extra]`` group that provides the missing modules.
        missing: The specific module names that could not be imported.
    """

    feature: str
    extra: str
    missing: tuple[str, ...]

    def __init__(self, *, feature: str, extra: str, missing: Sequence[str]) -> None:
        modules = ", ".join(missing)
        hint = f"pip install intrinsic-green-learning[{extra}]"
        super().__init__(f"feature {feature!r} requires modules [{modules}]; install with: {hint}")
        self.feature = feature
        self.extra = extra
        self.missing = tuple(missing)


class IGLNotFittedError(IGLError):
    """Raised when a model method is called before :meth:`fit`."""


__all__ = [
    "IGLConfigError",
    "IGLConvergenceError",
    "IGLDependencyError",
    "IGLError",
    "IGLNotFittedError",
]
