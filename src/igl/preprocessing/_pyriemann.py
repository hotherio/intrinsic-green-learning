"""Internal helper for pyriemann (and friends) gating.

``igl.preprocessing.AutoCovariances`` requires the ``[eeg]`` extra
(pyriemann + mne + moabb). This helper raises
:class:`igl.IGLDependencyError` at submodule import time so the failure
is loud and actionable, mirroring the ``igl.viz`` matplotlib pattern.
"""

from igl._optional import require_extra


def require_pyriemann() -> None:
    """Raise :class:`IGLDependencyError` if ``pyriemann`` is not importable."""
    require_extra(
        feature="EEG preprocessing (AutoCovariances)",
        extra="eeg",
        modules=["pyriemann"],
    )


__all__ = ["require_pyriemann"]
