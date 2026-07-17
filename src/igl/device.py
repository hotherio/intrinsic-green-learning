"""Torch device autodetection.

The sklearn estimators default to CPU; :func:`get_device` picks the best
available accelerator for users who want to opt in explicitly::

    estimator = IGLRegressor(device=igl.get_device())
"""

import torch

__all__ = ["get_device"]


def get_device() -> torch.device:
    """Return the best available torch device.

    Preference order: Apple ``mps``, then ``cuda``, then ``cpu``. The
    closed-form solver always runs on CPU regardless (``torch.linalg.lstsq``
    is unreliable on MPS); tensors round-trip transparently.

    Returns:
        The selected :class:`torch.device`.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
