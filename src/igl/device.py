"""Torch device autodetection and the per-device execution backends.

The sklearn estimators default to CPU; :func:`get_device` picks the best
available accelerator for users who want to opt in explicitly::

    estimator = IGLRegressor(device=igl.get_device())

:func:`select_backend` returns the execution branch the trainer uses on a
device (:class:`CpuBackend`, :class:`MpsBackend`, :class:`CudaBackend`).
"""

import torch

from igl.core._backend import Backend, CpuBackend, CudaBackend, MpsBackend, select_backend

__all__ = ["Backend", "CpuBackend", "CudaBackend", "MpsBackend", "get_device", "select_backend"]


def get_device() -> torch.device:
    """Return the best available torch device.

    Preference order: Apple ``mps``, then ``cuda``, then ``cpu``. Each device
    type has its own execution branch (see :func:`select_backend`): the CPU
    branch is the bit-exact reference, the MPS and CUDA branches keep every
    tensor on the device and solve the readout with an on-device Cholesky.

    Returns:
        The selected :class:`torch.device`.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
