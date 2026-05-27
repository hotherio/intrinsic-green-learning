"""Geometry-agnostic IGL primitives: encoder, Green kernel, solver, trainer."""

from igl.core.encoder import LinearEncoder, MLPEncoder
from igl.core.kernel import GreenKernel
from igl.core.loss import CrossEntropyLoss, MSELoss
from igl.core.normalization import normalize_phi
from igl.core.solver import direct_solve_weights
from igl.core.trainer import MatryoshkaTrainer, TrainingHistory

__all__ = [
    "CrossEntropyLoss",
    "GreenKernel",
    "LinearEncoder",
    "MLPEncoder",
    "MSELoss",
    "MatryoshkaTrainer",
    "TrainingHistory",
    "direct_solve_weights",
    "normalize_phi",
]
