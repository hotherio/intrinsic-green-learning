"""Matryoshka random-truncation sampling and post-hoc dimension-curve analysis."""

from igl.matryoshka.dimension_curve import detect_elbow, eval_dimension_curve
from igl.matryoshka.sampler import PowerLawSampler, UniformSampler

__all__ = [
    "PowerLawSampler",
    "UniformSampler",
    "detect_elbow",
    "eval_dimension_curve",
]
