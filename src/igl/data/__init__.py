"""Synthetic data generators and (optional) domain-specific loaders."""

from igl.data.synthetic import (
    embed_in_high_dim,
    make_flat_torus,
    make_flat_torus_labels,
    make_moons,
    make_swiss_roll,
)

__all__ = [
    "embed_in_high_dim",
    "make_flat_torus",
    "make_flat_torus_labels",
    "make_moons",
    "make_swiss_roll",
]
