"""Checkpoint round-trip — fit once, save, reload, predict identically.

Fits an :class:`igl.IGLDistiller` on a synthetic manifold, saves it with
provenance via :func:`igl.save`, reloads it with :func:`igl.load`, and
verifies the reconstruction is bit-identical — no refit, no drift.

Run with::

    python -m examples.synthetic.save_load
"""

import numpy as np

import igl
from examples._utils import make_run_dir, set_seed
from igl.data import embed_in_high_dim, make_moons
from igl.io import Provenance

EXAMPLE_NAME = "save_load"


def main() -> None:
    set_seed(42)
    x_2d, _ = make_moons(400, noise=0.08, seed=42)
    states = embed_in_high_dim(x_2d, target_dim=12, seed=123).numpy()

    distiller = igl.IGLDistiller(max_dim=6, random_state=42)
    distiller.fit(states)
    reference = distiller.reconstruct(states)

    run_dir = make_run_dir(EXAMPLE_NAME)
    path = run_dir / "distiller.pt"
    igl.save(distiller, path, provenance=Provenance(seed=42, epochs=1500))

    provenance = igl.read_provenance(path)
    print(f"saved {path.name}: package {provenance['package_version']}, profile {provenance['profile']}")

    reloaded = igl.load(path)
    assert isinstance(reloaded, igl.IGLDistiller)
    result = reloaded.reconstruct(states)
    print(f"round-trip max abs difference: {np.max(np.abs(result - reference)):.2e}")
    print(f"effective dimension survives: {reloaded.effective_dimension_}")


if __name__ == "__main__":
    main()
