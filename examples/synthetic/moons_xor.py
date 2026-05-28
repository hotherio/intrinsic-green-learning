"""Moons cls vs reg vs recon — demonstrate the canonical d_eff hierarchy.

The two interleaving half-moons in ``R²`` form a topologically 1-D manifold
embedded in ``R^16`` here. We solve three tasks on the *same* data through
the new sklearn-compatible wrappers:

- :class:`igl.IGLClassifier` — binary moon-id (cls task; ``d_eff`` should
  be small since one coordinate separates the moons).
- :class:`igl.IGLRegressor` — predict the original 2-D coordinates from
  the high-D embedding (reg task; needs at least 2 dims).
- :class:`igl.IGLAutoencoder` — reconstruct the 16-D embedding from the
  Matryoshka bottleneck (recon task; needs the manifold's intrinsic dim).

We then call :func:`igl.compare_d_eff` to check the empirical hierarchy
``d_eff(cls) ≤ d_eff(reg) ≤ d_eff(recon)``.

Run with::

    python -m examples.synthetic.moons_xor
"""

import warnings

import igl
from examples._utils import git_short_sha, make_run_dir, save_curve, set_seed
from igl.data import embed_in_high_dim, make_moons

EXAMPLE_NAME = "moons_xor"


def main() -> None:
    set_seed(42)

    n_total = 1200
    ambient_dim = 16

    x_2d, y_cls = make_moons(n_total, noise=0.1, seed=42)
    x = embed_in_high_dim(x_2d, target_dim=ambient_dim, seed=123)
    x_np = x.numpy()
    y_cls_np = y_cls.numpy()
    y_reg_np = x_2d.numpy()  # original 2-D coordinates as regression target

    fast = igl.IGLConfig(
        matryoshka=igl.MatryoshkaConfig(
            epochs=200,
            batch_size=128,
            inner_batch_size=n_total,
            scheduler=igl.SchedulerType.COSINE_WARM_RESTARTS,
            early_stop_patience=None,
            verbose=False,
        ),
    )

    common_kwargs = {
        "max_dim": 10,
        "n_anchors": 32,
        "n_scales": 3,
        "encoder_hidden": (96, 48),
        "random_state": 42,
        "config": fast,
    }

    print("=" * 60)
    print("Moons in R^16 — same data, three tasks")
    print("=" * 60)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        print("\nFitting IGLClassifier (cls)…")
        clf = igl.IGLClassifier(**common_kwargs).fit(x_np, y_cls_np)
        print(f"  val_acc = {clf.score(x_np, y_cls_np):.3f}")
        print(f"  d_eff   = {clf.effective_dimension_}")

        print("\nFitting IGLRegressor (reg)…")
        reg = igl.IGLRegressor(**common_kwargs).fit(x_np, y_reg_np)
        print(f"  R²      = {reg.score(x_np, y_reg_np):.3f}")
        print(f"  d_eff   = {reg.effective_dimension_}")

        print("\nFitting IGLAutoencoder (recon)…")
        ae = igl.IGLAutoencoder(**common_kwargs).fit(x_np)
        print(f"  d_eff   = {ae.effective_dimension_}")

    report = igl.compare_d_eff(
        cls=clf.dimension_curve_,
        reg=reg.dimension_curve_,
        recon=ae.dimension_curve_,
    )
    print("\n" + "=" * 60)
    print("Hierarchy check  d_cls ≤ d_reg ≤ d_recon")
    print("=" * 60)
    for task, d_eff in report.d_effs.items():
        print(f"  {task:>5s}: d_eff = {d_eff}")
    print(f"  hierarchy_holds: {report.hierarchy_holds}")

    # Persist results.
    run_dir = make_run_dir(EXAMPLE_NAME)
    for task_name, curve in (
        ("cls", clf.dimension_curve_),
        ("reg", reg.dimension_curve_),
        ("recon", ae.dimension_curve_),
    ):
        save_curve(run_dir, curve, name=f"{task_name}_curve.csv")
    print(f"\nResults written to: {run_dir}")
    print(f"git short sha: {git_short_sha()}")


if __name__ == "__main__":
    main()
