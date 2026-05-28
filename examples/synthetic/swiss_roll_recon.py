"""Swiss-roll reconstruction — recover the intrinsic 2-D structure.

The swiss roll in ``R³`` is a 2-D manifold (parameters ``t, h``). We:

1. Train :class:`igl.IGLAutoencoder` on the 3-D ambient data — d_eff should
   land near 2 (the manifold's intrinsic dimension).
2. Plot the dimension curve (if the ``[viz]`` extra is installed).
3. Optionally train an :class:`igl.IGLRegressor` predicting the ``(t, h)``
   parameters from ``X`` and confirm its d_eff is also ≈ 2.

Run with::

    python -m examples.synthetic.swiss_roll_recon

With ``pip install -e ".[viz]"`` installed, a ``curve.png`` is dropped into
the run directory; otherwise only CSV / JSON outputs are written.
"""

import warnings

import igl
from examples._utils import git_short_sha, make_run_dir, save_curve, set_seed
from igl.data import make_swiss_roll

EXAMPLE_NAME = "swiss_roll_recon"


def main() -> None:
    set_seed(42)

    n_total = 1000
    x, params = make_swiss_roll(n_total, noise=0.0, seed=42)
    x_np = x.numpy()
    params_np = params.numpy()

    config = igl.IGLConfig(
        matryoshka=igl.MatryoshkaConfig(
            epochs=300,
            batch_size=128,
            inner_batch_size=n_total,
            scheduler=igl.SchedulerType.COSINE_WARM_RESTARTS,
            early_stop_patience=None,
            verbose=False,
        ),
    )
    common_kwargs = {
        "max_dim": 8,
        "n_anchors": 32,
        "n_scales": 3,
        "encoder_hidden": (64, 32),
        "random_state": 42,
        "config": config,
    }

    print("=" * 60)
    print("Swiss roll in R^3 — autoencoder + regression")
    print("=" * 60)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        print("\nFitting IGLAutoencoder…")
        ae = igl.IGLAutoencoder(**common_kwargs).fit(x_np)
        print(f"  d_eff(recon) = {ae.effective_dimension_} (intrinsic dim = 2)")

        print("\nFitting IGLRegressor (predict (t, h) from x)…")
        reg = igl.IGLRegressor(**common_kwargs).fit(x_np, params_np)
        print(f"  R²           = {reg.score(x_np, params_np):.3f}")
        print(f"  d_eff(reg)   = {reg.effective_dimension_}")

    print("\nDimension curves:")
    for task_name, curve in (("autoencoder", ae.dimension_curve_), ("regressor", reg.dimension_curve_)):
        print(f"  {task_name}:")
        d_eff = igl.detect_elbow(curve)
        for k, v in curve.items():
            marker = " ← elbow" if k == d_eff else ""
            print(f"    k={k:2d}: {v:.5f}{marker}")

    run_dir = make_run_dir(EXAMPLE_NAME)
    save_curve(run_dir, ae.dimension_curve_, name="autoencoder_curve.csv")
    save_curve(run_dir, reg.dimension_curve_, name="regressor_curve.csv")

    # Optional plot via [viz] extra.
    try:
        from igl.viz import plot_dimension_curve  # noqa: PLC0415
    except igl.IGLDependencyError:
        print("\n(install [viz] extra for a PNG plot of the curves)")
    else:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig, ax = plt.subplots(figsize=(7, 4.5))
        plot_dimension_curve(
            ae.dimension_curve_,
            ax=ax,
            elbow=ae.effective_dimension_,
            title="Swiss roll dimension curves",
            label="autoencoder (recon)",
        )
        plot_dimension_curve(
            reg.dimension_curve_,
            ax=ax,
            elbow=reg.effective_dimension_,
            label="regressor",
        )
        fig.tight_layout()
        png_path = run_dir / "curves.png"
        fig.savefig(png_path)
        plt.close(fig)
        print(f"\nPlot written to {png_path}")

    print(f"\nResults directory: {run_dir}")
    print(f"git short sha: {git_short_sha()}")


if __name__ == "__main__":
    main()
