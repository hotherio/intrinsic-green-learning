"""1-D Poisson with closed-form vs spectral IGL.

Solve ``-u'' = f`` on ``[0, 1]`` with Neumann boundary conditions and a
non-zero-mean source. The Neumann Laplacian has constants as its kernel
— so the target ``u`` cannot be uniquely determined without an explicit
null-space augmentation.

We solve the same problem three ways:

1. **Closed-form**: integrate ``f`` analytically.
2. **Local IGL** (``GreenKernel`` with no null-space augmentation):
   the Tikhonov shrinkage pulls the DC component toward zero.
3. **Spectral IGL** (``SpectralKernel`` with ``FourierCosineBasis`` +
   ``ConstantNullSpace``): the constant mode is fitted explicitly via
   the un-regularised null-space column.

Run with::

    python -m examples.synthetic.poisson_1d
"""

import math
import warnings

import torch

import igl
from examples._utils import git_short_sha, make_run_dir, save_curve, set_seed
from igl.spectral import (
    ConstantNullSpace,
    FourierCosineBasis,
    SpectralKernel,
)

EXAMPLE_NAME = "poisson_1d"


def _make_problem() -> tuple[torch.Tensor, torch.Tensor]:
    """Build a 1-D Poisson problem.

    ``f(x) = A + B · cos(π x)`` has closed-form solution
    ``u(x) = A x²/2 + (B/π²) cos(π x) + C`` for Neumann BCs.
    """
    set_seed(0)
    n_samples = 400
    x = torch.linspace(0.0, 1.0, n_samples).unsqueeze(-1)
    a_const = 0.4
    b = 1.0
    u = 0.5 * a_const * x**2 + (b / math.pi**2) * torch.cos(math.pi * x)
    return x, u


def _train(
    module: igl.IGLModule,
    x: torch.Tensor,
    u: torch.Tensor,
    *,
    epochs: int = 600,
) -> float:
    trainer = igl.MatryoshkaTrainer(
        loss=igl.MSELoss(),
        config=igl.MatryoshkaConfig(
            epochs=epochs,
            batch_size=64,
            inner_batch_size=400,
            scheduler="cosine_warm_restarts",
            early_stop_patience=None,
            verbose=False,
        ),
    )
    trainer.fit(module, x, u)
    return float(torch.nn.functional.mse_loss(module(x).detach(), u).item())


def main() -> None:
    x, u = _make_problem()

    print("=" * 60)
    print("1-D Poisson with Neumann BCs (DC mode test)")
    print("=" * 60)
    print(f"Target shape: {tuple(u.shape)}; target mean: {u.mean().item():.4f}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # Local GreenKernel — no null space.
        print("\nLocal GreenKernel (no null space)…")
        gk_local = igl.GreenKernel(latent_dim=1, n_anchors=16, n_scales=3)
        mod_local = igl.IGLModule(
            input_dim=1,
            max_dim=1,
            output_dim=1,
            kernel=gk_local,
            normalize_input=False,
        )
        mse_local = _train(mod_local, x, u)
        pred_local = mod_local(x).detach()
        bias_local = (pred_local - u).mean().item()
        print(f"  MSE = {mse_local:.5f}")
        print(f"  mean bias = {bias_local:.5f}")

        # Local GreenKernel + ConstantNullSpace.
        print("\nLocal GreenKernel + ConstantNullSpace…")
        gk_null = igl.GreenKernel(
            latent_dim=1,
            n_anchors=16,
            n_scales=3,
            null_space=ConstantNullSpace(),
        )
        mod_null = igl.IGLModule(
            input_dim=1,
            max_dim=1,
            output_dim=1,
            kernel=gk_null,
            normalize_input=False,
        )
        mse_null = _train(mod_null, x, u)
        pred_null = mod_null(x).detach()
        bias_null = (pred_null - u).mean().item()
        print(f"  MSE = {mse_null:.5f}")
        print(f"  mean bias = {bias_null:.5f}")

        # SpectralKernel with FourierCosineBasis (Neumann LP).
        print("\nSpectralKernel(FourierCosineBasis) + ConstantNullSpace…")
        spk = SpectralKernel(
            latent_dim=1,
            bases=FourierCosineBasis(n_modes=16),
            n_anchors=12,
            null_space=ConstantNullSpace(),
        )
        mod_spec = igl.IGLModule(
            input_dim=1,
            max_dim=1,
            output_dim=1,
            kernel=spk,
            normalize_input=False,
        )
        mse_spec = _train(mod_spec, x, u)
        pred_spec = mod_spec(x).detach()
        bias_spec = (pred_spec - u).mean().item()
        print(f"  MSE = {mse_spec:.5f}")
        print(f"  mean bias = {bias_spec:.5f}")

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"local (no null)  : MSE={mse_local:.5f}, mean-bias={bias_local:.5f}")
    print(f"local + null     : MSE={mse_null:.5f}, mean-bias={bias_null:.5f}")
    print(f"spectral + null  : MSE={mse_spec:.5f}, mean-bias={bias_spec:.5f}")

    run_dir = make_run_dir(EXAMPLE_NAME)
    save_curve(
        run_dir,
        {1: mse_local, 2: mse_null, 3: mse_spec},
        name="mse_by_path.csv",
    )
    print(f"\nResults written to: {run_dir}")
    print(f"git short sha: {git_short_sha()}")


if __name__ == "__main__":
    main()
