"""Flat-torus example: discover the intrinsic dimension of ``T²``.

The flat torus ``T² ⊂ R⁴`` has intrinsic dimension 2. We embed it in
``R^32`` via a random orthogonal rotation, train two IGL models on the same
underlying data — one classification head (XOR-of-quadrants) and one
regression head (``sin/cos`` of both angles) — and report the dimension
curve for each. The regression curve typically gives a clean elbow at
``k ≈ 2``; the classification cross-entropy saturates quickly and the elbow
is less informative on this task (a known limitation addressed in M3 with an
error-rate curve metric).

Run with::

    python -m examples.synthetic.torus_classification

Outputs land in ``results/torus_classification/<git_sha>/``.
"""

import torch

import igl
from examples._utils import git_short_sha, make_run_dir, save_curve, save_history, set_seed
from igl.data import embed_in_high_dim, make_flat_torus, make_flat_torus_labels

EXAMPLE_NAME = "torus_classification"


def _run_classification(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    ambient_dim: int,
) -> None:
    print("=" * 60)
    print("Part 1 — Classification (XOR of quadrants)")
    print("=" * 60)

    module = igl.IGLModule(
        input_dim=ambient_dim,
        max_dim=12,
        output_dim=2,
        n_anchors=48,
        n_scales=4,
        operator="gaussian",
    )
    trainer = igl.MatryoshkaTrainer(
        loss=igl.CrossEntropyLoss(n_classes=2),
        config=igl.MatryoshkaConfig(
            epochs=400,
            batch_size=128,
            inner_batch_size=1500,
            encoder_lr=1e-3,
            scheduler="cosine_warm_restarts",
            early_stop_patience=None,
            verbose=False,
        ),
    )
    history = trainer.fit(module, x_train, y_train, x_val=x_val, y_val=y_val)
    curve = igl.eval_dimension_curve(module, x_val, y_val, loss=igl.CrossEntropyLoss(n_classes=2))
    d_eff = igl.detect_elbow(curve)
    print(f"  final val_acc = {history.val_metric[-1]:.4f}")
    print(f"  cross-entropy curve d_eff = {d_eff} (note: CE saturates on easy tasks)")


def _run_regression(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    ambient_dim: int,
) -> tuple[igl.TrainingHistory, dict[int, float], int]:
    print()
    print("=" * 60)
    print("Part 2 — Regression (sin/cos of both angles)")
    print("=" * 60)

    # Showcase the config-driven construction path: shape the encoder (here a
    # pyramidal 192 → 96 MLP) and kernel through IGLConfig, no manual
    # MLPEncoder instantiation required.
    config = igl.IGLConfig(
        max_dim=12,
        encoder=igl.EncoderConfig(hidden=(192, 96), depth=2, norm=igl.NormType.LAYER),
        kernel=igl.KernelConfig(n_anchors=48, n_scales=4, operator=igl.OperatorName.GAUSSIAN),
        matryoshka=igl.MatryoshkaConfig(
            epochs=400,
            batch_size=128,
            inner_batch_size=1500,
            encoder_lr=1e-3,
            scheduler=igl.SchedulerType.COSINE_WARM_RESTARTS,
            early_stop_patience=None,
            verbose=False,
        ),
    )
    module = igl.IGLModule(input_dim=ambient_dim, max_dim=12, output_dim=4, config=config)
    trainer = igl.MatryoshkaTrainer(loss=igl.MSELoss(), config=config.matryoshka)
    history = trainer.fit(module, x_train, y_train, x_val=x_val, y_val=y_val)
    curve = dict(igl.eval_dimension_curve(module, x_val, y_val, loss=igl.MSELoss()))
    d_eff = igl.detect_elbow(curve)
    print(f"  final val_mse = {history.val_metric[-1]:.5f}")
    print()
    print("  Dimension curve (lower-is-better):")
    for k, v in curve.items():
        marker = " ← elbow" if k == d_eff else ""
        print(f"    k={k:2d}: mse={v:.6f}{marker}")
    print(f"\n  Discovered d_eff = {d_eff} (intrinsic torus dim = 2)")
    return history, curve, d_eff


def main() -> None:
    set_seed(42)

    n_train, n_val = 1500, 500
    ambient_dim = 32

    x_4d, theta = make_flat_torus(n_train + n_val, seed=42)
    y_class = make_flat_torus_labels(theta, task="xor")
    y_reg = make_flat_torus_labels(theta, task="regression_smooth")
    x = embed_in_high_dim(x_4d, target_dim=ambient_dim, seed=123)

    x_train, x_val = x[:n_train], x[n_train:]
    y_class_train, y_class_val = y_class[:n_train], y_class[n_train:]
    y_reg_train, y_reg_val = y_reg[:n_train], y_reg[n_train:]
    print(f"Data: {tuple(x.shape)} ambient → 2-class XOR + 4-D regression labels")

    _run_classification(x_train, y_class_train, x_val, y_class_val, ambient_dim=ambient_dim)
    history, curve, d_eff = _run_regression(x_train, y_reg_train, x_val, y_reg_val, ambient_dim=ambient_dim)

    run_dir = make_run_dir(EXAMPLE_NAME)
    save_history(run_dir, history, name="regression_history.json")
    save_curve(run_dir, curve, name="regression_curve.csv")
    print(f"\nResults written to: {run_dir}")
    print(f"git short sha: {git_short_sha()}")
    print(f"summary: regression d_eff = {d_eff}")


if __name__ == "__main__":
    main()
