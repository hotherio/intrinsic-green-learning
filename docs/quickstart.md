# Quickstart

This page walks through fitting an IGL estimator and reading off the
effective dimension it discovered. Every example below uses two
attributes that every fitted IGL estimator carries:

- **`effective_dimension_`** — an integer: the smallest latent dimension
  the trained model needs to keep solving the task well.
- **`dimension_curve_`** — a `{k: score}` mapping showing how
  quality changes as you truncate the latent at $k = 1, 2, \dots,
  d_{\max}$. The curve is what `effective_dimension_` is read off from.

Install:

```bash
pip install intrinsic-green-learning
```

Optional extras:

```bash
pip install "intrinsic-green-learning[viz]"   # matplotlib plotting helpers
pip install "intrinsic-green-learning[eeg]"   # mne + moabb + pyriemann (v0.2)
pip install "intrinsic-green-learning[nlp]"   # transformers + datasets (v0.2)
pip install "intrinsic-green-learning[elbow]" # alternative kneedle detector
pip install "intrinsic-green-learning[all]"   # everything above
```

## A first classifier

The simplest IGL workflow is the sklearn-compatible
[`IGLClassifier`][igl.IGLClassifier]: train it on numpy arrays, predict,
and inspect the discovered effective dimension.

```python
import numpy as np
import igl
from igl.data import embed_in_high_dim, make_moons

# Build a 16-D embedding of the two-moons manifold (intrinsic dim = 1).
x_2d, y = make_moons(400, noise=0.1, seed=0)
x = embed_in_high_dim(x_2d, target_dim=16, seed=0).numpy()
y_np = y.numpy()

clf = igl.IGLClassifier(max_dim=8, random_state=0).fit(x, y_np)

print(f"accuracy        = {clf.score(x, y_np):.3f}")
print(f"discovered d_eff = {clf.effective_dimension_}")  # expect 1
```

Every fitted IGL estimator carries:

- `module_` — the underlying [`IGLModule`][igl.IGLModule].
- `history_` — per-epoch [`TrainingHistory`][igl.TrainingHistory].
- `dimension_curve_` — `{k: curve_score}` from
  [`igl.eval_dimension_curve`][igl.eval_dimension_curve].
- `effective_dimension_` — the discovered `d_eff` (via
  [`igl.detect_elbow`][igl.detect_elbow]).

## Comparing tasks on the same data

The same dataset will resolve into different effective dimensions for
classification, regression, and reconstruction — see
[Concepts: the task-conditioned hierarchy](concepts.md#the-task-conditioned-hierarchy)
for why. [`igl.compare_d_eff`][igl.compare_d_eff] checks the ordering
across any number of tasks:

```python
from igl.data import make_swiss_roll

x, params = make_swiss_roll(800, seed=0)
x_np = x.numpy(); params_np = params.numpy()

reg = igl.IGLRegressor(max_dim=8, random_state=0).fit(x_np, params_np)
ae  = igl.IGLAutoencoder(max_dim=8, random_state=0).fit(x_np)

report = igl.compare_d_eff(
    cls=clf.dimension_curve_,
    reg=reg.dimension_curve_,
    recon=ae.dimension_curve_,
)
print(report.d_effs)           # e.g. {'cls': 1, 'reg': 2, 'recon': 2}
print(report.hierarchy_holds)  # True
```

## Configuring the encoder

The encoder shape — depth, hidden width(s), norm, activation — is
controlled via [`igl.EncoderConfig`][igl.EncoderConfig] (or the
shorthand `encoder_hidden` / `encoder_depth` kwargs on the estimators):

```python
clf = igl.IGLClassifier(
    max_dim=8,
    encoder_hidden=(128, 64),     # pyramidal encoder, 2 hidden layers
    encoder_depth=2,
    config=igl.IGLConfig(
        encoder=igl.EncoderConfig(norm="layer", activation="silu"),
        kernel=igl.KernelConfig(n_anchors=64, operator="gaussian"),
        matryoshka=igl.MatryoshkaConfig(epochs=1500, source_l2=1e-3),
    ),
).fit(x, y_np)
```

All string-valued fields also accept their `enum.StrEnum` member form:
`igl.NormType.LAYER`, `igl.ActivationType.SILU`,
`igl.OperatorName.GAUSSIAN`, etc.

## SPD / covariance-valued data

Some inputs are not vectors but **symmetric positive-definite matrices**:
EEG channel covariances, fMRI connectivity matrices, financial
correlations. These live on a Riemannian manifold — the SPD cone — and
ordinary Euclidean losses misrepresent distances between them. The
`igl.spd` subpackage ships an AIRM-loss-based classifier that respects
that geometry, plus a vectoriser that maps SPD matrices into the
log-Euclidean tangent space where the kernel solver can operate
flatly:

```python
from igl.data import make_spd_dataset
from igl.spd import IGLReconSPDClassifier, LogEigVectorizer

spd, y = make_spd_dataset(400, d=8, n_classes=3, seed=0)
x = LogEigVectorizer().fit(spd.numpy()).transform(spd.numpy())

clf = IGLReconSPDClassifier(
    latent_dim=8,
    max_dim=12,
    orthogonality_weight=0.1,
    random_state=0,
).fit(x, y.numpy())

print(clf.effective_dimension_)
```

The `orthogonality_weight` argument plugs an
[`OrthogonalityPenalty`][igl.spd.OrthogonalityPenalty] into the stage-A
training loop via the trainer's
[`ExtraLoss`][igl.types.ExtraLoss] seam.

### Recommended EEG path: `make_igl_airm`

When the input is raw signal `[N, channels, time]` rather than
SPDs already, the [`make_igl_airm`][igl.make_igl_airm] factory
composes the recommended defaults from the maintainer memo into a
single sklearn pipeline:

```python
import igl  # requires: pip install intrinsic-green-learning[eeg]

pipe = igl.make_igl_airm(latent_dim=22)
pipe.fit(X_raw, y)
```

`make_igl_airm` chains:

1. [`AutoCovariances`][igl.preprocessing.AutoCovariances] — Ledoit-Wolf
   for trial length `T < 500`, sample covariance otherwise. Threshold
   picked at fit time and frozen.
2. [`IGLReconSPDClassifier`][igl.spd.IGLReconSPDClassifier] with
   Tikhonov ε = 10⁻⁶ preconditioning. Bit-near-identical to no
   preconditioning at `d ≤ 64` (with a BatchNorm encoder) and
   rescues `torch.linalg.eigh` from LAPACK error 8481 at `d ≥ 128`.

Override either default through `**kwargs` — e.g. `precondition="none"`
disables preconditioning, `precondition="tikhonov+trace"` adds trace
normalisation on top. See the
[`PreconditionMode`][igl.PreconditionMode] enum for the full set.

## Custom training loops

If you need finer control than the sklearn surface, compose the
building blocks directly:

```python
import torch
import igl

module = igl.IGLModule(
    input_dim=16, max_dim=8, output_dim=2,
    config=igl.IGLConfig(
        encoder=igl.EncoderConfig(hidden=(128, 64)),
        kernel=igl.KernelConfig(n_anchors=64),
    ),
)

trainer = igl.MatryoshkaTrainer(
    loss=igl.CrossEntropyLoss(n_classes=2),
    config=igl.MatryoshkaConfig(epochs=500),
)

x_t = torch.from_numpy(x).float()
y_t = torch.from_numpy(y_np).long()
history = trainer.fit(module, x_t, y_t)

curve = igl.eval_dimension_curve(
    module, x_t, y_t, loss=igl.CrossEntropyLoss(n_classes=2),
)
print("d_eff =", igl.detect_elbow(curve))
```

You can drop in a custom [`LossStrategy`][igl.types.LossStrategy] (e.g.
[`igl.spd.AIRMLoss`][igl.spd.AIRMLoss]) and one or more
[`ExtraLoss`][igl.types.ExtraLoss] regularizers via
`trainer.fit(..., extra_losses=[...])`.

## Plotting

With the `[viz]` extra installed:

```python
from igl.viz import plot_dimension_curve

ax = plot_dimension_curve(
    clf.dimension_curve_,
    elbow=clf.effective_dimension_,
    title="moons → classifier",
)
ax.figure.savefig("curve.png")
```

## Next steps

- Read the [Concepts](concepts.md) page for the math behind the library.
- Browse the [API reference](reference/index.md) for every public symbol.
- Look at `examples/synthetic/*.py` in the repository for end-to-end
  scripts that exercise the canonical hierarchy.
