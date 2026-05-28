# Concepts

This page summarises the mathematical setup IGL is built around. For the
API itself, see [Quickstart](quickstart.md) and the
[API reference](reference/index.md).

## Encoder + Green-kernel design

Given ambient inputs $x \in \mathbb{R}^D$, IGL trains:

1. An **encoder** $\Psi_\theta : \mathbb{R}^D \to \mathbb{R}^{d_{\max}}$
   ([`igl.MLPEncoder`][igl.MLPEncoder] or your own
   [`EncoderProtocol`][igl.types.EncoderProtocol]).
2. A **multi-scale Green's-function kernel**
   ([`igl.GreenKernel`][igl.GreenKernel]) with $R$ learnable anchors
   $\mu_r \in \mathbb{R}^{d_{\max}}$ that turns the latent
   $z = \Psi_\theta(x)$ into a structured design matrix
   $\Phi \in \mathbb{R}^{N \times R}$.
3. A linear **readout** $\hat y = \Phi w + b$ whose weights $w$ are not
   learned by gradient descent — they are refreshed in closed form by
   [`igl.direct_solve_weights`][igl.direct_solve_weights] (Tikhonov-
   regularised lstsq).

The kernel is a product over the latent dimensions and a weighted sum
over $K$ scales:

$$
\Phi_{n,r} = \sigma(\rho_r) \sum_k \gamma_k \prod_j G_{\text{op}}\!\bigl(z_{n,j} - \mu_{r,j},\, \sigma_{k,j}\bigr)^{m_j}
$$

with $G_{\text{op}}$ one of nine operators registered in
`igl.kernels` (Gaussian, Laplacian, Cauchy, Helmholtz,
Gabor, Mexican-hat, etc.), $\gamma_k$ a softmax over scale mixing
weights, and $m_j \in \{0, 1\}$ a Matryoshka truncation mask.

## Variable Projection training

The readout $w$ is closed-form given the encoder and kernel state, so
gradient descent only operates on $\theta$ and the kernel parameters.
Per training step:

1. Encode and (optionally) truncate: $z = \Psi_\theta(x)$,
   $z_k = z \odot m$ where $m$ keeps the first $k$ dimensions.
2. Compute the design matrix $\Phi_k = \text{GreenKernel}(z_k)$.
3. Solve $w_k^* = \arg\min \|\Phi_k w - y\|^2 + \lambda \|w\|^2$ in closed form
   (no autograd through the solve).
4. Backpropagate the task loss
   $\mathcal{L}\bigl(\Phi_k w_k^*, y\bigr)$ through $\Phi_k$ to update
   $\theta$ and the kernel.

This separation — *Variable Projection* — gives the optimiser a
much better-conditioned landscape than joint gradient descent on
$(\theta, w)$ would.

## Matryoshka truncation and effective dimension

At every training step the trainer samples a truncation level
$k \sim \text{Uniform}\{1, \dots, d_{\max}\}$ and masks the latent
beyond $k$. Because $k$ varies across batches, the encoder is forced to
work at *every* truncation level — its first dimension carries the most
useful information, its second the next-most, and so on. The model is
*Matryoshka-nested*: smaller models are subsets of larger ones.

After training, [`igl.eval_dimension_curve`][igl.eval_dimension_curve]
sweeps $k = 1, 2, \dots, d_{\max}$ and reports the curve score (error
rate for classification, MSE for regression / reconstruction,
AIRM² for SPD reconstruction). [`igl.detect_elbow`][igl.detect_elbow]
locates the elbow on a log scale: the smallest $k$ beyond which adding
more dimensions stops giving substantial improvements. That $k$ is the
**effective dimension** $d_{\text{eff}}$.

## The task-conditioned hierarchy

The same input data carries different amounts of "useful structure" for
different tasks:

- A **classifier** typically needs one direction to separate the moons.
- A **regressor** predicting smooth manifold coordinates needs both
  intrinsic directions.
- A **reconstruction** model has to capture the full intrinsic
  geometry.

Empirically, on a wide range of synthetic manifolds and real datasets:

$$
d_{\text{eff}}(\text{cls}) \;\le\; d_{\text{eff}}(\text{reg}) \;\le\; d_{\text{eff}}(\text{recon}).
$$

The library makes this measurable: [`igl.compare_d_eff`][igl.compare_d_eff]
takes any number of dimension curves (keyed by task name) and returns a
[`DimensionComparison`][igl.DimensionComparison] with the per-task
effective dimensions and a `hierarchy_holds` flag that's `True` iff the
values appear in non-decreasing order. The bundled
`examples/synthetic/moons_xor.py` demonstrates `True` out of the box.

## The SPD extension

For symmetric positive-definite (SPD) data — covariance matrices, EEG
epochs, clinical correlations — Euclidean MSE is not the right loss
because the SPD cone is a Riemannian manifold, not a flat vector space.
`igl.spd` ships:

- [`LogEigVectorizer`][igl.spd.LogEigVectorizer] — maps each SPD matrix
  $C$ to a flat vector $\operatorname{vec}(\log C)$ in the log-Euclidean
  tangent space at the identity.
- [`AIRMLoss`][igl.spd.AIRMLoss] — implements
  [`LossStrategy`][igl.types.LossStrategy] using the affine-invariant
  Riemannian metric

    $$\text{AIRM}(C, \hat C)^2 = \bigl\lVert \log\bigl(C^{-1/2}\, \hat C\, C^{-1/2}\bigr)\bigr\rVert_F^2.$$

  The trainer's lstsq still operates in log-Euclidean tangent space (a
  flat vector space where Euclidean lstsq is geometry-respecting), but
  the gradient signal is shaped by the manifold distance.
- [`OrthogonalityPenalty`][igl.spd.OrthogonalityPenalty] — an
  [`ExtraLoss`][igl.types.ExtraLoss] that drives the *pullback metric*
  $g = J J^\top$ (where $J = \partial \Psi / \partial x$) toward
  diagonality. When $g$ is diagonal at every $x$, the latent
  coordinates are first-order orthogonal — the Stäckel condition — and
  geodesics separate cleanly per coordinate.
- [`IGLReconSPDClassifier`][igl.spd.IGLReconSPDClassifier] — a
  two-stage classifier: stage A trains the encoder to reconstruct
  log-Eig vectors via AIRM, stage B fits a `LogisticRegression` on the
  frozen Green-kernel design matrix.

## Loss strategies and the `ExtraLoss` seam

The trainer is task-agnostic: every task plugs in via a
[`LossStrategy`][igl.types.LossStrategy] (provides `target`, `loss`,
`metric`, `curve_score`, and a `higher_is_better` flag) and zero or more
[`ExtraLoss`][igl.types.ExtraLoss] regularizers (called per batch,
multiplied by `weight`, added to the task loss before backprop).

Adding a new task — say, contrastive learning on a metric-space output
— is one new `LossStrategy`; no trainer changes. Adding a new
regularizer — say, a sparsity penalty on the latents — is one new
`ExtraLoss`. The reference implementation contains gate-sparsity,
Stäckel-pullback, and AIRM losses all sharing these two seams.

## What lives where

| Subpackage | Role |
|---|---|
| `igl` | Top-level surface: configs, building blocks, sklearn models, metrics. |
| `igl.core` | Geometry-agnostic primitives: encoder, kernel, solver, trainer. |
| `igl.kernels` | Operator zoo: Gaussian, Laplacian, Cauchy, Helmholtz, Gabor, Mexican-hat, Yukawa, multiquadric, soft-box. Registration via `register_operator`. |
| `igl.matryoshka` | Truncation samplers (uniform, power-law) and the post-fit dimension curve. |
| `igl.metrics` | `d_eff_from_curve`, `compare_d_eff`, elbow detectors. |
| `igl.models` | sklearn estimators: classifier, regressor, autoencoder. |
| `igl.nn` | Bare [`IGLModule`][igl.IGLModule] for custom training loops. |
| `igl.spd` | Riemannian extension: AIRM, log-Eig, orthogonality, reconstruction. |
| `igl.data` | Synthetic generators: torus, moons, swiss roll, SPD dataset. |
| `igl.viz` | Optional matplotlib helpers (gated behind the `[viz]` extra). |
