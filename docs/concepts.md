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

## Spectral formulation and the null space

The local `GreenKernel` is a *product* of fixed-shape 1-D kernels at
learnable anchors. The **spectral formulation** replaces this with the
eigendecomposition of an operator $L$:

$$
G(z, s) = \sum_n \frac{\phi_n(z)\,\phi_n(s)}{\max(\lambda_n, \varepsilon)}.
$$

The library ships eight bases in
[`igl.spectral`][igl.spectral.kernel.SpectralKernel]:

| Basis | Domain | Notes |
|---|---|---|
| [`FourierSineBasis`][igl.spectral.bases.fourier_sine.FourierSineBasis] | $[0, 1]$ | Dirichlet BCs, no null mode. |
| [`FourierCosineBasis`][igl.spectral.bases.fourier_cosine.FourierCosineBasis] | $[0, 1]$ | Neumann BCs, $\phi_0 = 1$ is the null mode. |
| [`ChebyshevBasis`][igl.spectral.bases.chebyshev.ChebyshevBasis] | $[-1, 1]$ | Polynomial spectral-element basis. |
| [`LegendreBasis`][igl.spectral.bases.legendre.LegendreBasis] | $[-1, 1]$ | Orthogonality w.r.t. uniform weight. |
| [`HermiteBasis`][igl.spectral.bases.hermite.HermiteBasis] | $\mathbb{R}$ | Gaussian-weighted. |
| [`LaguerreBasis`][igl.spectral.bases.laguerre.LaguerreBasis] | $[0, \infty)$ | Exponentially-weighted. |
| [`LearnedLaplacianBasis`][igl.spectral.bases.learned_lb.LearnedLaplacianBasis] | learned manifold | $k$-NN graph + sparse eigsh + Nyström extension. |
| [`GraphLaplacianBasis`][igl.spectral.bases.graph_laplacian.GraphLaplacianBasis] | user-supplied graph | Symmetric / random-walk / unnormalized. |

For multi-dimensional latents,
[`SpectralKernel`][igl.spectral.kernel.SpectralKernel] takes either one
basis (uniform across dims) or a sequence (per-dim). For mixtures on
the same dimension,
[`MultiSpectralBasis`][igl.spectral.multi.MultiSpectralBasis] wraps
$K$ bases with a softmax-mixed weighting.

### Null-space augmentation

Operators with non-trivial kernels — e.g. the Neumann Laplacian, whose
$\phi_0 = 1$ has $\lambda_0 = 0$ — cannot reach those modes via the
Green's expansion. The library exposes
`NullSpaceBasis` as a kernel-agnostic
add-on: extra design-matrix columns that the lstsq solve fits *without*
Tikhonov shrinkage, so the null component comes from the data.

Three concrete bases:

- [`ConstantNullSpace`][igl.spectral.null_space.ConstantNullSpace] —
  one column of ones (the DC mode).
- [`PolynomialNullSpace`][igl.spectral.null_space.PolynomialNullSpace] —
  constant + per-dimension monomials up to a given degree.
- [`CustomNullSpace`][igl.spectral.null_space.CustomNullSpace] — wraps
  an arbitrary callable.

Both the local [`GreenKernel`][igl.core.kernel.GreenKernel] and the
[`SpectralKernel`][igl.spectral.kernel.SpectralKernel] accept a
`null_space=` argument; the lstsq target column count and the
`source_weights` buffer width adjust automatically.

### Learned Laplace–Beltrami spectrum

For data on an unknown manifold, the operator to invert is the
Laplace–Beltrami operator of the *learned* metric $g = J^\top J$ where
$J$ is the encoder's Jacobian.
[`LearnedLaplacianBasis`][igl.spectral.bases.learned_lb.LearnedLaplacianBasis]
estimates the spectrum numerically:

1. Build a $k$-NN graph on the encoded latents.
2. Symmetric normalised Laplacian
   $L = I - D^{-1/2} W D^{-1/2}$ with Gaussian edge weights.
3. Sparse eigendecomposition via `scipy.sparse.linalg.eigsh`.
4. Nyström extension to evaluate the eigenfunctions on new points.

Because the metric drifts during training, the basis must be
refreshed periodically. The
[`LearnedLBRefresh`][igl.spectral.refresh.LearnedLBRefresh] hook plugs
into the trainer via the `extra_losses=` parameter and re-runs the
eigendecomposition every $N$ batches with the current encoder.

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
| `igl.spectral` | Spectral kernels + null-space augmentation: closed-form bases, learned LB, graph Laplacian. |
| `igl.data` | Synthetic generators: torus, moons, swiss roll, SPD dataset. |
| `igl.viz` | Optional matplotlib helpers (gated behind the `[viz]` extra). |
