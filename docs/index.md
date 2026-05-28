# Intrinsic Green Learning

**Task-conditioned intrinsic-dimensionality discovery** for high-dimensional
data, via a learned encoder, a multi-scale Green's-function kernel, and
Variable Projection / Matryoshka training.

## What it does

Given high-dimensional inputs $x \in \mathbb{R}^D$ and a task — classification,
regression, or reconstruction — IGL learns:

1. **An encoder** $\Psi_\theta : \mathbb{R}^D \to \mathbb{R}^{d_{\max}}$ that
   maps inputs onto an ambient latent space.
2. **A structured design matrix** $\Phi \in \mathbb{R}^{N \times R}$ built
   from a multi-scale product Green's kernel with $R$ learnable anchors.
3. **The effective dimension** $d_{\text{eff}}$ at which the task is solved,
   discovered automatically via random Matryoshka truncation during training.

The library validates the empirical hierarchy

$$
d_{\text{eff}}(\text{classification}) \;\le\; d_{\text{eff}}(\text{regression}) \;\le\; d_{\text{eff}}(\text{reconstruction})
$$

on the same underlying manifold — and lets you measure it on your own data.

## Status

This is **0.0.0** scaffolding. The public API is unstable until 0.1.0; see
[`docs/about.md`](about.md) for the roadmap and
[the guidelines](guidelines/index.md) for development practices.
