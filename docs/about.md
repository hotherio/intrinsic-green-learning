# About

Intrinsic Green Learning (IGL) is a Python library implementing the
**task-conditioned intrinsic-dimensionality discovery** framework
introduced in [*Quemy, A. (2026). Intrinsic Green's Learning: Supervised
Learning on Manifolds via Inverse PDE.* ICLR 2026 Workshop on AI and
PDE](https://openreview.net/forum?id=Y6RpdS98l8).

## Author

**Alexandre Quemy** ([`@aquemy`](https://github.com/aquemy),
[ORCID 0000-0002-5865-6403](https://orcid.org/0000-0002-5865-6403))
— research and implementation. The library is developed under
[Hother](https://hother.io) and released under the MIT license.

## Where IGL sits in the landscape

Intrinsic-dimensionality estimation has historically split into two
camps:

- **Purely geometric** estimators (Levina–Bickel, TwoNN, MLE-based
  approaches in [`scikit-dimension`](https://scikit-dimension.readthedocs.io/))
  look at the input data alone and ignore what the user actually
  wants to predict.
- **Implicit bottlenecks** in deep networks (autoencoders,
  information-bottleneck methods, variational approaches) *use* a
  low-dimensional latent but don't *report* its size without a
  separate analysis pass.

IGL takes a third position: solve the supervised task and read off
the effective dimension in the same training run. The discovered
`d_eff` is a property of *(input, task)*, not of the input alone — the
same dataset resolves into different effective dimensions for a
classifier, a regressor, and an autoencoder, and the hierarchy
$d_{\text{cls}} \le d_{\text{reg}} \le d_{\text{recon}}$ falls out
naturally.

## Theoretical underpinnings

The library realises the inverse-PDE formulation from the paper above:

- The latent-space operator $L$ is encoded by the **Green's-function
  kernel** $G(z, s)$, which can be expressed either as a local product
  of per-dimension kernels (the default `GreenKernel`) or via its
  spectral expansion $G(z, s) = \sum_n \phi_n(z)\,\phi_n(s) / \lambda_n$
  (the `SpectralKernel` plus closed-form Fourier / Chebyshev /
  Legendre / Hermite / Laguerre bases, or learned Laplace–Beltrami /
  user-supplied graph bases).
- The non-trivial **null space** of $L$ — the modes the Green's-
  function expansion cannot reach (e.g. the constant function under a
  Neumann Laplacian) — is fitted as un-regularised columns of the
  design matrix via the kernel-agnostic `NullSpaceBasis` protocol.
- **Variable Projection with random Matryoshka truncation** trains the
  encoder while sampling the latent-dimension cutoff $k$, so a single
  trained model exposes a quality-vs-`k` *dimension curve* that reads
  off `d_eff` at deployment time.

Concrete derivations and worked examples live under
[Concepts](concepts.md); the per-symbol API reference under
[Reference](reference/index.md).

## Status

IGL is **active research-grade software**. The API follows strict
typing discipline (`basedpyright --strict`, ≥99% test coverage with
property-based fuzzing on the numerical core), but minor versions may
introduce breaking changes as the framework evolves. The `0.x` line is
appropriate for research and experimentation; production users should
pin specific minor versions.

## Citing

See the
[Bibliography](https://github.com/hotherio/intrinsic-green-learning#bibliography)
section of the project README for the recommended citation, or the
[`CITATION.cff`](https://github.com/hotherio/intrinsic-green-learning/blob/main/CITATION.cff)
file for the structured form (GitHub's *"Cite this repository"*
widget reads it automatically).
