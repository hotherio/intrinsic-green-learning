# Intrinsic Green Learning

**Task-conditioned intrinsic-dimensionality discovery** for high-dimensional
data. IGL pairs a learned encoder with a multi-scale Green's-function
kernel and trains the system end-to-end via Variable Projection with random
Matryoshka truncation.

The model fits the task and simultaneously reveals how many dimensions the
task actually needs — usually far fewer than the ambient input.

## Why IGL?

For the same input data, a classifier usually needs fewer latent
dimensions than a regressor, which in turn needs fewer than a full
autoencoder. IGL discovers this hierarchy automatically:

$$
d_{\text{eff}}(\text{classification}) \;\le\; d_{\text{eff}}(\text{regression}) \;\le\; d_{\text{eff}}(\text{reconstruction})
$$

The library makes this measurable: every fitted estimator carries a
`dimension_curve_` attribute and an `effective_dimension_` integer, and
[`igl.compare_d_eff`][igl.compare_d_eff] verifies the ordering across
tasks.

## Getting started

- **[Quickstart](quickstart.md)** — install, import, fit your first model.
- **[Concepts](concepts.md)** — the math behind the library: encoder,
  Green kernel, Matryoshka truncation, effective dimension.
- **[API reference](reference/index.md)** — every public symbol.
- **[Guidelines](guidelines/index.md)** — development conventions.
- **[Security](security.md)** — supply-chain posture (OIDC, sigstore
  attestations, signed checksums).

## Status

Pre-release. The public API is stable from `0.1.0` onward; everything in
`igl.contrib` (when populated) carries a weaker stability promise. v0.1
ships the four sklearn estimators ([`IGLClassifier`][igl.IGLClassifier],
[`IGLRegressor`][igl.IGLRegressor], [`IGLAutoencoder`][igl.IGLAutoencoder],
[`IGLReconSPDClassifier`][igl.spd.IGLReconSPDClassifier]), the bare
[`IGLModule`][igl.IGLModule] for custom training loops, and the SPD
extension (`igl.spd`).
