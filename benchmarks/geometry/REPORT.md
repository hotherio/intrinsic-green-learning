# Geometry suite: results

Every number here is reproducible from this repository (`benchmarks/geometry/`, `make geometry-tier0`,
label `tier0`). Two kinds of results, never mixed: **analytic / grid** (no training; the geometry is
known) and **learned chart** (IGL trained). The deck's own numbers (two 2-D hills at full height:
ρ 0.507 / 0.739, Born depth 8 / 18, naive error 33 / 49 %, source ranks 4 / 9 in the data chart and
3 / 5 in the conformal chart, 5 / 7 in the orthogonal non-conformal chart) are reproduced by
`testbeds/hills.py`; everything else is new.

## Tier 0: analytic and grid results

### R0. The rank story, chart by chart (go / no-go)

Kind: analytic. Targets: the deck's Poisson problem on the hills; grid Poisson solves with a rank-one
heater on the other beds. "Atoms" is the number of Gaussian atoms on a fixed grid of centres and widths
that a sparse fit needs to reach the stated relative error, after rescaling each chart axis by a
constant (what the kernel's per-axis widths absorb, and nothing more). `rank u` is the separable rank
of the target resampled on a grid of the chart.

| bed | chart | rank f̂ (1 %) | rank u (1 %) | atoms 5 % | atoms 2 % | atoms 1 % | oracle √L_conf |
|---|---|---|---|---|---|---|---|
| S2xR2 | oracle-conf | – | 71 | – | – | – | 0.048 |
| S2xR2 | param | – | 77 | – | – | – | 0.048 |
| SPD2-airm | oracle-conf | – | 15 | 58 | 172 | – | 0.00719 |
| SPD2-airm | param | – | 16 | 64 | – | – | 0.00719 |
| graph-generic-d3 | oracle-conf | – | 15 | 148 | – | – | 0.099 |
| graph-generic-d3 | param | – | 16 | 117 | – | – | 0.099 |
| hill-lop | conf-pq | 5 | 3 | 32 | 90 | 202 | – |
| hill-lop | orth-pQ | 7 | 3 | 44 | 115 | 238 | – |
| hill-lop | xy | 9 | 4 | 30 | 91 | 186 | – |
| hill-round | conf-pq | 3 | 3 | 30 | 85 | 165 | – |
| hill-round | orth-pQ | 5 | 2 | 49 | 147 | – | – |
| hill-round | xy | 4 | 3 | 42 | 118 | 217 | – |

What it says:

- **The chart does change what the target costs the kernel, but less than the source rank suggests.**
  On the round hill the conformal chart needs 1.4× fewer atoms than the data chart at 2 % (85 against
  118), while the source rank falls from 4 to 3. On the lopsided hill the source rank falls from 9 to 5
  and the atom count does not move (90 against 91 at 2 %). On SPD(2) under the affine-invariant metric
  the oracle conformal chart reaches 2 % with 172 atoms where the log-Eig chart does not reach it with
  256.
- **The orthogonal non-conformal chart is worse than both on every count** (147 against 85 and 118
  atoms at 2 % on the round hill; 115 against 90 and 91 on the lopsided one). The deck's claim that
  today's penalty accepts charts that cost more holds for the kernel, not only for the source.
- **`rank(u)` does not track `rank(f̂)` across charts** (round hill: 3 / 2 / 3 against 4 / 5 / 3). The
  rank bound on the compensated source is a statement about the source; the kernel's cost follows the
  target. The deck keeps the bound as what it is and stops using it as a proxy for anchors.
- The decision rule of the plan (at least 1.3× fewer atoms on the hill) is met on the round hill and
  not on the lopsided one: proposal A can show through the kernel, bed by bed, and the learned-chart
  blocks decide. An earlier version of this measure rescaled each axis by its ranks, which made the
  orthogonal and conformal charts coincide; that was more freedom than the kernel has and was removed.
- Limits: the dictionary is isotropic in the rescaled chart while IGL's kernel has per-axis widths;
  in `d ≥ 3` the grid resampling of `rank u` from a few thousand points is noisy, and on `S²×R²` no
  chart reaches 5 % with 256 atoms, so that row says nothing yet.

### G1. The Born contraction across dimensions

Kind: analytic. Part A: hills and graph hypersurfaces at five heights; `ρ` by power iteration, the
proxy `max |A − I|`, the measured Born depth to `1e-3` against `log(1/ε)/log(1/ρ)`, and the error of
the flat solve that ignores the hill. Part B: random charts of the flat torus with a fixed per-entry
distortion of three kinds.

| bed | height s | ρ | max|A−I| | ρ / max|A−I| | depth 1e-3 | predicted | naive error |
|---|---|---|---|---|---|---|---|
| hill-lop | 0.2 | 0.13 | 0.131 | 0.998 | 2 | 3.392 | 0.047 |
| hill-lop | 0.4 | 0.361 | 0.358 | 1.007 | 4 | 6.774 | 0.157 |
| hill-lop | 0.6 | 0.54 | 0.54 | 1 | 8 | 11.222 | 0.281 |
| hill-lop | 0.8 | 0.66 | 0.663 | 0.995 | 12 | 16.602 | 0.395 |
| hill-lop | 1 | 0.739 | 0.748 | 0.988 | 18 | 22.844 | 0.491 |
| hill-round | 0.2 | 0.04 | 0.041 | 0.961 | 1 | 2.14 | 0.021 |
| hill-round | 0.4 | 0.142 | 0.147 | 0.963 | 2 | 3.533 | 0.078 |
| hill-round | 0.6 | 0.27 | 0.279 | 0.967 | 4 | 5.275 | 0.158 |
| hill-round | 0.8 | 0.396 | 0.408 | 0.971 | 6 | 7.459 | 0.247 |
| hill-round | 1 | 0.507 | 0.518 | 0.977 | 8 | 10.157 | 0.335 |
| graph-generic-d3 | 0.2 | 0.106 | 0.13 | 0.816 | 2 | 3.083 | 0.033 |
| graph-generic-d3 | 0.4 | 0.304 | 0.361 | 0.843 | 4 | 5.801 | 0.116 |
| graph-generic-d3 | 0.6 | 0.47 | 0.536 | 0.878 | 6 | 9.153 | 0.218 |
| graph-generic-d3 | 0.8 | 0.588 | 0.659 | 0.891 | 9 | 12.995 | 0.32 |
| graph-generic-d3 | 1 | 0.671 | 0.745 | 0.901 | 13 | 17.304 | 0.411 |
| graph-radial-d3 | 0.2 | 0.038 | 0.041 | 0.925 | 1 | 2.112 | 0.019 |
| graph-radial-d3 | 0.4 | 0.136 | 0.146 | 0.927 | 2 | 3.456 | 0.07 |
| graph-radial-d3 | 0.6 | 0.258 | 0.278 | 0.929 | 4 | 5.103 | 0.143 |
| graph-radial-d3 | 0.8 | 0.378 | 0.407 | 0.929 | 6 | 7.097 | 0.227 |
| graph-radial-d3 | 1 | 0.479 | 0.517 | 0.927 | 8 | 9.398 | 0.311 |
| graph-generic-d4 | 0.2 | 0.079 | 0.129 | 0.615 | 2 | 2.724 | 0.023 |
| graph-generic-d4 | 0.4 | 0.238 | 0.36 | 0.661 | 3 | 4.811 | 0.084 |
| graph-generic-d4 | 0.6 | 0.388 | 0.539 | 0.721 | 5 | 7.302 | 0.165 |
| graph-generic-d4 | 0.8 | 0.506 | 0.652 | 0.776 | 7 | 10.152 | 0.25 |
| graph-generic-d4 | 1 | 0.595 | 0.723 | 0.823 | 10 | 13.302 | 0.332 |
| graph-radial-d4 | 0.2 | 0.036 | 0.04 | 0.905 | 1 | 2.08 | 0.015 |
| graph-radial-d4 | 0.4 | 0.129 | 0.143 | 0.908 | 2 | 3.379 | 0.058 |
| graph-radial-d4 | 0.6 | 0.248 | 0.272 | 0.911 | 4 | 4.958 | 0.12 |
| graph-radial-d4 | 0.8 | 0.366 | 0.4 | 0.916 | 5 | 6.871 | 0.194 |
| graph-radial-d4 | 1 | 0.469 | 0.51 | 0.92 | 7 | 9.128 | 0.27 |

| distortion | d | mean ‖A−I‖ | max ‖A−I‖ | growth vs d = 2 |
|---|---|---|---|---|
| coherent | 2 | 0.145 | 0.424 | 1 |
| coherent | 4 | 0.27 | 1.295 | 1.867 |
| coherent | 8 | 0.401 | 2.318 | 2.774 |
| coherent | 16 | 1.02 | 8.583 | 7.055 |
| diagonal | 2 | 0.121 | 0.25 | 1 |
| diagonal | 4 | 0.131 | 0.335 | 1.078 |
| diagonal | 8 | 0.162 | 0.355 | 1.337 |
| diagonal | 16 | 0.193 | 0.355 | 1.595 |
| incoherent | 2 | 0.155 | 0.281 | 1 |
| incoherent | 4 | 0.254 | 0.533 | 1.634 |
| incoherent | 8 | 0.361 | 0.731 | 2.324 |
| incoherent | 16 | 0.541 | 1.053 | 3.487 |

What it says:

- **`ρ ≈ max |A − I|` holds in `d = 2` (ratio 0.96–1.01) and degrades with dimension and with the
  bump's asymmetry**: 0.93 for the radial bump and 0.82–0.90 for the generic one in `d = 3`, 0.91 and
  0.61–0.82 in `d = 4`. The deck's proxy is a conservative upper estimate of the contraction above
  `d = 2`, and the predicted depth over-estimates the measured one everywhere (10.2 against 8 on the
  round hill).
- **The growth of the distortion with `d` depends on its kind**: from `d = 2` to 16, ×1.6 for a
  diagonal distortion, ×3.5 for an incoherent one (the count said `√8 = 2.8`), ×7.1 for a coherent
  rank-one shear (against `d`-linear 8). The deck's `√d` is the incoherent case and a slight
  under-estimate; a coherent distortion of a chart, the kind a trained encoder produces, grows like `d`.

### G2. The conformally flat gauge: discretisation error and sensitivity

Kind: grid. Conformally flat metrics `g = λ δ` in `d = 3, 4` with an additive and a mixed `log λ`;
`v = λ^s u`, `V = s(Δ log λ + s |∇ log λ|²)`; the residual of `(−Δ + V) v − λ^{1+s} f` on the grid,
the round trip through the potential form, and the relative error of the recovered `u` when `λ` is
perturbed by a smooth field of relative RMS size `δ`.

| bed | d | s | FD residual of the identity | round trip | δλ/λ (RMS) | error of u | ratio |
|---|---|---|---|---|---|---|---|
| cf-additive-d3 | 3 | 0.25 | 5.43e-05 | 6.58e-05 | 0.02 | 0.023 | 1.167 |
| cf-mixed-d3 | 3 | 0.25 | 3.56e-05 | 5.02e-05 | 0.02 | 0.025 | 1.229 |
| cf-additive-d4 | 4 | 0.5 | 0.000366 | 0.000462 | 0.02 | 0.02 | 0.994 |
| cf-mixed-d4 | 4 | 0.5 | 0.000144 | 0.000187 | 0.02 | 0.023 | 1.143 |

What it says:

- **The gauge identity is exact**: the residual is at the level of the finite differences (5e-5 at
  25³, 4e-4 at 15⁴) and the round trip through the potential form recovers `u` to the same level.
- **The deck's sensitivity claim was wrong.** It said the gauged target inherits an error `s·δ` with
  `s = (d−2)/4`, so 0.25 δ in `d = 3`. The end-to-end error of the pipeline (potential form solved with
  the perturbed `λ`, then un-gauged) is `1.0–1.2 δ` in both dimensions, because `λ` also enters the
  source factor `λ^{1+s}` and the potential. A one per cent error in `λ` is a one per cent error in
  `u`, independent of `d`. The benefit slide of the deck ("0.25 / 1.5 / 5 % per 1 %") is to be replaced
  by "about 1 % per 1 %, at any `d`".

### G0. Geometry of the manifold matrix

Kind: analytic. For every embedded bed: intrinsic dimension `d`, ambient dimension `D`, the
obstruction to a conformal chart (the Weyl share `|W|²/|Riem|²` in `d ≥ 4`, the Cotton share in
`d = 3`, nothing in `d = 2`: every surface has isothermal coordinates), averaged and maximised over
8 sample points, and whether the suite has the conformal chart in closed form (stereographic for the
caps, `r = e^t` for the cylinders, angles for the tori, the numeric isothermal chart of the hills,
the identity for the log-Euclidean SPD twin).

| bed | d | D | obstruction | share (mean) | share (max) | conformal chart known |
|---|---|---|---|---|---|---|
| S2cap | 2 | 3 | none | 0 | 0 | yes |
| T2 | 2 | 4 | none | 0 | 0 | yes |
| hill-lop | 2 | 3 | none | 0 | 0 | yes |
| hill-round | 2 | 3 | none | 0 | 0 | yes |
| S2xR | 3 | 4 | cotton | 0 | 0 | yes |
| S3cap | 3 | 4 | cotton | 0 | 0 | yes |
| S3cap-axes1x1x2x3 | 3 | 4 | cotton | 0.023 | 0.342 | no |
| SPD2-airm | 3 | 3 | cotton | 0 | 0 | no |
| SPD2-logeuclid | 3 | 3 | cotton | 0 | 0 | yes |
| S2xR2 | 4 | 5 | weyl | 0.333 | 0.333 | no |
| S2xS2 | 4 | 6 | weyl | 0.667 | 0.667 | no |
| S3xR | 4 | 5 | weyl | 0 | 0 | yes |
| S4cap | 4 | 5 | weyl | 0 | 0 | yes |
| S4cap-axes1x1x1x1.25x1.5 | 4 | 5 | weyl | 0.000941 | 0.00344 | no |
| S4cap-axes1x1x1x1.5x2 | 4 | 5 | weyl | 0.00608 | 0.026 | no |
| S4cap-axes1x1x1x1.75x2.5 | 4 | 5 | weyl | 0.012 | 0.064 | no |
| S4cap-axes1x1x1x1x1 | 4 | 5 | weyl | 0 | 0 | yes |
| S4cap-axes1x1x1x2x3 | 4 | 5 | weyl | 0.017 | 0.105 | no |
| T4 | 4 | 8 | weyl | 0 | 0 | yes |
| hillxhill | 4 | 6 | weyl | 0.307 | 0.664 | no |
| SPD3-airm | 6 | 6 | weyl | 0.7 | 0.7 | no |
| S4xR4 | 8 | 9 | weyl | 0.286 | 0.286 | no |
| S4xS4 | 8 | 10 | weyl | 0.571 | 0.571 | no |
| S7xR | 8 | 9 | weyl | 0 | 0 | yes |
| S8cap | 8 | 9 | weyl | 0 | 0 | yes |
| T8 | 8 | 16 | weyl | 0 | 0 | yes |
| S16cap | 16 | 17 | weyl | 0 | 0 | yes |

Grid beds (`max_dev = ‖A − I‖∞` after the best global scale, `ρ` by power iteration on the Born
operator, `r_g` the separable rank of the metric fields at 1 %):

| bed | height s | ρ | max_dev | r_g |
|---|---|---|---|---|
| hill-round | 0.2 / 0.4 / 0.6 / 0.8 / 1.0 | 0.040 / 0.142 / 0.270 / 0.396 / 0.507 | 0.041 / 0.147 / 0.279 / 0.408 / 0.518 | 7 / 12 / 12 / 12 / 13 |
| hill-lop | 0.2 / 0.4 / 0.6 / 0.8 / 1.0 | 0.130 / 0.361 / 0.540 / 0.660 / 0.739 | 0.131 / 0.358 / 0.540 / 0.663 / 0.748 | 36 / 45 / 53 / 58 / 63 |
| graph-radial-d3 / graph-generic-d3 | – | 0.479 / 0.671 | 0.517 / 0.745 | – |
| graph-radial-d4 / graph-generic-d4 | – | 0.469 / 0.595 | 0.510 / 0.723 | – |

What it says:

- **The deck's shares are reproduced** and extended: `S²×R²` 1/3, `S²×S²` 2/3, and in `d = 8`
  `S⁴×R⁴` 2/7, `S⁴×S⁴` 4/7: the four values fit `d/(2(d − 1))` for `S^{d/2}×S^{d/2}` and half of
  it for `S^{d/2}×R^{d/2}`, so the share of these products stays near 1/2 and 1/4 as `d` grows. Every round cap, cylinder and torus is conformally flat to machine precision,
  in every `d` up to 16, and the Cotton tensor vanishes on `S³`, `S²×R` and on SPD(2) under the
  affine-invariant metric (SPD(2) ≅ R×H² is conformally flat, as the deck claims).
- **The ellipsoid family is a dial, but a weak one.** The mean Weyl share of
  `S⁴(1,1,1,a,b)` climbs 0 → 0.0009 → 0.006 → 0.012 → 0.017 along the five steps; the maximum over
  the cap reaches 0.105 (the deck's 0.097 was such a maximum). The curvature of a mildly squashed
  sphere is still mostly Ricci.
- **The known curved manifold is strongly obstructed from `n = 3`.** SPD(3) under the
  affine-invariant metric has Weyl share 0.70 everywhere (a symmetric space: the share is a
  constant), so no chart of SPD(3) can be conformal, while the log-Euclidean twin is flat. This is
  the cleanest "twin" for the penalty ablation: same data, two metrics, one obstructed and one not.
- **The product of two hills** carries a Weyl share of 0.31 on average (0.66 at the peaks): a
  product of surfaces is conformally flat only where one factor is flat.
- **Grid beds:** `ρ` tracks `‖A − I‖∞` within 2 % on the hills at every height, and within 7–18 %
  on the graphs in `d = 3, 4` (where the drift term grows, G1). The separable rank of the metric
  fields `r_g` grows slowly with height (7 → 13 on the round hill, 36 → 63 on the lopsided one).
- Columns the learned blocks consume: `obstruction_mean` is the x-axis of G4's floor plot and X2's
  trained-vs-oracle plot; `conformal_chart_known` selects the beds of O1's oracle arm. SPD(2)-AIRM
  has a constructible conformal chart that is not implemented yet, so O1 runs its oracle arm on the
  log-Euclidean twin only.

### G4. Oracle floors of the chart penalties

Kind: analytic, no readout and no encoder. On 2048 sample points of each bed a chart
`ψ_θ(t) = t + MLP(t)` (2 × 128, tanh, identity at initialisation) is trained for 3000 Adam steps
(cosine schedule, two restarts for the conformal target) to minimise the axis-normalised
conformality loss of the inverse metric it induces, `A = Dψ g⁻¹ Dψᵀ`, the matrix the package's
`ConformalityPenalty` sees. The same is done for the orthogonality target (`β = 0`) and the half-way
`β = 0.5`. The floor kept is the best state along the run, never worse than the identity chart,
and it is replaced by the floor of the bed's closed-form conformal chart when that is lower (the
optimiser can only over-estimate the true floor). "removed" is the fraction of `√L_conf` of the
parameter chart that the best chart removes.

| bed | d | obstruction | share | √L_conf param chart | oracle √L_conf | known chart | removed | converged | oracle √L_orth |
|---|---|---|---|---|---|---|---|---|---|
| S2cap | 2 | none | 0 | 0 | 0 | 0 | 1 |  | 0 |
| graph-generic-d2 | 2 | none | 0 | 0.174 | 0.013 | – | 0.925 |  | 0.00516 |
| graph-radial-d2 | 2 | none | 0 | 0.152 | 0.00343 | – | 0.977 |  | 0.00257 |
| hill-lop | 2 | none | 0 | 0.174 | 0.0074 | 0.0074 | 0.957 |  | 0.00338 |
| hill-round | 2 | none | 0 | 0.152 | 0.00123 | 0.00123 | 0.992 |  | 0.00089 |
| S2xR | 3 | cotton | 0 | 0.102 | 0 | 0 | 1 |  | 0 |
| S3cap | 3 | cotton | 0 | 0 | 0 | 0 | 1 |  | 0 |
| S3cap-axes1x1x2x3 | 3 | cotton | 0.015 | 0.421 | 0.139 | – | 0.668 |  | 0.014 |
| SPD2-airm | 3 | cotton | 0 | 0.061 | 0.00264 | – | 0.957 |  | 0.00108 |
| SPD2-logeuclid | 3 | cotton | 0 | 0 | 0 | 0 | 1 |  | 0 |
| graph-generic-d3 | 3 | cotton | 0.311 | 0.096 | 0.084 | – | 0.123 |  | 0.00782 |
| graph-radial-d3 | 3 | cotton | 0 | 0.086 | 0.0082 | – | 0.904 |  | 0.00637 |
| S2xR2 | 4 | weyl | 0.333 | 0.109 | 0.044 | – | 0.601 | no | 4.12e-05 |
| S2xS2 | 4 | weyl | 0.667 | 0.146 | 0.086 | – | 0.412 |  | 6.57e-05 |
| S3xR | 4 | weyl | 0 | 0.078 | 0 | 0 | 1 |  | 0 |
| S4cap | 4 | weyl | 0 | 0 | 0 | 0 | 1 |  | 0 |
| S4cap-axes1x1x1x1.25x1.5 | 4 | weyl | 0.000901 | 0.174 | 0.016 | – | 0.91 |  | 0.00554 |
| S4cap-axes1x1x1x1.5x2 | 4 | weyl | 0.00519 | 0.277 | 0.046 | – | 0.835 |  | 0.015 |
| S4cap-axes1x1x1x1.75x2.5 | 4 | weyl | 0.0097 | 0.34 | 0.09 | – | 0.735 |  | 0.03 |
| S4cap-axes1x1x1x1x1 | 4 | weyl | 0 | 0 | 0 | 0 | 1 |  | 0 |
| S4cap-axes1x1x1x2x3 | 4 | weyl | 0.012 | 0.381 | 0.144 | – | 0.623 |  | 0.034 |
| T4 | 4 | weyl | 0 | 0 | 0 | 0 | 1 |  | 0 |
| graph-generic-d4 | 4 | weyl | 0.179 | 0.052 | 0.051 | – | 0.011 |  | 0.00598 |
| graph-radial-d4 | 4 | weyl | 0 | 0.046 | 0.023 | – | 0.497 |  | 0.017 |
| hillxhill | 4 | weyl | 0.229 | 0.171 | 0.149 | – | 0.128 |  | 0.00387 |
| SPD3-airm | 6 | weyl | 0.7 | 0.094 | 0.082 | – | 0.128 |  | 0.00953 |
| S4xS4 | 8 | weyl | 0.571 | 0.107 | 0.089 | – | 0.168 |  | 3.62e-05 |
| S8cap | 8 | weyl | 0 | 0 | 0 | 0 | 1 |  | 0 |

Not converged after 3000 steps: `S²×R²` (its floor is an upper bound). The optimiser drifted
upward on `hillxhill` (best state at step 1879) and `graph-generic-d4` (step 1036): their floors
are the best state seen and are also upper bounds.

What it says:

- **The existence predictions hold where they can be checked.** `d = 2`: every bed reaches a floor
  below 0.013 (the hills at their known isothermal chart: 0.0012 / 0.0074; the graphs with the
  optimiser alone: 0.003 / 0.013). `d = 3`: zero on the beds with vanishing Cotton tensor (`S³` cap,
  `S²×R`, SPD(2)-AIRM 0.0026 with 96 % removed and no closed-form chart implemented, radial graph
  0.008) and clearly positive on the two with Cotton ≠ 0 (squashed `S³` 0.139, generic graph
  0.084). `d = 4`: exactly zero on `S⁴`, `S³×R`, `T⁴` (closed-form charts), positive on every bed
  with `W ≠ 0`: `S²×R²` 0.044 (share 1/3), `S²×S²` 0.086 (2/3), the product of hills 0.149 (0.23),
  the generic graph 0.051 (0.18). `d = 6, 8`: SPD(3) 0.082 (share 0.70), `S⁴×S⁴` 0.089 (4/7),
  `S⁸` cap 0.
- **One optimisation gap remains visible:** `graph-radial-d4` has `W = 0` (a hypersurface of
  revolution is conformally flat) but the optimiser removes only half of the deviation (0.023 left).
  In the first run of this block `S³×R` showed the same gap (0.0415) until the closed-form chart
  bounded it. Where no closed form exists, the 2 × 128 chart with 3000 steps is the limit of the
  oracle in `d = 4`, and X2's "trained floor minus oracle floor" must be read with that in mind.
- **The ellipsoid family is monotone but not proportional.** Along `(1,1,1,1,1) → (1,1,1,2,3)` the
  floor climbs 0 → 0.016 → 0.046 → 0.090 → 0.144 while the mean Weyl share climbs
  0 → 0.0009 → 0.005 → 0.010 → 0.012. The share is a ratio (Weyl against the whole curvature); the
  floor is an absolute distortion of the best chart, so a mildly squashed sphere with a tiny Weyl
  share still leaves a floor of 0.14, more than `S²×R²` with share 1/3. The Weyl share orders beds
  within a family; it does not predict the size of the floor across families.
- **Orthogonal charts exist almost everywhere, and they are not conformal.** The orthogonality
  floor is below 1e-4 on every product (`S²×R²`, `S²×S²`, `S⁴×S⁴`), 0.004 on the product of hills
  and at most 0.034 elsewhere, while the conformality of the orthogonality-optimal chart is 0.39
  on the round hill, 0.71 on the product of hills, 0.77 on the generic `d = 4` graph and 0.80 on
  SPD(3). The oracle side of claim A1 is confirmed: today's target is reachable on charts that are
  far from conformal, so it cannot steer towards them. Whether that costs anchors is X1's question.
- **The count `2/(d+1)` does not predict the removable fraction** (right panel of the figure). On
  unobstructed beds the best chart removes 90–100 % of `√L_conf` in every `d` from 2 to 8; on
  obstructed beds 1–60 %, set by the obstruction and the curvature size, not by `d`. The count says
  how many of the `d(d+1)/2` metric functions a chart can fix; it says nothing about the normalised
  penalty, and the deck's claim A3 is to be reworded accordingly.
- `β = 0.5` lands between the two floors on every bed (for example 0.034 on `S²×R²`, 0.122 on the
  product of hills): the single `β` axis of the package's loss spans today's target and the
  conformal one continuously, as intended.
- For the plan: X1 trains encoders on the core beds and X2 plots their floors against this column;
  the beds whose oracle floor is a closed-form zero (`S⁴` cap, `S³×R`, `T⁴`, the hills) give a clean
  "optimisation only" reading, SPD(2)-AIRM, `S²×R²` and `graph-radial-d4` give an upper bound.

## Tier 1: learned charts

### Protocol (revised after the O1 pilots, 2026-10-27)

Every learned row: `IGLModule(D, d, 1)` with the default encoder (256 × 2, LayerNorm, SiLU, orthogonal
init), 64 anchors × 4 scales, 400 epochs, batch 256, `source_l2 = 1e-3`, the package's gradient
clipping (1.0), constant learning rate 1e-3, no early stop within the budget but the best-validation
state restored at the end; `N = 4096 / 1024 / 2048` (8192 train in `d ≥ 8`), targets standardised,
inputs with noise 0.005. The readout is solved on the train split and scored on the test split (NRMSE
at the full prefix); costs are measured on the frozen chart: the greedy anchor-pruning curve, the
fixed-grid bump count of `u` at 5 / 2 / 1 % (censored at 128 atoms), the 2-D separable rank.

Two pins of the plan were dropped by the pilots on the round hill (nine fits per variant, three seeds):

- **No clipping** (`grad_clip = 0`, meant to avoid a hidden learning-rate cut in penalty arms): some
  initialisations blow up (`‖A − I‖∞` of 400–1000 on the test chart) in every arm, which destroys the
  pairing. Clipping is the package's behaviour and is kept; the penalty weights are calibrated
  instead.
- **Oracle charts through a trained encoder**: IGL trained on an exact chart as inputs is unstable at
  any learning rate (1e-3, 3e-4, 1e-4): the orthogonally initialised encoder already is a chart, and
  training only distorts it (validation peaks before epoch 30 and degrades, or the Jacobian blows
  up). Weight decay and input-noise augmentation do not help. The oracle arms (`conformal`, `param`)
  therefore **freeze the chart**: identity encoder, kernel and readout trained. With that, the three
  seeds converge (best epoch > 370, flat validation) and the conformal chart beats the parameter chart
  on every seed of the pilot (0.019 / 0.023 / 0.028 against 0.023 / 0.025 / 0.029; atoms at 5 %:
  28 / 42 / 36 against 34 / 47 / 54).

Chart inputs are standardised per column (a constant affine map, free for the geometry). Sphere caps
are parametrised by their conformal chart, so their non-conformal reference is the graph chart (the
first `d` ambient coordinates).

### O1. Oracle brackets

Kind: learned chart (label `tier1`, MPS, 3 seeds, 66 fits, 43 min). IGL with three kinds of inputs on
each bed: the **frozen conformal chart** (identity encoder), the **frozen parameter chart** (for the
sphere caps the graph chart, since their parameter chart is the conformal one), and the **embedded
ambient data** (IGL today: the chart is learned). No penalty in any arm. Columns: test NRMSE at the
full 64 anchors (median and seed range), then the median along the greedy pruning curve at 32, 16 and
8 anchors; fixed-grid Gaussian atoms of `u` at 5 % per seed (censored at 128); the conformality floor
of the chart on the test set; the elbow dimension.

| bed | d | D | inputs | encoder | NRMSE @64 (range) | @32 | @16 | @8 | atoms 5 % (seeds) | √L_conf | d_eff | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S2cap | 2 | 2 | conformal | identity (frozen chart) | 0.0222 (0.0207–0.0247) | 0.022 | 0.028 | 0.071 | 43/48/36 | 0 | 2 | 3 |
| S2cap | 2 | 2 | param (graph chart) | identity (frozen chart) | 0.0355 (0.0293–0.0422) | 0.034 | 0.043 | 0.132 | 59/88/61 | 0 | 2 | 3 |
| S2cap | 2 | 3 | ambient | mlp | 0.0228 (0.0219–0.0298) | 0.023 | 0.023 | 0.025 | 36/34/44 | 0.47 | 2 | 3 |
| S2cap | 2 | 256 | ambient | mlp | 0.0219 (0.0208–0.0222) | 0.022 | 0.022 | 0.023 | 41/43/36 | 0.454 | 2 | 3 |
| hill-round | 2 | 2 | conformal | identity (frozen chart) | 0.0227 (0.0192–0.0282) | 0.022 | 0.033 | 0.061 | 28/42/36 | 0 | 2 | 3 |
| hill-round | 2 | 2 | param (parameter chart) | identity (frozen chart) | 0.0246 (0.0228–0.0286) | 0.025 | 0.032 | 0.063 | 34/47/54 | 0 | 2 | 3 |
| hill-round | 2 | 8 | ambient | mlp | 0.0179 (0.0171–0.0193) | 0.018 | 0.018 | 0.018 | 35/40/34 | 0.494 | 2 | 3 |
| hill-round | 2 | 32 | ambient | mlp | 0.0189 (0.0175–0.0212) | 0.019 | 0.019 | 0.021 | 33/30/39 | 0.566 | 2 | 3 |
| SPD2-airm | 3 | 3 | ambient | mlp | 0.0585 (0.0256–0.0717) | 0.058 | 0.06 | 0.07 | 24/–/– | 0.725 | 2 | 3 |
| SPD2-airm | 3 | 3 | param (parameter chart) | identity (frozen chart) | 0.0436 (0.0376–0.0443) | 0.044 | 0.053 | 0.09 | 63/62/64 | 0 | 3 | 3 |
| S2xR2 | 4 | 4 | param (parameter chart) | identity (frozen chart) | 0.0684 (0.0660–0.0852) | 0.071 | 0.1 | 0.165 | –/–/– | 0 | 4 | 3 |
| S2xR2 | 4 | 5 | ambient | mlp | 0.0426 (0.0382–0.0501) | 0.043 | 0.043 | 0.044 | 86/124/74 | 0.667 | 2 | 3 |
| S3xR | 4 | 4 | conformal | identity (frozen chart) | 0.1043 (0.1031–0.1330) | 0.109 | 0.15 | 0.209 | –/–/– | 0 | 4 | 3 |
| S3xR | 4 | 4 | param (parameter chart) | identity (frozen chart) | 0.0658 (0.0655–0.0823) | 0.072 | 0.098 | 0.152 | –/–/– | 0 | 4 | 3 |
| S3xR | 4 | 5 | ambient | mlp | 0.0474 (0.0416–0.0496) | 0.047 | 0.048 | 0.049 | 111/79/52 | 0.682 | 3 | 3 |
| S4cap | 4 | 4 | conformal | identity (frozen chart) | 0.0675 (0.0649–0.0800) | 0.072 | 0.116 | 0.177 | –/–/– | 0 | 4 | 3 |
| S4cap | 4 | 4 | param (graph chart) | identity (frozen chart) | 0.1147 (0.0968–0.1196) | 0.118 | 0.154 | 0.205 | –/–/– | 0 | 4 | 3 |
| S4cap | 4 | 5 | ambient | mlp | 0.0462 (0.0449–0.0497) | 0.046 | 0.046 | 0.048 | 74/75/55 | 0.747 | 3 | 3 |
| S8cap | 8 | 8 | conformal | identity (frozen chart) | 0.0208 (0.0207–0.0331) | 0.021 | 0.026 | 0.054 | –/–/– | 0 | 4 | 3 |
| S8cap | 8 | 8 | param (graph chart) | identity (frozen chart) | 0.0245 (0.0236–0.0261) | 0.024 | 0.031 | 0.075 | –/–/– | 0 | 4 | 3 |
| S8cap | 8 | 9 | ambient | mlp | 0.0209 (0.0208–0.0213) | 0.021 | 0.021 | 0.021 | 25/27/23 | 0.798 | 4 | 3 |
| S8cap | 8 | 256 | ambient | mlp | 0.0224 (0.0219–0.0232) | 0.022 | 0.022 | 0.024 | 32/32/38 | 0.724 | 3 | 3 |
figure: /Users/aquemy/projects/hother/intrinsic-green-learning/results/benchmarks/geometry/tier1/o1_oracle_bracket/figures/o1_oracle_bracket.png

Figure: `results/benchmarks/geometry/tier1/o1_oracle_bracket/figures/o1_oracle_bracket.png` (NRMSE
against anchors kept, one panel per bed, the seed range shaded).

What it says:

- **Among frozen charts, the conformal chart wins where it is tame.** On the hills and `S²` it beats
  the parameter chart on every seed in error (0.023 vs 0.025 and 0.022 vs 0.036 at 64 anchors; the
  gap widens to 0.061 vs 0.063 and 0.071 vs 0.132 at 8 anchors) and in atoms at 5 % (28/42/36 vs
  34/47/54; 43/48/36 vs 59/88/61). On `S⁴` and `S⁸` the stereographic chart beats the graph chart
  (0.068 vs 0.115; 0.021 vs 0.025). This is the chart-level content of proposal A, confirmed on a
  learned readout: at a fixed set of atoms, a conformal chart is the cheaper of two given charts.
- **But the conformal chart loses where its conformal factor varies strongly.** On the cylinder
  `S³×R` the conformal chart `x = e^t ω` is the worst input of the three (0.104 against 0.066 for the
  cylindrical parameters and 0.047 learned): `λ = e^{−2t}` spreads the data over three decades and
  64 Gaussian atoms cannot follow. Conformality alone is not the right target for a Gaussian kernel;
  the kernel needs the conformal factor carried somewhere (the operator side, proposal E's gauge),
  exactly the deck's reading of `λ`.
- **IGL's learned chart is already better than any geometric chart, and by a wide margin at few
  anchors.** At 64 anchors the learned chart matches or beats the frozen conformal chart on every bed
  but SPD(2) (hill 0.018 vs 0.023; `S²` 0.023 vs 0.022; `S⁴` 0.046 vs 0.068; `S³×R` 0.047 vs 0.104;
  `S⁸` 0.021 vs 0.021). Along the pruning curve the learned chart is **flat down to 8 anchors**
  (hill 0.018 → 0.018, `S⁴` 0.046 → 0.048, `S⁸` 0.021 → 0.021) while every frozen chart climbs by a
  factor 2.5–4 (hill conformal 0.023 → 0.061, `S⁴` conformal 0.068 → 0.177). The encoder adapts the
  chart to the target `u`, not to the geometry: in the learned chart the solution is representable
  by about eight Gaussian atoms, which no geometric chart achieves. The oracle gap of the plan
  (conformal minus ambient) is therefore **zero or negative** on every bed; a geometry-only chart
  penalty cannot be justified by the readout error at these sizes, and must be judged by what it
  does to this co-adaptation (X1).
- **The one exception is SPD(2).** The frozen log-Eig chart (0.044, seed range 0.038–0.044) beats the
  learned chart (median 0.059, range 0.026–0.072), which is unstable on this bed; its learned
  pruning curve is flat as elsewhere. The learned chart's conformality floor is 0.73 on SPD(2) and
  0.75–0.80 on the `d ≥ 4` caps, against 0.45–0.57 on the 2-D beds: the learned charts are far from
  conformal everywhere, and the readout does not care.
- **Ambient dimension costs little at these sizes.** `S²` in `R²⁵⁶` is as good as in `R³`
  (0.022 vs 0.023), `S⁸` in `R²⁵⁶` loses 7 % (0.0224 vs 0.0209) and the hill in `R³²` 6 %; the
  off-manifold share of the Jacobian grows from 0.09–0.13 at native `D` to 0.14–0.21 at `D = 256`.
- **The 2-D separable rank** of `u` is 5 in the frozen charts of the hill and 4 / 7 in the conformal /
  graph charts of `S²`, against 14–22 in the learned charts: the learned chart is not axis-aligned,
  and its rank says nothing about its atom count (the R0 reading, now on learned charts).
- **Elbows:** `d_eff` = 2 on every 2-D bed, 3–4 on `S⁴` (true 4), 3–4 on `S⁸` (true 8): the elbow of
  the dimension curve under-reads `d` as soon as the target is smooth, with `max_dim = d`.

Consequence for the plan: the premise that the anchor count is set by the geometry of the chart
holds between *given* charts (A, confirmed) and fails against the *learned* chart, which co-adapts
to the target. X1 asks the only remaining question for proposals A and B: does the conformality
penalty, at a calibrated weight, preserve this co-adaptation (no regression on the flat pruning
curve) while adding what the deck promises (a chart-limited certificate, a smaller `λ` spread,
stability on SPD(2))? Proposal E's reading of `λ` gains a direct witness in `S³×R`.
