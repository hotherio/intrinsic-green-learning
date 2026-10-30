# Geometry suite: results

Every number here is reproducible from this repository (`benchmarks/geometry/`, `make geometry-tier0`,
label `tier0`). Two kinds of results, never mixed: **analytic / grid** (no training; the geometry is
known) and **learned chart** (IGL trained). The deck's own numbers (two 2-D hills at full height:
ρ 0.507 / 0.739, Born depth 8 / 18, naive error 33 / 49 %, source ranks 4 / 9 in the data chart and
3 / 5 in the conformal chart, 5 / 7 in the orthogonal non-conformal chart) are reproduced by
`testbeds/hills.py`; everything else is new.

## Tier 0: analytic and grid results

Audit notes (2026-10-27, two independent code reviews plus a self-review; the fixes are in the suite
and G1, G2 and G4 were rerun under the fixed code; the tables below are the post-audit numbers):

- R0's two cost measures use different tolerances: the separable rank is relative to `‖u‖`, the atom
  count to `‖u − mean u‖` (the NRMSE convention), so "1 %" is about 1.5× stricter for the atom count.
  Both are fair across charts (same samples, same `u`); they are not comparable to each other. The
  rank on a chart whose image is not a box includes nearest-neighbour fill outside the convex hull;
  verified immaterial at 1 % on the hills.
- G1's "incoherent" distortion reused the same six Fourier modes for every entry (mean entry
  correlation 0.3), so it was partly coherent; its predicted Born depth dropped the naive starting
  error (`n = log(naive/ε)/log(1/ρ)`); its `ρ` against `‖A − I‖∞` pair was taken without the best
  global scale; and the power iteration returned an unconverged norm (the deck's 0.507 is about 1 %
  high). All four are fixed and G1 is rerun.
- G2 measures the end-to-end sensitivity of `u` to `λ` through the gauge pipeline, which is the full
  metric sensitivity of the Poisson solution (order one at every `d`, `d = 2` included where
  `s = 0`); the deck's `s·δ` is the gauge factor alone, not the total. The conclusion stands; the
  comparison line in the figure is relabelled.
- G4's loss had an absolute floor on its denominator that a Jacobian spike at a few points could
  exploit, and no non-degeneracy was logged. The block now rescales `A` to unit mean trace before the
  loss, logs the range of `tr A` and the smallest `|det Dψ|` of the best chart, draws the restarts and
  the known-chart bound on the same sample of points, and keeps the optimiser's and the known chart's
  numbers apart. The hills' numeric conformal chart was conformal only to 6e-2 at the lopsided peak on
  the deck's 49-point grid; it is now built on a 145-point grid (7e-3 at the worst point), so their
  known-chart floors move.
- Targets: the graph-Laplacian family (used in `d ≥ 5` where no eigenfunctions exist, and on SPD(n ≥ 3))
  cut the Gaussian kernel at a density-dependent kNN radius inside the Gaussian, which breaks the
  `α = 1` density correction, and SPD(n) under the affine-invariant metric was built with Euclidean
  distances, identical to its log-Euclidean twin. It is now a fixed-radius truncated Gaussian with the
  affine-invariant distances on SPD, validated on closed spheres (shape error of `u`: `S²` 4–6 %,
  `S⁸` 6–9 %). On the beds' bounded patches no boundary condition is imposed, so `u` is a smooth
  graph-Laplacian field of `f`, the same for every arm, not the Dirichlet solution. No result below
  used this family yet (the core beds are `d ≤ 4` or have eigenfunctions).
- The parameter-chart substitution for beds parametrised by their conformal chart now applies to
  hypersurface caps only; on tori the first `d` ambient coordinates are not a chart (no torus ran
  through it).

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

Kind: analytic. Part A: hills and graph hypersurfaces at five heights; `ρ` by power iteration (300
iterations, geometric mean of the growth ratios over the second half), the proxy `max |A − I|` after
the best global scale, the measured Born depth to `1e-3` against the predicted
`log(e_naive/ε)/log(1/ρ)` where `e_naive` is the error of the flat solve that ignores the hill, and
that naive error. Part B: random charts of the flat torus, `J = I + ε E(θ)`, with `E` built from
random Fourier fields of unit per-entry standard deviation of three kinds: diagonal, incoherent
(every entry its own independent field) and coherent (a rank-one shear `a bᵀ φ(θ)`).

| bed | height s | ρ | max|A−I| | ρ / max|A−I| | depth 1e-3 | predicted | naive error |
|---|---|---|---|---|---|---|---|
| hill-lop | 0.2 | 0.131 | 0.131 | 1.001 | 2 | 1.891 | 0.047 |
| hill-lop | 0.4 | 0.361 | 0.358 | 1.007 | 4 | 4.958 | 0.157 |
| hill-lop | 0.6 | 0.54 | 0.54 | 0.999 | 8 | 9.149 | 0.281 |
| hill-lop | 0.8 | 0.658 | 0.663 | 0.993 | 12 | 14.299 | 0.395 |
| hill-lop | 1 | 0.736 | 0.748 | 0.984 | 18 | 20.223 | 0.491 |
| hill-round | 0.2 | 0.04 | 0.041 | 0.961 | 1 | 0.941 | 0.021 |
| hill-round | 0.4 | 0.141 | 0.147 | 0.963 | 2 | 2.226 | 0.078 |
| hill-round | 0.6 | 0.269 | 0.279 | 0.965 | 4 | 3.858 | 0.158 |
| hill-round | 0.8 | 0.394 | 0.408 | 0.967 | 6 | 5.921 | 0.247 |
| hill-round | 1 | 0.503 | 0.518 | 0.97 | 8 | 8.453 | 0.335 |
| graph-generic-d3 | 0.2 | 0.107 | 0.13 | 0.82 | 2 | 1.567 | 0.033 |
| graph-generic-d3 | 0.4 | 0.305 | 0.361 | 0.846 | 4 | 4.004 | 0.116 |
| graph-generic-d3 | 0.6 | 0.469 | 0.536 | 0.876 | 6 | 7.115 | 0.218 |
| graph-generic-d3 | 0.8 | 0.584 | 0.659 | 0.886 | 9 | 10.74 | 0.32 |
| graph-generic-d3 | 1 | 0.666 | 0.745 | 0.894 | 13 | 14.787 | 0.411 |
| graph-radial-d3 | 0.2 | 0.038 | 0.041 | 0.931 | 1 | 0.896 | 0.019 |
| graph-radial-d3 | 0.4 | 0.137 | 0.146 | 0.935 | 2 | 2.134 | 0.07 |
| graph-radial-d3 | 0.6 | 0.261 | 0.278 | 0.939 | 4 | 3.697 | 0.143 |
| graph-radial-d3 | 0.8 | 0.384 | 0.407 | 0.944 | 6 | 5.661 | 0.227 |
| graph-radial-d3 | 1 | 0.49 | 0.517 | 0.948 | 8 | 8.053 | 0.311 |
| graph-generic-d4 | 0.2 | 0.081 | 0.129 | 0.625 | 2 | 1.251 | 0.023 |
| graph-generic-d4 | 0.4 | 0.24 | 0.36 | 0.667 | 3 | 3.108 | 0.084 |
| graph-generic-d4 | 0.6 | 0.381 | 0.539 | 0.707 | 5 | 5.294 | 0.165 |
| graph-generic-d4 | 0.8 | 0.506 | 0.652 | 0.776 | 7 | 8.106 | 0.25 |
| graph-generic-d4 | 1 | 0.595 | 0.723 | 0.823 | 10 | 11.175 | 0.332 |
| graph-radial-d4 | 0.2 | 0.036 | 0.04 | 0.905 | 1 | 0.82 | 0.015 |
| graph-radial-d4 | 0.4 | 0.129 | 0.143 | 0.908 | 2 | 1.985 | 0.058 |
| graph-radial-d4 | 0.6 | 0.248 | 0.272 | 0.911 | 4 | 3.437 | 0.12 |
| graph-radial-d4 | 0.8 | 0.366 | 0.4 | 0.916 | 5 | 5.237 | 0.194 |
| graph-radial-d4 | 1 | 0.469 | 0.51 | 0.92 | 7 | 7.397 | 0.27 |

| distortion | d | mean ‖A−I‖ | max ‖A−I‖ | growth vs d = 2 |
|---|---|---|---|---|
| coherent | 2 | 0.097 | 0.35 | 1 |
| coherent | 4 | 0.206 | 0.947 | 2.707 |
| coherent | 8 | 0.41 | 2.699 | 7.714 |
| coherent | 16 | 1.002 | 8.069 | 23.065 |
| diagonal | 2 | 0.098 | 0.279 | 1 |
| diagonal | 4 | 0.146 | 0.378 | 1.357 |
| diagonal | 8 | 0.175 | 0.388 | 1.393 |
| diagonal | 16 | 0.202 | 0.403 | 1.446 |
| incoherent | 2 | 0.12 | 0.321 | 1 |
| incoherent | 4 | 0.232 | 0.495 | 1.542 |
| incoherent | 8 | 0.358 | 0.694 | 2.161 |
| incoherent | 16 | 0.531 | 0.862 | 2.684 |

Figure: `results/benchmarks/geometry/tier0/g1_rho_law/figures/g1_rho_law.png`.

What it says:

- **`ρ ≈ max |A − I|` holds in `d = 2` (ratio 0.96–1.00) and degrades with dimension and with the
  bump's asymmetry**: 0.93–0.95 for the radial bump and 0.82–0.89 for the generic one in `d = 3`,
  0.91 and 0.63–0.82 in `d = 4`. The deck's proxy is a conservative upper estimate of the contraction
  above `d = 2`. The converged `ρ` of the unit round hill is 0.503 (the deck's 0.507 came from an
  unconverged power iteration) and 0.736 on the lopsided one (deck 0.739).
- **The predicted depth tracks the measured one once the naive error is in the formula**: 8.5
  against 8 and 20.2 against 18 on the hills, 8.1 / 14.8 against 8 / 13 for the `d = 3` graphs,
  7.4 / 11.2 against 7 / 10 in `d = 4`. The prediction is at most two iterations high (it uses the
  asymptotic rate; the early iterates contract faster), and never low. The earlier version of this
  table predicted 10.2 against 8 because the formula dropped the naive error; "the proxy
  over-predicts the depth" was that bug, not a property of the Born series.
- **The growth of the distortion with `d` depends on its kind**: from `d = 2` to 16, ×1.45 for a
  diagonal distortion, ×2.7 for an incoherent one (the count's `√8 = 2.8`), ×23 for a coherent
  rank-one shear (the count said `d`, that is ×8). The deck's `√d` is the incoherent case, now with
  every entry an independent field (the earlier run reused the same modes across entries, which
  correlated them and gave ×3.5). A coherent distortion grows faster than `d` because `A − I` is
  quadratic in `εE` and the square of a rank-one term with unit entries scales like `d²`; the
  Jacobian of a trained encoder is of the coherent kind, so the operator side's `ρ` cannot be
  assumed to stay below one past `d ≈ 4` without a chart penalty that keeps the distortion
  incoherent or small.

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
  (Rerun after the audit with the converged power iteration: the table is unchanged to the digits
  shown; the spectral radius is not part of it.)
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
conformality loss of the inverse metric it induces, `A = Dψ g⁻¹ Dψᵀ` rescaled to unit mean trace,
the matrix the package's `ConformalityPenalty` sees. The same is done for the orthogonality target
(`β = 0`) and the half-way `β = 0.5`. The floor kept is the best state along the run, never worse
than the identity chart, and it is replaced by the floor of the bed's closed-form conformal chart,
evaluated on the same sample, when that is lower (the optimiser can only over-estimate the true
floor). "removed" is the fraction of `√L_conf` of the parameter chart that the best chart removes.

| bed | d | obstruction | share | √L_conf param chart | oracle √L_conf | known chart | removed | converged | oracle √L_orth |
|---|---|---|---|---|---|---|---|---|---|
| S2cap | 2 | none | 0 | 1.02e-16 | 1.02e-16 | 1.02e-16 | 1 |  | 3.75e-17 |
| graph-generic-d2 | 2 | none | 0 | 0.176 | 0.012 | – | 0.931 |  | 0.0051 |
| graph-radial-d2 | 2 | none | 0 | 0.155 | 0.00539 | – | 0.965 |  | 0.00257 |
| hill-lop | 2 | none | 0 | 0.177 | 0.000879 | 0.000879 | 0.995 |  | 0.000681 |
| hill-round | 2 | none | 0 | 0.155 | 0.000138 | 0.000138 | 0.999 |  | 9.92e-05 |
| S2xR | 3 | cotton | 1.65e-31 | 0.103 | 1.88e-16 | 1.88e-16 | 1 |  | 2.96e-17 |
| S3cap | 3 | cotton | 7.15e-32 | 1.28e-16 | 1.28e-16 | 1.28e-16 | 1 |  | 4.01e-17 |
| S3cap-axes1x1x2x3 | 3 | cotton | 0.015 | 0.421 | 0.118 | – | 0.718 | no | 0.015 |
| SPD2-airm | 3 | cotton | 1.99e-29 | 0.06 | 0.00257 | – | 0.957 |  | 0.00108 |
| SPD2-logeuclid | 3 | cotton | 0 | 0 | 0 | 0 | 1 |  | 0 |
| graph-generic-d3 | 3 | cotton | 0.311 | 0.096 | 0.092 | – | 0.042 |  | 0.00844 |
| graph-radial-d3 | 3 | cotton | 4.36e-31 | 0.087 | 0.013 | – | 0.853 |  | 0.00723 |
| S2xR2 | 4 | weyl | 0.333 | 0.109 | 0.029 | – | 0.735 | no | 2.73e-17 |
| S2xS2 | 4 | weyl | 0.667 | 0.147 | 0.088 | – | 0.399 |  | 3.9e-17 |
| S3xR | 4 | weyl | 1.02e-31 | 0.079 | 2.67e-16 | 2.67e-16 | 1 |  | 3.5e-17 |
| S4cap | 4 | weyl | 2.25e-31 | 1.29e-16 | 1.29e-16 | 1.29e-16 | 1 |  | 3.88e-17 |
| S4cap-axes1x1x1x1.25x1.5 | 4 | weyl | 0.000901 | 0.174 | 0.015 | – | 0.915 |  | 0.0054 |
| S4cap-axes1x1x1x1.5x2 | 4 | weyl | 0.00519 | 0.276 | 0.045 | – | 0.836 | no | 0.015 |
| S4cap-axes1x1x1x1.75x2.5 | 4 | weyl | 0.0097 | 0.34 | 0.079 | – | 0.768 |  | 0.03 |
| S4cap-axes1x1x1x1x1 | 4 | weyl | 2.25e-31 | 1.29e-16 | 1.29e-16 | 1.29e-16 | 1 |  | 3.88e-17 |
| S4cap-axes1x1x1x2x3 | 4 | weyl | 0.012 | 0.381 | 0.126 | – | 0.668 |  | 0.03 |
| T4 | 4 | weyl | 0 | 1.03e-16 | 1.03e-16 | 1.03e-16 | 1 |  | 0 |
| graph-generic-d4 | 4 | weyl | 0.179 | 0.052 | 0.051 | – | 0.015 |  | 0.00605 |
| graph-radial-d4 | 4 | weyl | 2.62e-31 | 0.048 | 0.024 | – | 0.496 |  | 0.016 |
| hillxhill | 4 | weyl | 0.229 | 0.171 | 0.152 | – | 0.114 |  | 0.00393 |
| SPD3-airm | 6 | weyl | 0.7 | 0.094 | 0.077 | – | 0.182 |  | 0.00943 |
| S4xS4 | 8 | weyl | 0.571 | 0.108 | 0.09 | – | 0.167 |  | 3.97e-17 |
| S8cap | 8 | weyl | 8.89e-32 | 1.51e-16 | 1.51e-16 | 1.51e-16 | 1 |  | 3.31e-17 |

Figure: `results/benchmarks/geometry/tier0/g4_oracle_floor/figures/g4_oracle_floor.png`.

Not converged after 3000 steps (floor still moving over the last 10 % of the run by more than the
restart spread): `S²×R²`, the (1,1,1,1.5,2) ellipsoid and the squashed `S³`; their floors are upper
bounds. "tr-ratio" (max/min of `tr A` over the sample) and `min |det Dψ|` of the best chart are
logged for the optimiser's best chart on every bed to exclude degenerate charts: no bed collapses
(the smallest determinant is 0.047 on an ellipsoid; the largest trace ratio, 24.7, is the optimiser's
chart of the lopsided hill, whose reported floor comes from the known isothermal chart anyway).

What it says:

- **The existence predictions hold where they can be checked.** `d = 2`: every bed reaches a floor
  below 0.013 (the hills at their known isothermal chart: 1.4e-4 / 8.8e-4 on the 145-point chart,
  which is conformal to 7e-3 at its worst point; the graphs with the optimiser alone: 0.005 /
  0.012). `d = 3`: zero on the beds with vanishing Cotton tensor (`S³` cap, `S²×R`, SPD(2)-AIRM 0.0026
  with 96 % removed and no closed-form chart implemented, radial graph 0.013) and clearly positive
  on the two with Cotton ≠ 0 (squashed `S³` 0.118, generic graph 0.092). `d = 4`: exactly zero on
  `S⁴`, `S³×R`, `T⁴` (closed-form charts), positive on every bed with `W ≠ 0`: `S²×R²` 0.029 (share
  1/3), `S²×S²` 0.088 (2/3), the product of hills 0.152 (0.23), the generic graph 0.051 (0.18).
  `d = 6, 8`: SPD(3) 0.077 (share 0.70), `S⁴×S⁴` 0.090 (4/7), `S⁸` cap 0.
- **One optimisation gap remains visible:** `graph-radial-d4` has `W = 0` (a hypersurface of
  revolution is conformally flat) but the optimiser removes only half of the deviation (0.024 left,
  with a healthy chart: trace ratio 2.1, determinant 0.95). The floor of `S³×R` showed the same gap
  (0.04) until the closed-form chart bounded it. Where no closed form exists, the 2 × 128 chart with
  3000 steps is the limit of the oracle in `d = 4`, and X2's "trained floor minus oracle floor" must
  be read with that in mind.
- **The ellipsoid family is monotone but not proportional.** Along `(1,1,1,1,1) → (1,1,1,2,3)` the
  floor climbs 0 → 0.015 → 0.045 → 0.079 → 0.127 while the mean Weyl share climbs
  0 → 0.0009 → 0.005 → 0.010 → 0.012. The share is a ratio (Weyl against the whole curvature); the
  floor is an absolute distortion of the best chart, so a mildly squashed sphere with a tiny Weyl
  share still leaves a floor of 0.13, more than `S²×R²` with share 1/3. The Weyl share orders beds
  within a family; it does not predict the size of the floor across families.
- **Orthogonal charts exist almost everywhere, and they are not conformal.** The orthogonality
  floor is at machine precision on every product (`S²×R²`, `S²×S²`, `S⁴×S⁴`: block charts are
  orthogonal), 0.004 on the product of hills and at most 0.03 elsewhere, while the conformality of
  the orthogonality-optimal chart is 0.39 on the round hill, 0.71 on the product of hills, 0.77 on
  the generic `d = 4` graph and 0.80 on SPD(3). The oracle side of claim A1 is confirmed: today's
  target is reachable on charts that are far from conformal, so it cannot steer towards them.
  Whether that costs anchors is X1's question.
- **The count `2/(d+1)` does not predict the removable fraction** (right panel of the figure). On
  unobstructed beds the best chart removes 85–100 % of `√L_conf` in every `d` from 2 to 8; on
  obstructed beds 1–75 %, set by the obstruction and the curvature size, not by `d`. The count says
  how many of the `d(d+1)/2` metric functions a chart can fix; it says nothing about the normalised
  penalty, and the deck's claim A3 is to be reworded accordingly.
- `β = 0.5` lands between the two floors on every bed: the single `β` axis of the package's loss
  spans today's target and the conformal one continuously, as intended.
- Against the pre-audit run (absolute `eps` in the loss, 49-point hill chart, unconverged power
  iteration elsewhere): every floor moved by less than 0.02, the hills' known-chart floors fell by
  an order of magnitude with the finer chart, and no conclusion changed.
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
  off-manifold part of the Jacobian grows with `D` (the ratio of the on-manifold to the full metric norm
  falls from 0.87–0.91 at native `D` to 0.79–0.86 at `D = 256`; later rows report the energy share
  `1 − tr(A_tan)/tr(A_full)` instead).
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

### X1. Penalty ablation on the core beds (reduced arm set)

Kind: learned chart (label `tier1`, MPS, 3 seeds, 84 fits + 18 sweep fits, 2.4 h). Arms on ambient inputs: `none`;
`orth-today` (the package's `OrthogonalityPenalty(0.1, every=20)`); `conf` (`ConformalityPenalty`,
`β = 1`, axis-normalised, all prefixes, `every = 2`, weight calibrated once per bed so that the
penalty gradient is 0.3 of the task gradient at epoch 50 of a penalty-free fit); `conf-tan` (the
same with the bed's tangent projector, so the penalty sees `J P Jᵀ` instead of `J Jᵀ`). Paired
design: the delta column is per seed against `none` on the same data and minibatches. `conf@0.1` and
`conf@0.03` are `conf` at 0.1× and 0.03× the calibrated weight (the weight sweep).

| bed | d | D | arm | weight | NRMSE @64 (range) | @16 | @8 | Δ vs none @64 (median [min, max], worse/n) | √L_orth | √L_conf | ‖A−I‖∞ | d_eff | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T2 | 2 | 4 | none | – | 0.0119 (0.0111–0.0119) | 0.012 | 0.012 | – | 0.135 | 0.544 | 7.794 | 2 | 3 |
| T2 | 2 | 4 | orth-today | – | 0.0119 (0.0113–0.0120) | 0.012 | 0.013 | +0.0001 [+0.0000, +0.0002] 3/3 | 0.00917 | 0.294 | 7.927 | 2 | 3 |
| T2 | 2 | 4 | conf | 0.00722 | 0.0115 (0.0114–0.0119) | 0.012 | 0.012 | -0.0000 [-0.0004, +0.0002] 1/3 | 0.0098 | 0.083 | 7.521 | 2 | 3 |
| T2 | 2 | 4 | conf-tan | 0.00722 | 0.0117 (0.0113–0.0118) | 0.012 | 0.012 | -0.0002 [-0.0002, +0.0002] 1/3 | 0.065 | 0.402 | 11.337 | 2 | 3 |
| hill-lop | 2 | 8 | none | – | 0.0200 (0.0184–0.0295) | 0.02 | 0.022 | – | 0.286 | 0.502 | 12.75 | 2 | 3 |
| hill-lop | 2 | 8 | orth-today | – | 0.0198 (0.0198–0.0263) | 0.02 | 0.024 | +0.0013 [-0.0097, +0.0063] 2/3 | 0.065 | 0.276 | 9.983 | 2 | 3 |
| hill-lop | 2 | 8 | conf | 0.0781 | 0.0415 (0.0259–0.0441) | 0.044 | 0.051 | +0.0120 [+0.0059, +0.0257] 3/3 | 0.096 | 0.325 | 612.367 | 2 | 3 |
| hill-lop | 2 | 8 | conf-tan | 0.0781 | 0.0267 (0.0251–0.0482) | 0.03 | 0.049 | +0.0068 [-0.0044, +0.0297] 2/3 | 0.169 | 0.392 | 8.203 | 2 | 3 |
| hill-round | 2 | 8 | none | – | 0.0179 (0.0171–0.0193) | 0.018 | 0.018 | – | 0.413 | 0.494 | 8.408 | 2 | 3 |
| hill-round | 2 | 8 | orth-today | – | 0.0183 (0.0172–0.0215) | 0.018 | 0.019 | +0.0004 [+0.0000, +0.0021] 3/3 | 0.031 | 0.263 | 6.117 | 2 | 3 |
| hill-round | 2 | 8 | conf | 0.0861 | 0.0253 (0.0199–0.0506) | 0.026 | 0.027 | +0.0075 [+0.0028, +0.0313] 3/3 | 0.037 | 0.084 | 8.993 | 2 | 3 |
| hill-round | 2 | 8 | conf@0.1 | 0.00861 | 0.0185 (0.0172–0.0227) | 0.018 | 0.019 | +0.0006 [+0.0001, +0.0034] 3/3 | 0.029 | 0.132 | 7.138 | 2 | 3 |
| hill-round | 2 | 8 | conf@0.03 | 0.00258 | 0.0176 (0.0175–0.0246) | 0.017 | 0.018 | +0.0004 [-0.0003, +0.0052] 2/3 | 0.074 | 0.207 | 7.334 | 2 | 3 |
| hill-round | 2 | 8 | conf-tan | 0.0861 | 0.0209 (0.0189–0.0445) | 0.021 | 0.03 | +0.0030 [+0.0018, +0.0252] 3/3 | 0.047 | 0.27 | 12.403 | 2 | 3 |
| SPD2-airm | 3 | 3 | none | – | 0.0585 (0.0256–0.0717) | 0.06 | 0.07 | – | 0.647 | 0.725 | 338.241 | 2 | 3 |
| SPD2-airm | 3 | 3 | orth-today | – | 0.0390 (0.0328–0.0726) | 0.039 | 0.041 | +0.0134 [-0.0388, +0.0141] 2/3 | 0.126 | 0.646 | 16.706 | 2 | 3 |
| SPD2-airm | 3 | 3 | conf | 0.167 | 0.0717 (0.0702–0.0839) | 0.081 | 0.104 | +0.0254 [+0.0000, +0.0446] 3/3 | 0.312 | 0.505 | 622.454 | 2 | 3 |
| SPD2-airm | 3 | 3 | conf-tan | 0.167 | 0.0688 (0.0665–0.1083) | 0.074 | 0.094 | +0.0103 [-0.0051, +0.0827] 2/3 | 0.28 | 0.496 | 726.263 | 2 | 3 |
| SPD2-logeuclid | 3 | 3 | none | – | 0.0553 (0.0264–0.0689) | 0.055 | 0.058 | – | 0.554 | 0.739 | 518.228 | 2 | 3 |
| SPD2-logeuclid | 3 | 3 | orth-today | – | 0.0620 (0.0271–0.0667) | 0.069 | 0.085 | -0.0069 [-0.0282, +0.0402] 1/3 | 0.335 | 0.725 | 419.4 | 2 | 3 |
| SPD2-logeuclid | 3 | 3 | conf | 0.199 | 0.0783 (0.0734–0.0846) | 0.083 | 0.1 | +0.0293 [+0.0094, +0.0469] 3/3 | 0.245 | 0.538 | 687.871 | 2 | 3 |
| SPD2-logeuclid | 3 | 3 | conf-tan | 0.199 | 0.0783 (0.0734–0.0846) | 0.083 | 0.1 | +0.0293 [+0.0094, +0.0469] 3/3 | 0.245 | 0.538 | 687.871 | 2 | 3 |
| S2xR2 | 4 | 5 | none | – | 0.0426 (0.0382–0.0501) | 0.043 | 0.044 | – | 0.315 | 0.667 | 22.02 | 2 | 3 |
| S2xR2 | 4 | 5 | orth-today | – | 0.0427 (0.0413–0.0488) | 0.043 | 0.044 | +0.0002 [-0.0013, +0.0031] 2/3 | 0.122 | 0.595 | 22.379 | 2 | 3 |
| S2xR2 | 4 | 5 | conf | 0.0412 | 0.0464 (0.0459–0.0494) | 0.046 | 0.048 | +0.0038 [-0.0007, +0.0077] 2/3 | 0.048 | 0.417 | 16.224 | 2 | 3 |
| S2xR2 | 4 | 5 | conf@0.1 | 0.00412 | 0.0445 (0.0443–0.0487) | 0.045 | 0.046 | +0.0019 [-0.0013, +0.0061] 2/3 | 0.081 | 0.437 | 23.689 | 3 | 3 |
| S2xR2 | 4 | 5 | conf@0.03 | 0.00123 | 0.0433 (0.0420–0.0516) | 0.043 | 0.044 | +0.0015 [-0.0006, +0.0051] 2/3 | 0.159 | 0.558 | 23.636 | 2 | 3 |
| S2xR2 | 4 | 5 | conf-tan | 0.0412 | 0.0461 (0.0448–0.0494) | 0.046 | 0.053 | +0.0022 [-0.0007, +0.0079] 2/3 | 0.105 | 0.466 | 14.508 | 2 | 3 |
| S4cap | 4 | 5 | none | – | 0.0462 (0.0449–0.0497) | 0.046 | 0.048 | – | 0.731 | 0.747 | 38.522 | 3 | 3 |
| S4cap | 4 | 5 | orth-today | – | 0.0475 (0.0446–0.0485) | 0.047 | 0.048 | -0.0003 [-0.0022, +0.0023] 1/3 | 0.188 | 0.653 | 33.574 | 2 | 3 |
| S4cap | 4 | 5 | conf | 0.546 | 0.0623 (0.0619–0.0678) | 0.063 | 0.07 | +0.0157 [+0.0126, +0.0230] 3/3 | 0.044 | 0.172 | 9.672 | 2 | 3 |
| S4cap | 4 | 5 | conf@0.1 | 0.0546 | 0.0508 (0.0507–0.0549) | 0.051 | 0.052 | +0.0052 [+0.0046, +0.0058] 3/3 | 0.038 | 0.211 | 35.154 | 2 | 3 |
| S4cap | 4 | 5 | conf@0.03 | 0.0164 | 0.0492 (0.0435–0.0506) | 0.049 | 0.049 | +0.0008 [-0.0014, +0.0030] 2/3 | 0.058 | 0.33 | 39.709 | 2 | 3 |
| S4cap | 4 | 5 | conf-tan | 0.546 | 0.0614 (0.0593–0.0703) | 0.063 | 0.075 | +0.0131 [+0.0117, +0.0254] 3/3 | 0.169 | 0.246 | 37.622 | 2 | 3 |

Figure: `results/benchmarks/geometry/tier1/x1_penalty/figures/x1_penalty.png` (relative change of
the error per seed, and the conformality floor each arm reaches).

What it says:

- **The conformality penalty, at the calibrated weight, costs accuracy on every curved bed and does
  nothing on the flat one.** Round hill +0.0075 median (worse on 3 of 3 seeds), lopsided hill
  +0.012 (3/3), `S⁴` cap +0.016 (3/3), `S²×R²` +0.004 (2/3), SPD(2) under both metrics +0.03 to
  +0.05 (3/3); `T²` ±0.0004. The pruning curves move with it: the penalised charts are no cheaper at
  16 or 8 anchors than the free ones. Read with O1: the free encoder co-adapts the chart to the
  target, and a penalty that pulls the chart towards the geometry pays for it in the readout.
- **The penalty does what it says to the chart.** The conformality floor falls from 0.49 to 0.08
  on the round hill, from 0.75 to 0.17 on `S⁴` and from 0.67 to 0.42 on `S²×R²` (where the Weyl
  tensor forbids 0: the prediction holds, the global penalty cannot flatten a product), with the
  per-bed weights 0.04–0.5. The gap to G4's oracle floors (0 on the hills and `S⁴`) is optimisation,
  not obstruction (claim A4 fails at 400 epochs; X2's question).
- **Today's orthogonality penalty is neutral**: within ±0.002 of `none` on every bed but SPD(2),
  where it is the one arm that stabilises the learned chart (median 0.039 against 0.059, `‖A − I‖∞`
  17 against 340), at the price of one seed. The learned chart on SPD(2) is unstable in every other
  arm (`‖A − I‖∞` of 300–700): the chart problem of SPD(2) is a stability problem first.
- **The tangent projector changes little**: `conf-tan` tracks `conf` on every bed and coincides with
  it on the log-Euclidean twin (where `P = I` by construction, a check that passed). At native
  ambient dimension the off-manifold part of the Jacobian is not what hurts.
- **A weaker pull does not buy anything either (weight sweep).** On `S⁴` the cost falls from +0.016
  at the calibrated weight to +0.005 at 0.1× (still worse on 3/3 seeds) and to +0.001 at 0.03×
  (2/3), while the conformality floor the chart reaches climbs back from 0.17 to 0.21 and 0.33: the
  accuracy cost and the floor gain scale together, there is no weight at which the chart is more
  conformal at no cost. On `S²×R²` the same: +0.004 → +0.002 → +0.0015 with floors 0.42 → 0.44 →
  0.56 (against 0.67 free). On the round hill +0.0075 → +0.0006 (3/3) → +0.0004 (2/3) with floors
  0.08 → 0.13 → 0.21 (0.49 free): at 0.03× the penalty is within the seed noise of `none` on all
  three beds, and the chart it leaves is barely more conformal than the free one.

Consequence for the plan: proposals A and B do not improve IGL's readout at `N = 4096`, 64 anchors
and 400 epochs on any synthetic bed of the core set; the value of the conformality target, if any,
lies in the certificate (D) and in the operator side (E), not in regularising the learned chart.

## Scale: the chart proposals at application size (plan v2)

Kind: learned chart on real data (no analytic metric) and, in L1, on the synthetic ladder to `d = 64`. Plan v2
(2026-10-30) moves the question from "does a penalty help on small manifolds" (no: the anchor budget never binds
there, see X1) to "does anything about the chart matter where anchors bind": MNIST, DBpedia-14 sentence
embeddings, the GPT-2 token table, a Mac-sized GPT-2 hidden-state pilot. Primary pre-registered hypothesis H4:
the calibrated conformality penalty lowers the error at the binding anchor count. The rest (H0–H9) is in the plan
file and is read with Holm correction within a block.

### Protocol (v2)

- Inputs standardised per feature on the training split; the same tensors for every arm; splits seeded.
- `IGLModule(D, k, C)` with the default encoder (Stiefel init), `R` anchors × 4 scales, batch 256, a constant
  budget of 12 000 optimiser steps whatever `N` (50 epochs at 60k rows), constant lr 1e-3, clip 1.0, no early
  stop, best-validation state restored; `inner_batch_size = max(4096, 16 R)` so the training-time readout is
  never under-determined at large `R`; one-hot ridge readout with `CrossEntropyLoss` for classification.
- Anchors: k-means on the epoch-0 latents of 16k training rows (identical across arms sharing a seed),
  scale ladder kept (`kmeans-keep`); `random` and `kmeans-median` as the S1 controls.
- Penalties: `orth-today` is the package's `OrthogonalityPenalty(0.1, every=20)`; `conf` is the axis-normalised
  conformality penalty on all prefixes every 2 batches, its weight calibrated per (set, R) so that the penalty
  gradient is 0.3 of the task gradient of the trainer's own step at 10 % of the budget of a penalty-free fit;
  the realised ratio is logged at 10 / 50 / 100 %.
- Scoring: readout solved on the full training split (chunked Φ), error rate on the test split; the wrong rows
  are kept, so two arms sharing a seed are compared on the same rows (pooled paired bootstrap, McNemar).
  Thresholds come from the measured seed SD of `none` (W1): the minimum detectable paired effect is quoted as
  3.6 SD at three seeds and 1.9 SD at five.
- Baselines on the same split and seeds: linear probe, kNN, the capacity-matched MLP head (IGL's encoder →
  `Linear(k, R) → SiLU → Linear(R, C)`, same budget), and the frozen PCA-`k` chart through IGL's own kernel and
  readout (identity encoder, k-means anchors): the control that isolates what the *learned* chart adds.
- Every fit records the geometry report on the top-64 PCA directions as the tangent proxy, `d_eff`, snapshots,
  Φ column CV and effective rank, and a stability flag (final `√L_conf` above its epoch-0 value, or a median
  condition number of `JJᵀ` above 1e4).

### W1. The week-1 gate (MNIST)

_Pending: the noise floor, the frozen PCA chart against the learned chart at R ∈ {32, 512}, the inner-solve and
ridge checks. The block is written; the run starts once the machine is free._

