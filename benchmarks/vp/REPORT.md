# VP solver verification — findings

Exploratory wave (seeds {42,123,456} for training cards; median-of-5 timings
for microbenchmarks; CPU / Apple M4). Every number here is reproduced by a
JSON under `results/benchmarks/vp/<experiment>/<sha>/`. Nothing is a
confirmed claim until a follow-up wave re-runs the headline numbers under a
fuller seed protocol.

The program verifies a set of claims from a design discussion about IGL's
variable-projection (VP) inner/outer loops. The one-line summary: **the
mathematics (envelope equivalence, determinism, √κ conditioning) all holds;
the cost story holds with sharper numbers than predicted; and two of the
"acceleration" predictions were refuted in a useful way.** The MHALM /
IGL-AIRM workloads then reframe what actually matters for throughput.

## Verdict table

| # | Claim | Verdict | Measured |
|---|---|---|---|
| P1 | Envelope-gradient bias is O(inner suboptimality) | **confirmed** | log-log slope 1.19 (moons), 1.22 (swiss-roll); target [0.7,1.3] |
| P2 | Converged-CG training ≡ exact VP | **confirmed** | mean final-val gap 5e-5 / 7e-5, inside seed noise |
| P3 | direct→CG crossover in [8k,40k] | **confirmed** | p\* = **10,240** (n=4096, D=768) |
| P4 | 48 GB Gram memory wall in [30k,80k] | **confirmed** | extrapolated p ≈ **72,000** |
| P5 | CG ~ √κ, GD ~ κ | **confirmed** | CG 195 vs pred 239 at κ≈5e3; GD 349→9,306 over κ grid |
| P6 | warm start ≥3× fewer CG iters | **partial** | median **2.4×** at per-step drift; below the 3× bar |
| P7a | an accelerated outer arm ≥2× faster to baseline val loss | **confirmed** | full-batch L-BFGS **32× / 34×** (D=64 / D=512) |
| P7b | safeguarded Anderson never worse than baseline | **refuted** | AA never *beat* baseline on any bed; safe but not helpful |
| P8 | direct solve bit-identical across repeats | **confirmed** | SHA-256 of full state-dict identical |
| P9 | effective-dimension readout solver-invariant | **confirmed** | same-module curves agree 0.02%; elbow & knee identical |
| P10 | true-CE inner ≤ one-hot-LS surrogate on CE | **confirmed** | CE 0.21 vs 0.82 (four-blobs) at equal accuracy |
| P11 | Nyström-PCG κ-independent (rank p/10) | **refuted** | iters 32→225 across κ grid on log-spaced spectra |

## What each experiment established

**E1 / E5 — the core mathematics is exact.** The envelope (Danskin) theorem
holds numerically: the outer-gradient error scales linearly with the
*achieved* inner suboptimality (not the nominal tolerance — loose tolerances
saturate the gradient direction and fall outside the O(ε) law, which is why
the slope is fit against realized w-error). Training with CG converged to
1e-6 reproduces exact-VP training to within seed noise. Direct solves are
bit-identical across repeats; iterative solves fluctuate at tolerance level.
Consequence: **solver choice is a pure cost/precision decision — it cannot
change where training converges**, only how fast and how reproducibly.

**E2 — the crossover, with numbers (n=4096, D=768, tol 1e-5):**

| p | package lstsq | Cholesky | matrix-free CG (iters) |
|---|---|---|---|
| 2,048 | 0.37 s | **0.06 s** | 0.43 s (32) |
| 4,096 (=n) | 1.6 s | **0.21 s** | 5.9 s (280) |
| 8,192 | 6.5 s | **0.84 s** | 1.5 s (32) |
| 16,384 | 49 s | 4.3 s | **1.7 s** (17) |
| 24,576 | 152 s | 12 s | **2.0 s** (13) |
| 32,768 | 344 s | 26 s | **2.5 s** (12) |

- Crossover vs the package's stacked-QR solver: **p ≈ 8k**. Vs the best exact
  solver (Cholesky): **p ≈ 16k**. Cost-model fit gives p\* = 10,240.
- The crossover is *earlier* than the discussion's 15–25k guess because real
  CG iteration counts are 12–32, not the assumed 100–200: once p > n, the
  ridge collapses the ΦᵀΦ spectrum into an n-cluster plus a huge λ-cluster
  that CG treats as one eigenvalue, so **larger dictionaries are easier for
  CG**.
- **The square case p ≈ n is pathological for CG** (280 iterations at
  p=n=4096, confirmed again at p=n=16384). Design rule: keep the dictionary
  size away from the solve-batch size.
- Memory wall extrapolates to p ≈ 72k on 48 GB.
- Real IGL design matrices (κ~1e6) need 173 CG iterations vs ~17 for benign
  iid matrices of the same size — conditioning, not size, is the binding
  constraint for kernel Φ (see E3).

**E3 — conditioning, and the whitening question resolved.** Formal condition
numbers of kernel Φ reach 1e10 at 8k anchors, yet CG converges in hundreds
of iterations, not the ~700k √κ predicts: the spectrum clusters above the
ridge floor, so κ over-predicts cost for kernel dictionaries. **Softmax
normalization needs 2–3× fewer CG iterations than NW** at equal κ (a free
config win). On whitening: with **per-column** relative CG stopping, raw /
logit / Fisher / damped targets all need the *same* inner tolerance — the
whitener's rescaling is normalized away. The feared "whitened targets are
tolerance-hungry" pathology appears only under an **aggregate** stopping
rule (Fisher then needs 10× tighter tolerance). Design rule for any iterative
backend: **stop per column, and whitening costs nothing.**

**E4 — outer acceleration is a two-sided result.** Full-batch L-BFGS on the
reduced functional reaches the minibatch baseline's final validation loss
**32× (D=64) and 34× (D=512) faster**, at a *better* optimum — and the win
does not decay with output width (the large-RHS / LLM regime). But **every
Anderson/RNA arm failed to beat its baseline** (EMA-minibatch and
deterministic-full-batch alike); the safeguard keeps AA *safe* (jumps get
rejected) but it is not *helpful* here. On saturated classification (moons,
already at the Bayes floor) nothing beats Adam — the discriminator is
**saturation, not task type**. So claim 4 splits: the L-BFGS half is strongly
confirmed (2–5× predicted, 32× measured); the Anderson half is refuted at
these scales. Caveat: full-batch L-BFGS assumes a bounded outer loop and does
not transfer to a streaming corpus (see Throughput).

**E6 — the instrument is solver-invariant.** Reading one fitted module's
dimension curve with the direct solve vs CG(1e-8) gives curves agreeing to
0.02%, with identical detected effective dimension and identical knockout
knee across all seeds and beds. (The first version compared two
independently-*trained* models and spuriously failed on the flat tail — that
was re-measuring E1's trajectory chaos, not the solver.)

**E7 — generalized VP is a real, shippable lever.** A true cross-entropy
inner problem (convex in w, solved by L-BFGS, envelope outer gradient) beats
the one-hot least-squares surrogate on cross-entropy at equal-or-better
accuracy every time: four-blobs CE 0.21 vs 0.82 at 0.94 vs 0.94 accuracy;
moons CE ≈0 vs 0.31 at 100% vs 100%. The surrogate reaches the right
*decision* but never sharpens *probabilities*, and the entire cost is paid in
calibration — i.e. in perplexity / bpb.

## Throughput: what actually matters for the real workloads

Reading MHALM (`laplacian/golf-competition/mhalm/vp.py`) and IGL-AIRM
(`paper-igl-airm/`) reframes the program. The six claims were about the
*linear solver*, but the production workloads are not p→∞ dense solves; they
are small-dictionary, huge-RHS, streaming or SPD problems. The real levers:

1. **CUDA-resident Cholesky.** Both MHALM's `RLSBuffer.solve()` and the igl
   package copy Σ, c to CPU for the factorization ("MPS/CUDA lstsq
   unreliable"), a hard serialization point that idles the GPU. At R≈128–256
   the factorization is tiny; the fix is a device-resident `cholesky` +
   `cholesky_solve` (CUDA-native, GEMM-shaped), fused into the forward graph.
   E2's 8× Cholesky-over-lstsq win is the same lesson. **Candidate PR.**

2. **Streaming sufficient statistics (RLS).** Σ = ΦᵀΦ + λI is R×R,
   *independent of vocab V and of token count*; only c = ΦᵀY is R×V. Scaling
   MHALM to a 50k vocabulary is a non-event for the solve — one R×R
   factorization reused across all 50k right-hand sides (a GEMM
   back-substitution). The corpus/vocab axes are the *friendly* ones; this is
   why the "small dictionary, huge RHS" corner is the correct scaling
   strategy and why the p→∞ crossover is a research-future question, not a
   ship-now one.

3. **Newton-Schulz matrix functions.** IGL-AIRM's cost is not the linear
   solve (n small, p=128, D≈250–2000 — all milliseconds) but the per-matrix
   `eigh` in `AIRMLoss`, whose backward produces NaN gradients on
   ill-conditioned EEG SPDs (forcing the `skip_failing_batches` guard).
   Replacing eigh-based matrix sqrt/inv-sqrt with a Newton-Schulz iteration is
   GEMM-only, differentiable, and NaN-free — the *same* primitive MHALM's blog
   reports as its single biggest fix, and the one Muon uses. Shared GPU
   primitive across encoder, SPD loss, and optimizer. **Candidate PR.**

## Follow-up backlog (not run this wave)

- **E4 hybrid** (explore-then-polish: minibatch Adam picks the basin, L-BFGS
  polishes) — tests whether moons' non-acceleration is basin selection;
  results in `e4_outer_acceleration` hybrid pass.
- **E9 forgetting-factor RLS** — γ-decayed Σ, c instead of per-cycle reset, so
  every batch feeds both statistics and gradient (the RLS analog of the E2
  warm-start effect). Prototype against `mhalm/` on tiny-shakespeare.
- **Dictionary scaling** (E8 issue): residual-driven atom growth vs multi-head
  atlas decomposition — two ways to keep every head on the cheap side of the
  crossover.
- Sketch-preconditioned solvers beyond Nyström-rank-p/10 (P11 refutation
  suggests larger sketches / different sketch for log-spaced spectra).

## Upstream candidates for the igl package

- Conditioned **Cholesky fast-path** in the solver (8× at p≥1k when
  well-conditioned; QR fallback otherwise). E2.
- **CUDA-resident** solve path (no CPU hop). Throughput §1.
- **block-CG inner backend** for p ≳ 10k, per-column stopping, warm-started
  across outer steps. E2/E3.
- **full-batch L-BFGS outer mode** for bounded-data regression/distillation
  fits (IGLDistiller). E4.
- **true-CE inner option** for classification/LM readouts (better bpb at equal
  accuracy). E7.
- **Newton-Schulz** matrix-function path in `igl.spd`. Throughput §3.
