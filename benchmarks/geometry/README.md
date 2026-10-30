# Geometry benchmark suite

The experiments behind the research deck on the chart in IGL (conformality instead of
orthogonality, separability of the conformal factor, the operator side, the geometry report,
conformally flat IGL), each measured **against IGL today** on the same data, sizes, seeds and
trainer. Run manually, never in CI; `REPORT.md` holds the results.

Two kinds of results, never mixed: **analytic/grid** blocks (no training; the geometry is known)
and **learned-chart** blocks (IGL trained). Every table names its kind.

## Layout

- `testbeds/`: the manifold matrix. `hills` (the deck's two 2-D hills, exact), `graphs` (graph
  hypersurfaces in `d = 2, 3, 4`, matrix-free solver), `manifolds` (sphere caps, cylinders, products,
  ellipsoids, flat tori, product of hills, hyperbolic half-space; Weyl and Cotton tensors by nested
  autograd), `spd_beds` (SPD(n) under the affine-invariant and log-Euclidean metrics), `targets`
  (grid Poisson, eigenfunction and graph-Laplacian targets), `package_sets` (embedding into higher
  ambient dimension, the package's own sets).
- `testbeds/real.py`: the real sets of plan v2 (MNIST, Fashion-MNIST, DBpedia-14 MiniLM embeddings, the GPT-2
  token table, a GPT-2-small hidden-state pilot), downloaded once and cached under `IGL_DATA_CACHE`
  (default `~/igl_data_cache`), standardised on the training split.
- `geometry_report.py`: proposal D's report from a Jacobian. `charts.py`: chart-only cost measures.
- `fitting.py`: the learned-chart protocol (one fit per set, arm and seed; regression and classification);
  `baselines.py`: the linear probe, kNN, the capacity-matched MLP head and the Nyström reference.
- `blocks/`: one script per block of the plan (`r0_…`, `g0_…`, `o1_…`, `x1_…` for the synthetic tiers;
  `w1_gate`, `s0_baselines`, `s1_anchor_budget`, `s2_penalty_at_scale`, `s3_learning_curve` for the scale
  tier of plan v2), each with `--smoke`, `--seeds`, `--device`, `--label`; the scale blocks add `--sets`,
  `--k`, `--anchors`, `--anchor-init`, `--steps`. Results as JSON under
  `results/benchmarks/geometry/<label>/<block>/<bed-or-set-cell>/<arm>/seed<k>.json`, figures next to them.
- `report.py` (+ `report_scale.py`): the Markdown tables and figures of every block, paired tests included.

## Running

```bash
uv run python -m benchmarks.geometry.blocks.g0_geometry --smoke      # under a minute
IGL_BENCH_LABEL=run1 uv run python -m benchmarks.geometry.blocks.g1_rho_law
make geometry-tier0                                                   # R0, G0, G1, G2, G4
uv run python -m benchmarks.geometry.blocks.g4_oracle_floor --beds hillxhill,S2xR2   # one bed or a few
make geometry-scale-smoke                                             # W1, S0-S3 on a synthetic stand-in
make geometry-w1 SETS=mnist                                           # the week-1 gate (MPS by default)
make geometry-s2 SETS=mnist R_BIND=32                                 # after S1 named the binding R
uv run python -m benchmarks.geometry.report --label scale --blocks w1_gate,s1_anchor_budget --figures
```

The machine-state gate of the device suite applies to learned-chart blocks (quiet machine, load
average below 6); analytic blocks run anywhere.
