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
- `geometry_report.py`: proposal D's report from a Jacobian. `charts.py`: chart-only cost measures.
- `blocks/`: one script per block of the plan (`r0_…`, `g0_…` … `x9_…`), each with `--smoke`,
  `--seeds`, `--device`, `--label`; results as JSON under
  `results/benchmarks/geometry/<label>/<block>/<bed>/<arm>/seed<k>.json`, figures next to them.

## Running

```bash
uv run python -m benchmarks.geometry.blocks.g0_geometry --smoke      # under a minute
IGL_BENCH_LABEL=run1 uv run python -m benchmarks.geometry.blocks.g1_rho_law
make geometry-tier0                                                   # R0, G0, G1, G2, G4
```

The machine-state gate of the device suite applies to learned-chart blocks (quiet machine, load
average below 6); analytic blocks run anywhere.
