# IGL examples

Standalone scripts that exercise the public IGL API on synthetic data —
nothing external is required.

## How to run

```bash
uv sync --group dev
source .venv/bin/activate
python -m examples.synthetic.torus_classification
```

Each script writes its output to `results/<example>/<git_short_sha>/`:

| File | Content |
|---|---|
| `history.json` | Per-epoch training history (train/val loss, val metric, truncation k). |
| `curve.csv` | Two-column dimension/loss curve from `igl.eval_dimension_curve`. |

If you install the `[viz]` extra (`pip install -e ".[viz]"`), future
examples will additionally produce a PNG plot of the dimension curve. The
viz module is not implemented yet in this milestone.

## v0.1 examples

| Script | Manifold | Tasks | Expected outcome |
|---|---|---|---|
| `synthetic/torus_classification.py` | Flat torus T² in R⁴ → R³² | XOR cls + sin/cos reg | d_eff ≈ 2 |
| `synthetic/moons_xor.py` | Moons in R² → R^16 | cls + reg + recon | d_cls ≤ d_reg ≤ d_recon |
| `synthetic/swiss_roll_recon.py` | Swiss roll in R³ | autoencoder + reg | d_eff ≈ 2 |
| `synthetic/poisson_1d.py` | $-u'' = f$ on $[0, 1]$ (Neumann) | regression | spectral + null-space recovers the DC mode |
