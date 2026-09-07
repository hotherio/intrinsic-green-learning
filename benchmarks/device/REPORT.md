# Device backends: measurements

Everything in this report comes from the committed harness under `benchmarks/device/`,
run through `run_matrix.sh` so that every library version is measured by the same code
from the same checkout. Results are filed under `results/benchmarks/device/<label>/<device>/`
(gitignored) and assembled by `python -m benchmarks.device.report`.

Versions measured:

| label | what |
|---|---|
| `6c66262` | v0.13.0, the baseline before this work |
| `34d11bd` | item 1: per-device backends, sync-free device readout solve |
| `620cf27` | item 2: contracted fast kernel path, exact sign skipping |
| `68fa992` | item 3: one host transfer per epoch |
| `4823d23` | item 5: spectral gate blend on the device |
| `b03bb29` | item 6: vectorised Jacobian for the orthogonality penalty |
| `c0f9ced` | items 7–9 plus the whitener, SPD unpack and spectral-index fixes found by the census |

Machines: Apple M4 Max (CPU and MPS, torch 2.12, Python 3.14) and a RunPod H100 80 GB
(torch 2.8.0+cu128, Python 3.12). Timing runs only count when the machine is quiet
(load gate, machine state recorded in every JSON); synchronisation counts do not depend
on load and were taken while the H100 was shared with another job.

## Host synchronisations on CUDA (the property the work is about)

Counted with `torch.cuda.set_sync_debug_mode("warn")` around a two-epoch fit of the
`small` problem (1024 rows, 8 batches per epoch), warnings tallied by the Python call
site; the modules are built before the measured region. `_backend.py:117` is the one
sanctioned transfer per epoch (`host_scalars`). `nn/modules/module.py:1355` in the
estimator rows is the parameter upload when the estimator builds its module inside
`fit` (one copy per parameter, once per fit, not a training synchronisation).

Per-label summary (`cuda sync warnings/epoch`; the wide table with the profiler op counts
and the top call sites is the `report.py` output below):

| workload | 6c66262 | 34d11bd | 620cf27 | 68fa992 | 4823d23 | b03bb29 | c0f9ced |
|---|---|---|---|---|---|---|---|
| classifier (cross-entropy) | 66 | 11 | 11 | 1 | 1 | 1 | **1** |
| regressor (MSE) | 66 | 11 | 11 | 1 | 1 | 1 | **1** |
| spectral kernel (cosine basis) | 259 | 204 | 204 | 194 | 217 | 217 | **1** |
| orthogonality penalty | 72 | 17 | 17 | 7 | 7 | 1 | **1** |
| AIRM loss, `eigh` (default) | 152 | 97 | 97 | 89 | 89 | 89 | 33 |
| AIRM loss, `matrix_method="iterative"` | n/a | n/a | n/a | n/a | n/a | n/a | **1** |
| `IGLAutoencoder.fit` (estimator, includes 8 parameter uploads) | 84.5 | 25.5 | 25.5 | 16.5 | 16.5 | 16.5 | 15 |
| `IGLDistiller.fit` (estimator, includes 8 parameter uploads) | failed | failed | failed | failed | failed | failed | 16.5 |

| device | commit | workload | scalar_dense/epoch | to_copy/epoch | memcpy DtoH/epoch | cuda sync warnings/epoch | sites (per epoch) |
|---|---|---|---|---|---|---|---|
| cuda | 6c66262 | classifier_ce | 314.00 | 99.00 | 181.50 | 66.00 | igl/core/solver.py:86 (18), igl/core/solver.py:36 (18), igl/core/solver.py:110 (9) |
| cuda | 6c66262 | regressor_mse | 314.00 | 80.00 | 182.50 | 66.00 | igl/core/solver.py:86 (18), igl/core/solver.py:36 (18), igl/core/solver.py:110 (9) |
| cuda | 6c66262 | spectral_cosine | 330.00 | 194.00 | 247.00 | 259.00 | igl/spectral/kernel.py:278 (64), igl/spectral/kernel.py:243 (43), igl/spectral/kernel.py:244 (43) |
| cuda | 6c66262 | orthogonality | 320.00 | 80.00 | 182.00 | 72.00 | igl/core/solver.py:86 (18), igl/core/solver.py:36 (18), igl/core/solver.py:110 (9) |
| cuda | 6c66262 | airm | 344.00 | 95.00 | 304.50 | 152.00 | igl/spd/linalg.py:71 (48), igl/spd/linalg.py:30 (32), igl/core/solver.py:86 (18) |
| cuda | 6c66262 | airm_iterative | - | - | - | - | skipped: TypeError("AIRMLoss.__init__() got an unexpected keyword arg |
| cuda | 6c66262 | autoencoder_estimator | 290.00 | 97.50 | 199.00 | 84.50 | igl/core/solver.py:86 (20), igl/core/solver.py:36 (20), igl/core/solver.py:110 (10) |
| cuda | 6c66262 | distiller_estimator | - | - | - | - | skipped: RuntimeError('Expected all tensors to be on the same device, |
| cuda | 34d11bd | classifier_ce | 27.00 | 64.00 | 11.00 | 11.00 | igl/core/trainer.py:503 (8), igl/core/_backend.py:117 (1), igl/core/trainer.py:595 (1) |
| cuda | 34d11bd | regressor_mse | 27.00 | 45.00 | 11.00 | 11.00 | igl/core/trainer.py:503 (8), igl/core/_backend.py:117 (1), igl/core/trainer.py:595 (1) |
| cuda | 34d11bd | spectral_cosine | 91.00 | 159.00 | 75.00 | 204.00 | igl/spectral/kernel.py:278 (64), igl/spectral/kernel.py:243 (43), igl/spectral/kernel.py:244 (43) |
| cuda | 34d11bd | orthogonality | 33.00 | 45.00 | 17.00 | 17.00 | igl/core/trainer.py:503 (8), igl/spd/orthogonality.py:140 (6), igl/core/_backend.py:117 (1) |
| cuda | 34d11bd | airm | 57.00 | 60.00 | 97.00 | 97.00 | igl/spd/linalg.py:71 (48), igl/spd/linalg.py:30 (32), torch/autograd/graph.py:829 (8) |
| cuda | 34d11bd | airm_iterative | - | - | - | - | skipped: TypeError("AIRMLoss.__init__() got an unexpected keyword arg |
| cuda | 34d11bd | autoencoder_estimator | 30.00 | 56.50 | 14.00 | 25.50 | nn/modules/module.py:1355 (8), igl/core/trainer.py:503 (7), igl/core/loss.py:81 (3) |
| cuda | 34d11bd | distiller_estimator | - | - | - | - | skipped: RuntimeError('Expected all tensors to be on the same device, |
| cuda | 620cf27 | classifier_ce | 27.00 | 30.00 | 11.00 | 11.00 | igl/core/trainer.py:503 (8), igl/core/_backend.py:117 (1), igl/core/trainer.py:595 (1) |
| cuda | 620cf27 | regressor_mse | 27.00 | 11.00 | 11.00 | 11.00 | igl/core/trainer.py:503 (8), igl/core/_backend.py:117 (1), igl/core/trainer.py:595 (1) |
| cuda | 620cf27 | spectral_cosine | 91.00 | 159.00 | 75.00 | 204.00 | igl/spectral/kernel.py:278 (64), igl/spectral/kernel.py:243 (43), igl/spectral/kernel.py:244 (43) |
| cuda | 620cf27 | orthogonality | 33.00 | 11.00 | 17.00 | 17.00 | igl/core/trainer.py:503 (8), igl/spd/orthogonality.py:140 (6), igl/core/_backend.py:117 (1) |
| cuda | 620cf27 | airm | 57.00 | 27.00 | 97.00 | 97.00 | igl/spd/linalg.py:71 (48), igl/spd/linalg.py:30 (32), torch/autograd/graph.py:829 (8) |
| cuda | 620cf27 | airm_iterative | - | - | - | - | skipped: TypeError("AIRMLoss.__init__() got an unexpected keyword arg |
| cuda | 620cf27 | autoencoder_estimator | 30.00 | 22.50 | 14.00 | 25.50 | nn/modules/module.py:1355 (8), igl/core/trainer.py:503 (7), igl/core/loss.py:81 (3) |
| cuda | 620cf27 | distiller_estimator | - | - | - | - | skipped: RuntimeError('Expected all tensors to be on the same device, |
| cuda | 68fa992 | classifier_ce | 16.00 | 39.00 | 1.00 | 1.00 | igl/core/_backend.py:117 (1) |
| cuda | 68fa992 | regressor_mse | 16.00 | 20.00 | 1.00 | 1.00 | igl/core/_backend.py:117 (1) |
| cuda | 68fa992 | spectral_cosine | 80.00 | 168.00 | 65.00 | 194.00 | igl/spectral/kernel.py:278 (64), igl/spectral/kernel.py:243 (43), igl/spectral/kernel.py:244 (43) |
| cuda | 68fa992 | orthogonality | 22.00 | 20.00 | 7.00 | 7.00 | igl/spd/orthogonality.py:140 (6), igl/core/_backend.py:117 (1) |
| cuda | 68fa992 | airm | 48.00 | 34.00 | 89.00 | 89.00 | igl/spd/linalg.py:71 (48), igl/spd/linalg.py:30 (32), torch/autograd/graph.py:829 (8) |
| cuda | 68fa992 | airm_iterative | - | - | - | - | skipped: TypeError("AIRMLoss.__init__() got an unexpected keyword arg |
| cuda | 68fa992 | autoencoder_estimator | 20.00 | 30.50 | 5.50 | 16.50 | nn/modules/module.py:1355 (8), igl/core/solver.py:188 (2), igl/core/loss.py:89 (2) |
| cuda | 68fa992 | distiller_estimator | - | - | - | - | skipped: RuntimeError('Expected all tensors to be on the same device, |
| cuda | 4823d23 | classifier_ce | 16.00 | 39.00 | 1.00 | 1.00 | igl/core/_backend.py:117 (1) |
| cuda | 4823d23 | regressor_mse | 16.00 | 20.00 | 1.00 | 1.00 | igl/core/_backend.py:117 (1) |
| cuda | 4823d23 | spectral_cosine | 16.00 | 255.00 | 1.00 | 217.00 | igl/spectral/kernel.py:243 (72), igl/spectral/kernel.py:244 (72), igl/spectral/kernel.py:245 (72) |
| cuda | 4823d23 | orthogonality | 22.00 | 20.00 | 7.00 | 7.00 | igl/spd/orthogonality.py:140 (6), igl/core/_backend.py:117 (1) |
| cuda | 4823d23 | airm | 48.00 | 34.00 | 89.00 | 89.00 | igl/spd/linalg.py:71 (48), igl/spd/linalg.py:30 (32), torch/autograd/graph.py:829 (8) |
| cuda | 4823d23 | airm_iterative | - | - | - | - | skipped: TypeError("AIRMLoss.__init__() got an unexpected keyword arg |
| cuda | 4823d23 | autoencoder_estimator | 20.00 | 30.50 | 5.00 | 16.50 | nn/modules/module.py:1355 (8), igl/core/solver.py:188 (2), igl/core/loss.py:89 (2) |
| cuda | 4823d23 | distiller_estimator | - | - | - | - | skipped: RuntimeError('Expected all tensors to be on the same device, |
| cuda | b03bb29 | classifier_ce | 16.00 | 39.00 | 1.00 | 1.00 | igl/core/_backend.py:117 (1) |
| cuda | b03bb29 | regressor_mse | 16.00 | 20.00 | 1.00 | 1.00 | igl/core/_backend.py:117 (1) |
| cuda | b03bb29 | spectral_cosine | 16.00 | 255.00 | 1.00 | 217.00 | igl/spectral/kernel.py:243 (72), igl/spectral/kernel.py:244 (72), igl/spectral/kernel.py:245 (72) |
| cuda | b03bb29 | orthogonality | 16.00 | 20.00 | 1.00 | 1.00 | igl/core/_backend.py:117 (1) |
| cuda | b03bb29 | airm | 48.00 | 34.00 | 89.00 | 89.00 | igl/spd/linalg.py:71 (48), igl/spd/linalg.py:30 (32), torch/autograd/graph.py:829 (8) |
| cuda | b03bb29 | airm_iterative | - | - | - | - | skipped: TypeError("AIRMLoss.__init__() got an unexpected keyword arg |
| cuda | b03bb29 | autoencoder_estimator | 20.00 | 30.50 | 5.50 | 16.50 | nn/modules/module.py:1355 (8), igl/core/solver.py:188 (2), igl/core/loss.py:89 (2) |
| cuda | b03bb29 | distiller_estimator | - | - | - | - | skipped: RuntimeError('Expected all tensors to be on the same device, |
| cuda | c0f9ced | classifier_ce | 16.00 | 39.00 | 1.00 | 1.00 | igl/core/_backend.py:117 (1) |
| cuda | c0f9ced | regressor_mse | 16.00 | 20.00 | 1.00 | 1.00 | igl/core/_backend.py:117 (1) |
| cuda | c0f9ced | spectral_cosine | 16.00 | 39.00 | 1.00 | 1.00 | igl/core/_backend.py:117 (1) |
| cuda | c0f9ced | orthogonality | 16.00 | 20.00 | 1.00 | 1.00 | igl/core/_backend.py:117 (1) |
| cuda | c0f9ced | airm | 48.00 | 34.00 | 33.00 | 33.00 | igl/spd/linalg.py:53 (32), igl/core/_backend.py:117 (1) |
| cuda | c0f9ced | airm_iterative | 16.00 | 34.00 | 1.00 | 1.00 | igl/core/_backend.py:117 (1) |
| cuda | c0f9ced | autoencoder_estimator | 18.00 | 31.00 | 3.50 | 15.00 | nn/modules/module.py:1355 (8), igl/core/solver.py:188 (2), igl/models/_base.py:292 (1) |
| cuda | c0f9ced | distiller_estimator | 20.50 | 36.50 | 3.50 | 16.50 | nn/modules/module.py:1355 (8), igl/core/solver.py:188 (2), igl/whitening/whitener.py:85 (1.5) |

Reading the columns: the baseline's 66 on the classifier are the readout solve
(`isfinite` guards, `col_scale.item()`, the CPU round trip: `solver.py:36/86/110/135`)
plus `float(task_loss.item())` per batch (`trainer.py`); item 1 removes the solve's
share, item 3 the per-batch loss read, leaving the one sanctioned transfer. The spectral
kernel's remaining 194–217 after item 5 were the kept-mode index lists uploaded three
times per forward (`kernel.py:243–245`); item 5 itself traded the 64 mask reads for
calling `_factor` on every dimension, which is why the count rises slightly at `4823d23`
before the buffers remove it. The orthogonality penalty's 6 extra per epoch were the
`int(gate_mask.sum().item())` reads (item 6). The AIRM loss keeps 4 `eigh` calls per
batch (`linalg.py:53`) with the default method; the census also caught the boolean-mask
indexing in `unpack_sym_vec` (6 per batch, gone at `c0f9ced`). `IGLDistiller` could not
train on CUDA or MPS at all before the whitener fix (its constants stayed on the CPU); the
estimator rows include the eight parameter uploads of module construction inside `fit`
and the post-fit reads (`d_eff`, dimension curve, the final solve's failure flag).

Which primitives synchronise on CUDA (H100, torch 2.8, one call each, measured with the
same debug mode; used to choose the replacements):

| primitive | syncs | used as |
|---|---|---|
| `torch.linalg.eigh` | 1 | AIRM default; documented exception |
| `torch.linalg.matrix_exp` | 2 forward, 2 backward | replaced by a fixed-order Taylor scaling-and-squaring |
| boolean-mask indexing (`t[mask]`) | 1 (`nonzero`) | replaced by a triangular transpose in `unpack_sym_vec` |
| `t[:, [i, j]]` or `t[:, idx]` | 1 | replaced by `index_select` on a device buffer |
| `torch.linalg.cholesky_ex` + `cholesky_solve` | 0 | the device readout solve |
| `torch.linalg.inv_ex`, `solve_ex` | 0 | Denman–Beavers roots, `atanh` series |
| `index_select`, `index_put_`, `triu_indices(device=cuda)`, `eye(device=cuda)` | 0 | |

## Accuracy of the sync-free replacements

- Device readout solve (Cholesky of `ΦᵀΦ + λI` plus one refinement step) versus the CPU
  `lstsq`: relative error 3.5e-6 to 1.25e-4 across the three problem sizes, at or below
  `lstsq`'s own 1.4e-6 to 6e-5 (`components.py`; the test suite pins < 1e-4 on predictions).
- Fast kernel path versus the reference formulation: values and gradients agree to 1e-6,
  gradcheck in float64; the reference path is bit-identical to v0.13.0 (`tests/test_kernel_paths.py`).
- Iterative SPD functions versus fp64 `eigh` on EEG-like spectra spanning six orders of
  magnitude (`d` = 16): logarithm 1.3e-4, inverse square root 1.4e-3, exponential 1e-6 to
  4e-6 on symmetric spectra in [-40, 40]; fp32 `eigh` itself sits at 7.4e-4 (log) and
  6e-6 (exp) on the same inputs. Neither reaches the 1e-5 the plan hoped for in fp32, so
  `eigh` stays the default and the iterative method is opt-in.

## Per-batch regions (ms per batch, medians)

<!-- TIMING:regions -->

## End-to-end examples (wall seconds and headline outputs)

<!-- TIMING:e2e -->

## Op-level profile

<!-- TIMING:profile -->

## Component benchmarks

<!-- TIMING:components -->

## Status of the CUDA timing rows

The H100 is shared with a training chain that occupies the GPU until it drains; the
synchronisation census above does not depend on that, the timing rows do. They are
produced by the same driver (`IGL_BENCH_DEVICE=cuda PY=python3 benchmarks/device/run_matrix.sh
6c66262=/workspace/igl-base/src ... c0f9ced=/workspace/igl-src/src`) and will be added to
this report when the GPU is free.
