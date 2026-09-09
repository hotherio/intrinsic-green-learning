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
| `2259436` | as above plus the MPS eigensolver fallback (the library measured as "HEAD" below) |
| `08d5efc` | eager follow-up after the profile: gate masks from a per-epoch table, no inner permutation on device branches |
| `4866d9e` | the batch step replayed from a CUDA graph on the CUDA branch (`cuda_graphs`, default on), optional `torch_compile` fusion; the final library in this change |

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

On MPS the same census counts `aten::_local_scalar_dense` (every `.item()` is a host
synchronisation on Apple silicon too) and `aten::_to_copy`; the sync-debug counter is CUDA
only:

| device | commit | workload | scalar_dense/epoch | to_copy/epoch | memcpy DtoH/epoch | cuda sync warnings/epoch | sites (per epoch) |
|---|---|---|---|---|---|---|---|
| mps | 6c66262 | classifier_ce | 242.00 | 749.00 | 0.00e+00 | 0.00e+00 |  |
| mps | 6c66262 | regressor_mse | 242.00 | 730.00 | 0.00e+00 | 0.00e+00 |  |
| mps | 6c66262 | spectral_cosine | 332.00 | 1267.00 | 0.00e+00 | 0.00e+00 |  |
| mps | 6c66262 | orthogonality | 248.00 | 856.00 | 0.00e+00 | 0.00e+00 |  |
| mps | 6c66262 | airm | - | - | - | - | skipped: NotImplementedError("The operator 'aten::_linalg_eigh.eigenv |
| mps | 6c66262 | airm_iterative | - | - | - | - | skipped: TypeError("AIRMLoss.__init__() got an unexpected keyword arg |
| mps | 6c66262 | autoencoder_estimator | 234.00 | 676.50 | 0.00e+00 | 0.00e+00 |  |
| mps | 6c66262 | distiller_estimator | - | - | - | - | skipped: RuntimeError('Expected all tensors to be on the same device, |
| mps | v0.14.0 | classifier_ce | 218.00 | 707.00 | 0.00e+00 | 0.00e+00 |  |
| mps | v0.14.0 | regressor_mse | 218.00 | 688.00 | 0.00e+00 | 0.00e+00 |  |
| mps | v0.14.0 | spectral_cosine | 338.00 | 1367.00 | 0.00e+00 | 0.00e+00 |  |
| mps | v0.14.0 | orthogonality | 218.00 | 742.00 | 0.00e+00 | 0.00e+00 |  |
| mps | v0.14.0 | airm | 258.00 | 950.00 | 0.00e+00 | 0.00e+00 |  |
| mps | v0.14.0 | airm_iterative | 124122.00 | 2414.00 | 0.00e+00 | 0.00e+00 |  |
| mps | v0.14.0 | autoencoder_estimator | 214.00 | 622.00 | 0.00e+00 | 0.00e+00 |  |
| mps | v0.14.0 | distiller_estimator | 216.50 | 644.00 | 0.00e+00 | 0.00e+00 |  |

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

The per-batch timer wraps each region of the training step (outer forward, inner
encode, inner Green-kernel design matrix, readout solve, loss + backward + optimizer step)
with device synchronisation around the measurement, medians of repeats. Problem sizes:
small 1024 rows, 8 features, 32 anchors, 3 scales, latent 4; medium 4096 × 16, 64 anchors,
4 scales, latent 8, inner subset 4096; large 16384 × 64, 128 anchors, 4 scales, latent 16.

**CUDA, H100** (GPU idle; the pod's 64-core host reported a load average of 20–27
from other tenants during the run, which bounds the CPU-side launch overhead from above):

| device | commit | problem | outer_fwd | inner_encode | inner_kernel | solve | loss_bwd_step | total |
|---|---|---|---|---|---|---|---|---|
| cuda | 6c66262 | small | 0.54 | 0.21 | 0.29 | 1.51 | 1.21 | 3.76 |
| cuda | 6c66262 | medium | 0.55 | 0.23 | 0.33 | 2.69 | 1.19 | 4.98 |
| cuda | 6c66262 | large | 0.56 | 0.24 | 0.81 | 6.00 | 1.32 | 8.93 |
| cuda | 34d11bd | small | 0.50 | 0.20 | 0.27 | 0.75 | 1.09 | 2.80 |
| cuda | 34d11bd | medium | 0.51 | 0.21 | 0.31 | 0.76 | 1.08 | 2.88 |
| cuda | 34d11bd | large | 0.51 | 0.21 | 0.80 | 0.76 | 1.09 | 3.37 |
| cuda | 620cf27 | small | 0.50 | 0.21 | 0.25 | 0.77 | 1.18 | 2.93 |
| cuda | 620cf27 | medium | 0.50 | 0.22 | 0.26 | 0.78 | 1.17 | 2.92 |
| cuda | 620cf27 | large | 0.50 | 0.22 | 0.25 | 0.79 | 1.17 | 2.92 |
| cuda | 68fa992 | small | 0.51 | 0.21 | 0.26 | 0.77 | 1.43 | 3.18 |
| cuda | 68fa992 | medium | 0.51 | 0.22 | 0.26 | 0.78 | 1.35 | 3.12 |
| cuda | 68fa992 | large | 0.51 | 0.22 | 0.25 | 0.80 | 1.35 | 3.12 |
| cuda | 4823d23 | small | 0.49 | 0.21 | 0.25 | 0.76 | 1.16 | 2.88 |
| cuda | 4823d23 | medium | 0.49 | 0.22 | 0.25 | 0.78 | 1.15 | 2.89 |
| cuda | 4823d23 | large | 0.49 | 0.22 | 0.25 | 0.82 | 1.16 | 2.94 |
| cuda | b03bb29 | small | 0.49 | 0.21 | 0.25 | 0.77 | 1.30 | 3.02 |
| cuda | b03bb29 | medium | 0.50 | 0.22 | 0.26 | 0.78 | 1.22 | 2.97 |
| cuda | b03bb29 | large | 0.48 | 0.21 | 0.25 | 0.82 | 1.29 | 3.05 |
| cuda | 2259436 | small | 0.50 | 0.21 | 0.26 | 0.77 | 1.17 | 2.90 |
| cuda | 2259436 | medium | 0.50 | 0.22 | 0.26 | 0.79 | 1.16 | 2.92 |
| cuda | 2259436 | large | 0.50 | 0.22 | 0.26 | 0.90 | 1.18 | 3.06 |

Reading the CUDA columns: the baseline batch is dominated by the readout solve (1.5, 2.7,
6.0 ms of 3.8, 5.0, 8.9 ms), which was a float64 SVD on the device followed by a host
round trip. Item 1 (`34d11bd`) makes the solve 0.76 ms at every size; item 2 (`620cf27`)
takes the large problem's kernel from 0.80 to 0.25 ms. After those two the batch costs
2.9–3.1 ms at every problem size, so the step is launch-bound: items 3–6 remove
synchronisations without changing this timer (differences of ±0.2 ms in `loss_bwd_step`
between the later columns are within the run-to-run noise on the shared host). End to end
the medium problem's batch is 1.7× faster and the large one's 2.9×.

**MPS, M4 Max** (load 2.5–4.4 during every run, the benchmark's own threads included;
`v0.14.0` is main, the library of `4866d9e` plus a type-alias change):

| device | commit | problem | outer_fwd | inner_encode | inner_kernel | solve | loss_bwd_step | total |
|---|---|---|---|---|---|---|---|---|
| mps | 6c66262 | small | 2.41 | 2.67 | 2.20 | 3.12 | 6.40 | 16.79 |
| mps | 6c66262 | medium | 1.49 | 1.60 | 5.03 | 5.06 | 4.48 | 17.65 |
| mps | 6c66262 | large | 4.53 | 3.62 | 21.15 | 11.26 | 12.56 | 53.12 |
| mps | 34d11bd | small | 0.56 | 0.54 | 0.42 | 1.30 | 1.82 | 4.64 |
| mps | 34d11bd | medium | 0.59 | 0.67 | 2.07 | 3.41 | 1.83 | 8.57 |
| mps | 34d11bd | large | 1.33 | 0.76 | 6.90 | 6.63 | 2.43 | 18.04 |
| mps | 620cf27 | small | 0.47 | 0.54 | 0.37 | 1.30 | 1.87 | 4.55 |
| mps | 620cf27 | medium | 0.61 | 0.81 | 1.29 | 3.55 | 2.03 | 8.30 |
| mps | 620cf27 | large | 0.77 | 0.81 | 2.38 | 6.58 | 2.11 | 12.66 |
| mps | 68fa992 | small | 0.75 | 1.22 | 0.71 | 1.53 | 2.56 | 6.78 |
| mps | 68fa992 | medium | 0.54 | 0.71 | 1.13 | 3.51 | 1.86 | 7.75 |
| mps | 68fa992 | large | 0.69 | 0.70 | 2.03 | 6.43 | 1.93 | 11.79 |
| mps | 4823d23 | small | 0.44 | 0.53 | 0.36 | 1.28 | 1.76 | 4.37 |
| mps | 4823d23 | medium | 0.51 | 0.66 | 1.05 | 3.47 | 1.86 | 7.55 |
| mps | 4823d23 | large | 0.69 | 0.73 | 2.05 | 6.42 | 1.97 | 11.86 |
| mps | b03bb29 | small | 0.48 | 0.57 | 0.38 | 1.29 | 2.09 | 4.81 |
| mps | b03bb29 | medium | 0.54 | 0.78 | 1.25 | 3.51 | 1.94 | 8.03 |
| mps | b03bb29 | large | 0.69 | 0.73 | 2.06 | 6.41 | 1.91 | 11.80 |
| mps | 08d5efc | small | 0.42 | 0.53 | 0.35 | 1.24 | 1.80 | 4.34 |
| mps | 08d5efc | medium | 0.52 | 0.67 | 1.06 | 3.47 | 1.90 | 7.62 |
| mps | 08d5efc | large | 0.71 | 0.74 | 2.06 | 6.52 | 1.97 | 12.00 |
| mps | v0.14.0 | small | 0.44 | 0.55 | 0.36 | 1.28 | 1.78 | 4.40 |
| mps | v0.14.0 | medium | 0.51 | 0.66 | 1.06 | 3.55 | 1.93 | 7.72 |
| mps | v0.14.0 | large | 0.68 | 0.72 | 2.05 | 6.61 | 1.99 | 12.05 |

Apple silicon tells the same story with different weights: the readout solve (a CPU
`lstsq` round trip at v0.13.0) and the kernel are the batch. Item 1 takes the large batch
from 53.1 to 18.0 ms and item 2 to 12.7 ms; the later items are flat, as on CUDA. The
device solve stays the largest region on MPS (6.6 ms of 12 on the large problem: the
Cholesky and triangular solves run through Metal Performance Shaders at fp32) and is the
next thing to look at on this device. Per batch, main is 3.8×, 2.3× and 4.4× faster than
v0.13.0 on the three sizes. The two runs of main agree within 2% (a first run at a
15-minute load average of 6.7 came out uniformly 2× slower and was discarded).

**CPU, M4 Max** (the bit-identical reference branch; load 2.8–5.9):

| device | commit | problem | outer_fwd | inner_encode | inner_kernel | solve | loss_bwd_step | total |
|---|---|---|---|---|---|---|---|---|
| cpu | 6c66262 | small | 0.59 | 0.64 | 1.15 | 0.41 | 1.49 | 4.28 |
| cpu | 6c66262 | medium | 1.52 | 1.93 | 5.39 | 2.95 | 3.38 | 15.17 |
| cpu | 6c66262 | large | 3.84 | 2.13 | 16.78 | 5.97 | 7.34 | 36.06 |
| cpu | 34d11bd | small | 0.60 | 0.65 | 1.16 | 0.40 | 1.51 | 4.32 |
| cpu | 34d11bd | medium | 1.53 | 1.89 | 5.35 | 2.59 | 3.52 | 14.88 |
| cpu | 34d11bd | large | 3.84 | 2.10 | 16.85 | 6.06 | 7.55 | 36.40 |
| cpu | 620cf27 | small | 0.54 | 0.65 | 0.86 | 0.40 | 1.49 | 3.94 |
| cpu | 620cf27 | medium | 1.17 | 1.88 | 3.67 | 2.57 | 3.14 | 12.43 |
| cpu | 620cf27 | large | 2.73 | 2.08 | 11.64 | 5.95 | 7.26 | 29.65 |
| cpu | 68fa992 | small | 0.52 | 0.63 | 0.84 | 0.38 | 1.42 | 3.80 |
| cpu | 68fa992 | medium | 1.24 | 1.90 | 3.85 | 2.61 | 3.39 | 12.99 |
| cpu | 68fa992 | large | 2.71 | 2.07 | 11.42 | 5.99 | 7.25 | 29.44 |
| cpu | 4823d23 | small | 0.54 | 0.65 | 0.85 | 0.39 | 1.48 | 3.92 |
| cpu | 4823d23 | medium | 1.19 | 1.85 | 3.76 | 2.57 | 3.22 | 12.60 |
| cpu | 4823d23 | large | 2.77 | 2.05 | 11.42 | 5.96 | 7.14 | 29.34 |
| cpu | b03bb29 | small | 0.53 | 0.64 | 0.86 | 0.39 | 1.48 | 3.90 |
| cpu | b03bb29 | medium | 1.26 | 1.89 | 3.81 | 2.59 | 3.31 | 12.86 |
| cpu | b03bb29 | large | 2.67 | 2.10 | 11.49 | 5.97 | 7.29 | 29.52 |
| cpu | 08d5efc | small | 0.53 | 0.64 | 0.87 | 0.39 | 1.48 | 3.93 |
| cpu | 08d5efc | medium | 1.24 | 1.97 | 3.92 | 2.59 | 3.38 | 13.11 |
| cpu | 08d5efc | large | 2.73 | 2.04 | 11.67 | 5.82 | 7.15 | 29.41 |
| cpu | v0.14.0 | small | 0.51 | 0.62 | 0.82 | 0.40 | 1.38 | 3.73 |
| cpu | v0.14.0 | medium | 1.21 | 2.00 | 3.73 | 2.53 | 3.25 | 12.73 |
| cpu | v0.14.0 | large | 2.75 | 2.14 | 11.62 | 6.02 | 7.38 | 29.91 |

On the CPU only item 2's exact sign skipping is allowed to change anything, and that is
what the columns show: the kernel region drops from 5.4 to 3.7 ms (medium) and 16.8 to
11.6 ms (large), every other region is unchanged within noise, and every later item is
flat. Main is 1.15–1.2× faster per batch than v0.13.0 on the CPU.

## End-to-end examples (wall seconds and headline outputs)

Each bundled example runs as a subprocess with `IGL_EXAMPLE_DEVICE` set; wall time,
peak RSS and the example's printed headline numbers are recorded.

**CUDA, H100** (the last column is the final library, graph replay on by default; the
MPS and CPU tables follow the CUDA paragraph):

| device | example | 6c66262 | 34d11bd | 620cf27 | 68fa992 | 4823d23 | b03bb29 | 08d5efc | 4866d9e | headlines (first) |
|---|---|---|---|---|---|---|---|---|---|---|
| cuda | moons_xor | 27.9 | - | - | - | - | - | 17.8 | 7.3 | {'acc': ['1.000'], 'd_eff': ['1', '3', '3', '1', '3', '3'], 'r2': ['1.000'], 'hierarchy': ['True']} |
| cuda | poisson_1d | 56.9 | - | - | - | - | - | 37.4 | 11.3 | {'mse': ['0.00000', '0.00000', '0.00000', '0.00000', '0.00000', '0.00000']} |
| cuda | save_load | 4.0 | - | - | - | - | - | 11.1 | 9.3 | {} |
| cuda | swiss_roll_recon | 24.8 | - | - | - | - | - | 15.6 | 7.2 | {'r2': ['0.999']} |
| cuda | torus_classification | 55.0 | - | - | - | - | - | 32.8 | 10.0 | {'acc': ['0.9960'], 'd_eff': ['1', '3', '3']} |
| cuda | whitened_regression | 20.6 | - | - | - | - | - | 38.4 | 19.1 | {'kl': ['2.3803', '1.7501', '0.5782', '0.1738']} |

Two examples could not run on CUDA (or MPS) at v0.13.0 at all: `whitened_regression` and
`save_load` both use `IGLDistiller`, whose target whitener kept its constants on the CPU;
their baseline wall times are the time to the crash. `save_load` additionally needed the
checkpoint writer to stop requiring an installed distribution (`igl.__version__` is
recorded instead). With the eager work alone (`08d5efc`) the other four ran 1.4–1.8×
faster; with the graph replay (`4866d9e`) they run 3.4–5.5× faster than v0.13.0 end to end
(moons 27.9 → 7.3 s, swiss roll 24.8 → 7.2 s, torus 55.0 → 10.0 s, Poisson 56.9 → 11.3 s)
with the same headline outputs (accuracy, `d_eff`, R², MSE, round-trip error). The
examples use the cosine warm-restart scheduler, which is why the learning rate lives in a
device tensor the recorded step reads: an earlier version re-recorded the graph on every
rate change and made these examples 2–3× slower instead.

**MPS, M4 Max** (no graph replay on MPS; the wins are the device solve and the kernel path):

| device | example | 6c66262 | 34d11bd | 620cf27 | 68fa992 | 4823d23 | b03bb29 | 08d5efc | v0.14.0 | headlines (first) |
|---|---|---|---|---|---|---|---|---|---|---|
| mps | moons_xor | 53.7 | - | - | - | - | - | - | 21.7 | {'acc': ['1.000'], 'd_eff': ['1', '3', '3', '1', '3', '3'], 'r2': ['1.000'], 'hierarchy': ['True']} |
| mps | poisson_1d | 208.6 | - | - | - | - | - | - | 54.4 | {'mse': ['0.00000', '0.00000', '0.00000', '0.00000', '0.00000', '0.00000']} |
| mps | save_load | 7.7 | - | - | - | - | - | - | 11.3 | {} |
| mps | swiss_roll_recon | 35.0 | - | - | - | - | - | - | 19.7 | {'r2': ['0.997']} |
| mps | torus_classification | 128.7 | - | - | - | - | - | - | 45.0 | {'acc': ['0.9960'], 'd_eff': ['1', '3', '3']} |
| mps | whitened_regression | 61.3 | - | - | - | - | - | - | 46.5 | {'kl': ['2.5124', '2.0802', '0.5746', '0.1727']} |

Moons 53.7 → 21.7 s, swiss roll 35.0 → 19.7 s, torus 128.7 → 45.0 s, Poisson 208.6 → 54.4 s
(2.5–3.8×), and the two distiller examples run instead of crashing. On MPS the training
trajectory is not the CPU's (fp32 device solve, fast kernel path, float32 accumulation),
and two headline readings move with it: the torus classifier reads 0.9920 and `d_eff`
1, 4, 4 (v0.13.0 on MPS: 0.9960 and 1, 3, 3), and the moons example's nested-budget check
reads `False` with `d_eff` 1, 4, 3 (v0.13.0: `True` with 1, 3, 3) while its accuracy and
R² stay 1.000. Those are different local optima of the same problem, not accuracy losses;
the CPU and CUDA runs of the same examples keep their readings.

**CPU, M4 Max:**

| device | example | 6c66262 | 34d11bd | 620cf27 | 68fa992 | 4823d23 | b03bb29 | 08d5efc | v0.14.0 | headlines (first) |
|---|---|---|---|---|---|---|---|---|---|---|
| cpu | moons_xor | 24.3 | - | - | - | - | - | - | 20.6 | {'acc': ['1.000'], 'd_eff': ['1', '4', '4', '1', '4', '4'], 'r2': ['1.000'], 'hierarchy': ['True']} |
| cpu | poisson_1d | 25.4 | - | - | - | - | - | - | 25.0 | {'mse': ['0.00000', '0.00000', '0.00000', '0.00000', '0.00000', '0.00000']} |
| cpu | save_load | 12.7 | - | - | - | - | - | - | 11.6 | {'roundtrip': ['0.00e+00']} |
| cpu | swiss_roll_recon | 15.9 | - | - | - | - | - | - | 13.8 | {'r2': ['0.998']} |
| cpu | torus_classification | 83.8 | - | - | - | - | - | - | 67.2 | {'acc': ['0.9980'], 'd_eff': ['11', '3', '3']} |
| cpu | whitened_regression | 99.7 | - | - | - | - | - | - | 81.7 | {'kl': ['1.7005', '1.3271', '0.5272', '0.1737', '1.0501', '0.8264', '0.2217', '0.0429', '1.1264']} |

Every headline output is identical to v0.13.0 on the CPU (checked field by field across
the six examples: accuracy, `d_eff`, R², MSE, KL, round-trip error), which is the
end-to-end form of the bit-identity claim; the examples run 1.0–1.25× faster (torus
83.8 → 67.2 s, whitened regression 99.7 → 81.7 s), the kernel's share of their time.

## Op-level profile

`torch.profiler` over a five-epoch fit of the medium problem with CPU and CUDA
activities, figures per epoch (Chrome traces are written next to the JSONs). The
five-epoch protocol amortises the one graph capture the way real training does; the
baseline and `08d5efc` were re-profiled under the same protocol.

| device | commit | wall ms | device busy ms | busy fraction | kernel launches | peak device MB |
|---|---|---|---|---|---|---|
| cuda | 6c66262 | 261.08 | 105.33 | 0.40 | 7508.00 | 150.65 |
| cuda | 08d5efc | 177.67 | 21.72 | 0.12 | 3663.00 | 94.32 |
| cuda | 4866d9e | 68.46 | 11.04 | 0.16 | 3681.00 | 157.87 |

### top ops by self device time, cuda @ 6c66262

| op | calls | self cpu ms | self device ms |
|---|---|---|---|
| aten::_linalg_svd | 85 | 13.02 | 182.12 |
| void gesvdbj_batch_32x16<double, double>(long, i | 2451 | 0.00e+00 | 92.03 |
| void geqr2_gmem_domino<double, double, 9>(int, i | 85 | 0.00e+00 | 46.17 |
| Optimizer.step#AdamW.step | 80 | 0.00e+00 | 18.10 |
| void svd_column_rotate_batch<double, 5, 3>(long, | 4902 | 0.00e+00 | 16.89 |
| aten::sum | 1310 | 8.31 | 12.56 |
| void svd_row_rotate_batch_32x16<double>(long, in | 2451 | 0.00e+00 | 11.13 |
| aten::mul | 2970 | 14.60 | 7.52 |

### top ops by self device time, cuda @ 08d5efc

| op | calls | self cpu ms | self device ms |
|---|---|---|---|
| Optimizer.step#AdamW.step | 80 | 0.00e+00 | 5.78 |
| aten::mm | 905 | 12.49 | 5.68 |
| aten::mul | 3285 | 15.77 | 5.14 |
| aten::sum | 810 | 5.11 | 4.55 |
| aten::_fused_adamw_ | 80 | 0.64 | 3.76 |
| void at::native::(anonymous namespace)::multi_te | 80 | 0.00e+00 | 3.76 |
| void at::native::reduce_kernel<128, 4, at::nativ | 560 | 0.00e+00 | 3.30 |
| aten::_cholesky_solve_helper | 170 | 2.36 | 3.29 |

### top ops by self device time, cuda @ 4866d9e

| op | calls | self cpu ms | self device ms |
|---|---|---|---|
| void at::native::(anonymous namespace)::multi_te | 80 | 0.00e+00 | 3.81 |
| void at::native::reduce_kernel<128, 4, at::nativ | 560 | 0.00e+00 | 3.19 |
| void kernel<getrf_wo_pivot_params_<float, 0, 256 | 85 | 0.00e+00 | 2.50 |
| void cublasLt::splitKreduce_kernel<32, 16, int,  | 655 | 0.00e+00 | 2.19 |
| void at::native::elementwise_kernel<128, 2, at:: | 1230 | 0.00e+00 | 2.18 |
| void at::native::(anonymous namespace)::vectoriz | 340 | 0.00e+00 | 2.01 |
| void kernel_trsm_l_mul32<float, 8, false, true,  | 170 | 0.00e+00 | 1.51 |
| void at::native::vectorized_elementwise_kernel<4 | 975 | 0.00e+00 | 1.50 |

The baseline epoch spent 72% of its GPU time in a float64 batched SVD (`aten::_linalg_svd`:
the readout `lstsq` solved in double on the device) and launched about 7500 kernels from
the host. The eager work (`08d5efc`) cuts the GPU time per epoch from 105 to 22 ms and the
launches to 3660; the largest remaining items are the fused AdamW step, the matmuls of the
encoder and kernel, and the Cholesky solve. The device is then busy 12% of the epoch
because the wall time (261 → 178 ms) is set by Python and launch overhead: the H100 idles
between kernels. The graph replay (`4866d9e`) removes that overhead: 68 ms per epoch, the
same kernels executed (the profiler still counts them, 3681 per epoch) but launched as one
graph per batch, GPU time 11 ms. What remains is the GPU's own time for many tiny
kernels, which fusion reduces further (next section).

On the Mac the same five-epoch profile gives, per epoch of the medium problem, 157 → 102 ms
on MPS and 276 → 249 ms on the CPU (the CPU's change being the kernel path only).

## Component benchmarks

Formulation-level comparisons measured in isolation (`components.py`).

**CUDA, H100:**

### cuda @ 6c66262

| d | fwd current | fwd einsum | fwd+bwd current | fwd+bwd einsum | rel err |
|---|---|---|---|---|---|
| 8 | 0.24 | 0.14 | 0.83 | 0.68 | 1.12e-07 |
| 16 | 0.35 | 0.13 | 1.26 | 0.72 | 1.26e-07 |

| n | r | c | method | ms | rel pred err |
|---|---|---|---|---|---|
| 4096 | 64 | 2 | package_lstsq | 2.62 | 2.35e-07 |
| 4096 | 64 | 2 | cholesky_refined | 0.19 | 8.49e-06 |
| 4096 | 64 | 2 | lstsq_gels | 0.69 | 1.54e-06 |
| 4096 | 64 | 512 | package_lstsq | 2.60 | 1.97e-07 |
| 4096 | 64 | 512 | cholesky_refined | 0.17 | 1.30e-05 |
| 4096 | 64 | 512 | lstsq_gels | 0.73 | 2.52e-06 |
| 4096 | 256 | 2000 | package_lstsq | 15.06 | 3.66e-07 |
| 4096 | 256 | 2000 | cholesky_refined | 0.71 | 1.72e-05 |
| 4096 | 256 | 2000 | lstsq_gels | 2.91 | 8.68e-06 |

| batch | input_dim | latent_dim | loop ms | vmap ms | |Δloss| |
|---|---|---|---|---|---|
| 256 | 10 | 4 | 4.80 | 2.64 | 2.98e-08 |
| 256 | 2080 | 16 | 16.27 | 2.66 | 2.98e-08 |

### cuda @ 2259436

| d | fwd current | fwd einsum | fwd+bwd current | fwd+bwd einsum | rel err |
|---|---|---|---|---|---|
| 8 | 0.17 | 0.12 | 0.72 | 0.57 | 0.00e+00 |
| 16 | 0.17 | 0.12 | 0.67 | 0.56 | 0.00e+00 |

| n | r | c | method | ms | rel pred err |
|---|---|---|---|---|---|
| 4096 | 64 | 2 | package_lstsq | 0.68 | 8.49e-06 |
| 4096 | 64 | 2 | cholesky_refined | 0.19 | 8.49e-06 |
| 4096 | 64 | 2 | lstsq_gels | 0.70 | 1.54e-06 |
| 4096 | 64 | 512 | package_lstsq | 0.62 | 1.30e-05 |
| 4096 | 64 | 512 | cholesky_refined | 0.19 | 1.30e-05 |
| 4096 | 64 | 512 | lstsq_gels | 0.73 | 2.52e-06 |
| 4096 | 256 | 2000 | package_lstsq | 1.20 | 1.73e-05 |
| 4096 | 256 | 2000 | cholesky_refined | 0.71 | 1.72e-05 |
| 4096 | 256 | 2000 | lstsq_gels | 2.90 | 8.68e-06 |

| batch | input_dim | latent_dim | loop ms | vmap ms | |Δloss| |
|---|---|---|---|---|---|
| 256 | 10 | 4 | 2.83 | 2.67 | 0.00e+00 |
| 256 | 2080 | 16 | 2.69 | 2.66 | 0.00e+00 |

The kernel table compares the module's current formulation with the raw contracted
`einsum` (at HEAD the "current" path already is the contraction, so the two columns
nearly coincide); relative error is against the reference formulation. The solver table
times the package's `direct_solve_weights` (`package_lstsq`) against the Cholesky-plus-
refinement device solve and the raw `gels` least squares, with the prediction error
against a float64 reference: at HEAD the package solve is the device Cholesky, 12.5× faster
than the baseline's on the 4096 × 256 × 2000 system (15.06 → 1.20 ms) at 1.7e-5 relative
prediction error versus 3.7e-7 for the float64 SVD, well below the training loss's own
noise floor. The Jacobian table is the orthogonality penalty's loop versus
`vmap(jacrev)`: 6.1× faster at input dimension 2080 with an identical loss.

**MPS, M4 Max** (the component benchmark's error metric is computed on the CPU; MPS has no
float64, which is why the baseline's first attempt at this table failed):

### mps @ 6c66262

| d | fwd current | fwd einsum | fwd+bwd current | fwd+bwd einsum | rel err |
|---|---|---|---|---|---|
| 8 | 1.29 | 1.04 | 4.26 | 2.39 | 1.12e-07 |
| 16 | 1.98 | 1.13 | 8.06 | 2.58 | 1.29e-07 |

| n | r | c | method | ms | rel pred err |
|---|---|---|---|---|---|
| 4096 | 64 | 2 | package_lstsq | 3.52 | 1.42e-06 |
| 4096 | 64 | 2 | cholesky_refined | 0.91 | 3.35e-06 |
| 4096 | 64 | 512 | package_lstsq | 8.64 | 3.86e-06 |
| 4096 | 64 | 512 | cholesky_refined | 0.88 | 4.49e-05 |
| 4096 | 256 | 2000 | package_lstsq | 42.00 | 5.82e-05 |
| 4096 | 256 | 2000 | cholesky_refined | 3.04 | 1.48e-04 |

| batch | input_dim | latent_dim | loop ms | vmap ms | |Δloss| |
|---|---|---|---|---|---|
| 256 | 10 | 4 | 4.60 | 2.37 | 0.00e+00 |
| 256 | 2080 | 16 | 14.30 | 6.06 | 0.00e+00 |

### mps @ v0.14.0

| d | fwd current | fwd einsum | fwd+bwd current | fwd+bwd einsum | rel err |
|---|---|---|---|---|---|
| 8 | 1.06 | 0.98 | 2.54 | 2.37 | 0.00e+00 |
| 16 | 1.12 | 1.10 | 2.74 | 2.58 | 0.00e+00 |

| n | r | c | method | ms | rel pred err |
|---|---|---|---|---|---|
| 4096 | 64 | 2 | package_lstsq | 3.18 | 1.42e-06 |
| 4096 | 64 | 2 | cholesky_refined | 0.77 | 3.35e-06 |
| 4096 | 64 | 512 | package_lstsq | 8.70 | 3.86e-06 |
| 4096 | 64 | 512 | cholesky_refined | 0.82 | 4.49e-05 |
| 4096 | 256 | 2000 | package_lstsq | 40.69 | 4.92e-06 |
| 4096 | 256 | 2000 | cholesky_refined | 3.02 | 1.48e-04 |

| batch | input_dim | latent_dim | loop ms | vmap ms | |Δloss| |
|---|---|---|---|---|---|
| 256 | 10 | 4 | 2.25 | 1.91 | 0.00e+00 |
| 256 | 2080 | 16 | 5.96 | 5.89 | 0.00e+00 |

**CPU, M4 Max:**

### cpu @ 6c66262

| d | fwd current | fwd einsum | fwd+bwd current | fwd+bwd einsum | rel err |
|---|---|---|---|---|---|
| 8 | 4.20 | 2.58 | 13.62 | 7.12 | 1.21e-07 |
| 16 | 6.33 | 3.31 | 22.62 | 9.06 | 1.39e-07 |

| n | r | c | method | ms | rel pred err |
|---|---|---|---|---|---|
| 4096 | 64 | 2 | package_lstsq | 2.40 | 1.42e-06 |
| 4096 | 64 | 2 | cholesky_refined | 0.17 | 6.59e-05 |
| 4096 | 64 | 512 | package_lstsq | 7.97 | 3.86e-06 |
| 4096 | 64 | 512 | cholesky_refined | 1.43 | 1.24e-04 |
| 4096 | 256 | 2000 | package_lstsq | 38.47 | 4.94e-05 |
| 4096 | 256 | 2000 | cholesky_refined | 8.59 | 1.29e-04 |

| batch | input_dim | latent_dim | loop ms | vmap ms | |Δloss| |
|---|---|---|---|---|---|
| 256 | 10 | 4 | 19.24 | 6.93 | 0.00e+00 |
| 256 | 2080 | 16 | 109.03 | 37.06 | 0.00e+00 |

### cpu @ v0.14.0

| d | fwd current | fwd einsum | fwd+bwd current | fwd+bwd einsum | rel err |
|---|---|---|---|---|---|
| 8 | 2.93 | 2.55 | 12.36 | 7.06 | 1.21e-07 |
| 16 | 4.98 | 3.24 | 20.52 | 9.14 | 1.39e-07 |

| n | r | c | method | ms | rel pred err |
|---|---|---|---|---|---|
| 4096 | 64 | 2 | package_lstsq | 2.27 | 1.42e-06 |
| 4096 | 64 | 2 | cholesky_refined | 0.17 | 6.59e-05 |
| 4096 | 64 | 512 | package_lstsq | 8.16 | 3.86e-06 |
| 4096 | 64 | 512 | cholesky_refined | 1.38 | 1.24e-04 |
| 4096 | 256 | 2000 | package_lstsq | 38.32 | 4.94e-05 |
| 4096 | 256 | 2000 | cholesky_refined | 8.98 | 1.29e-04 |

| batch | input_dim | latent_dim | loop ms | vmap ms | |Δloss| |
|---|---|---|---|---|---|
| 256 | 10 | 4 | 18.66 | 7.17 | 0.00e+00 |
| 256 | 2080 | 16 | 112.33 | 35.36 | 0.00e+00 |

On MPS the device solve is 3–8× faster than the CPU `lstsq` round trip at the same
prediction error scale, and the Jacobian's `vmap` form is used off-CPU only; on the CPU the
loop and the reference kernel formulation stay (the fast path is a device-branch choice).


## CUDA graph replay of the batch step (implemented) and `torch.compile` fusion (opt-in)

After the synchronisation work the batch step was launch-bound: about 250 kernels per
batch and 2.5–2.9 ms at every problem size, the GPU busy 12% of the epoch.
`benchmarks/device/launch_bound.py` first rebuilt the step over static buffers to measure
what removes the launch overhead (medium: eager 2.54 ms, whole-step CUDA graph replay
0.74 ms, `torch.compile` fusion alone 1.72 ms; large: 2.55 / 0.92 / 1.81 ms). The library
now does the same inside the CUDA branch (`igl.core._graph.GraphRunner`): static index,
gate-mask and learning-rate buffers, two eager warm-up batches, one capture per fit,
replay for every full batch, the last partial batch eager; the fused AdamW is built
`capturable` and reads the learning rate from a device tensor, so a scheduler step copies a
value instead of forcing a re-capture. The runner steps aside for `PrefixForward` modules,
extra losses, a loss that synchronises (`AIRMLoss` with `eigh`) and data-driven spectral
bases, and falls back to eager training with a warning if a capture is refused.

Epoch wall of `MatryoshkaTrainer.fit` per mode (`epoch_modes.py`; H100, medians of five
epochs after the first, device-synced; the first-epoch column carries the warm-up and
capture, or the compilation):

| problem | eager | graph (default) | compile only | graph + compile | first epoch: graph / compile / both |
|---|---|---|---|---|---|
| small (1024 × 8) | 21.9 ms | **4.37 ms** (5.0×) | 16.0 ms | **3.18 ms** (6.9×) | 0.11 s / 3.5 s / 0.39 s |
| medium (4096 × 16) | 42.8 ms | **10.3 ms** (4.2×) | 31.4 ms | **8.19 ms** (5.2×) | 0.16 s / 1.5 s / 0.31 s |
| large (16384 × 64) | 86.5 ms | **31.1 ms** (2.8×) | 65.1 ms | **21.9 ms** (4.0×) | 0.21 s / 1.5 s / 0.30 s |

The final training loss agrees across modes to 1e-3 (the parity tests in
`tests/test_cuda_graphs.py` assert this on the H100). Fusion on its own is worth 1.3–1.4×
and costs seconds of compilation per fit; on top of the graph it takes another 20–30% off
at 0.3–0.4 s per fit, so it stays opt-in (`MatryoshkaConfig.torch_compile`) for long fits.
The synchronisation census reads exactly one host transfer per epoch in graph mode (the
capture's own host copies were removed: `torch.full` and `fill_` instead of
`torch.tensor` and `as_tensor` for the learning rate).

## Follow-up round: MPS readout solve, CPU thread cap, CUDA default batch size

Three candidates from the per-device assessment after v0.14.0, each measured at the fit
level (`epoch_modes.py`, median epoch after the first, device-synced) and with the region
timer and the examples.

**MPS hybrid readout solve: a negative result, not shipped in the trainer.** In isolation,
forming the Gram on the device and factoring the `R × R` system in float64 on the CPU is
3× faster than Metal's Cholesky at `R = 256` (0.9 vs 2.8 ms, 1e-6 prediction error). In
the training loop it is slower: the device-to-host copy drains the asynchronous Metal
queue every batch, and the pipelining lost costs more than the solve saved.

| MPS, fit epoch wall | on-device solve (v0.14.0) | hybrid solve |
|---|---|---|
| small | 35.2 ms | 42.1 ms |
| medium | 80.5 ms | 88.7 ms |
| large | 232.7 ms | 226.1 ms |
| examples (moons / swiss / torus / whitened / Poisson) | 21.7 / 19.7 / 45.0 / 46.5 / 54.4 s | 24.5 / 22.3 / 54.0 / 55.4 / 63.6 s |

`MpsBackend` keeps the on-device solve; the hybrid (`ridge_solve_hybrid`) only serves the
one-off public `direct_solve_weights` on MPS, where the previous path was a CPU `lstsq`
fallback. After the revert the fit-level timer reads 35.4 / 80.9 / 234.0 ms and the
examples 22.2 / 19.9 / 45.9 / 47.1 / 52.8 s, back on v0.14.0's numbers within noise. The lesson generalises: a per-call micro-benchmark that synchronises after every
call cannot see the cost of breaking an asynchronous queue.

**CPU thread cap (`MatryoshkaConfig.cpu_threads`, opt-in).** Torch's default on the M4 Max
is 12 intra-op threads; small tensors oversubscribe them.

| CPU, fit epoch wall | default (12) | 4 threads | 6 threads | 8 threads |
|---|---|---|---|---|
| small | 37.3 ms | 26.9 ms | 29.5 ms | 34.6 ms |
| medium | 220.8 ms | 211.6 ms | 197.7 ms | 209.7 ms |
| large | 956 ms | 1049 ms | 955 ms | 939 ms |

Six threads is the safe cap on this machine (21% and 10% faster on small and medium, no
loss on large; four threads costs 10% on large). It stays off by default because the
thread count changes the order of BLAS reductions: the reference trajectory differs at
the 5e-5 level in the weights, so results are no longer bit-identical to the default's.

**CUDA default batch size 1024** (`MatryoshkaConfig.batch_size` left at `None` resolves to
1024 on CUDA and 256 elsewhere; an explicit value is used as given). The per-batch cost on
the H100 is flat in the batch size (2.9 ms eager, 0.6 ms replayed, at every problem size),
so an epoch with a quarter of the steps costs about a quarter. The examples that do not
set a batch size take the new default on CUDA; their wall times and headline outputs
under it are recorded below once the GPU is free.

Headline outputs under the 1024 default, H100 (the GPU was shared with a training job at
82% utilisation, so these are the outputs only; the wall times of that run are not
reported): moons, swiss roll, torus and Poisson, the four examples that leave the batch
size unset, produce exactly the headlines they produced at 256 (accuracy 1.000 and
0.9960, `d_eff` 1, 3, 3, R² 0.999, MSE 0). Whitened regression and save/load set their
batch size explicitly and are unaffected; their last digits move at the 1e-2 (KL) and
1e-7 (round trip) level between two CUDA runs, the run-to-run spread of TF32 matmuls.
Timings under the default, H100 idle but its 64-core host at a load average of 31 from
other tenants (2026-09-09 19:26 UTC): the examples whose batch size is fixed slowed by
13–35% against the idle-host run, which dates every absolute wall time of that run, so
the batch effect was taken as an A/B in one process, alternating the two settings
(medium-size model, 8 epochs, median epoch after the third, device-synced):

| rows | batch 256 | batch 1024 (default) | steps per epoch, full + partial |
|---|---|---|---|
| 1000 | 8.45 ms | 6.58 ms | 3 + 1 → 0 + 1 |
| 2000 | 12.5 ms | 6.99 ms | 7 + 1 → 1 + 1 |
| 4096 | 22.4 ms | 7.99 ms | 16 → 4 |
| 16384 | 98.2 ms | 27.0 ms | 64 → 16 |

The epoch is 1.3× to 3.6× faster, the ratio growing with the dataset because the per-batch
cost is flat. The trade is the usual one: a quarter of the optimizer steps per epoch, so a
short fit converges less per epoch (after 8 epochs the training loss reads 0.56 against
0.45 at 1000 rows, 0.45 against 0.38 at 4096); the examples run hundreds of epochs and
land on the same headlines. Below 1024 rows the whole epoch is one partial batch, which the
graph replay does not cover (it captures full batches only); a second graph for the tail
shape would extend the replay to small datasets and is the next follow-up.

## Status

Complete. CUDA on the H100 (idle GPU, 2026-09-08): full suite for the baseline and HEAD,
region timer and census for each intermediate commit, profiler, epoch modes and the
examples for the graph-replay commit. MPS and CPU on the M4 Max (2026-09-08 09:22–09:52
UTC, load 2–6 with the benchmark's own threads, no other process above 120% CPU): baseline,
each intermediate commit and main (`v0.14.0`), with one contaminated MPS batch discarded
and re-measured.
