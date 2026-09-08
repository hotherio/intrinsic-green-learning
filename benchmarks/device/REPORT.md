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

**CPU and MPS (M4 Max):** the baseline MPS column was taken at load 3.3; the HEAD and
ablation MPS columns and the whole CPU matrix are pending a quiet machine (the first
attempt ran into the other session's jobs, load 12, and was discarded).

## End-to-end examples (wall seconds and headline outputs)

Each bundled example runs as a subprocess with `IGL_EXAMPLE_DEVICE` set; wall time,
peak RSS and the example's printed headline numbers are recorded.

**CUDA, H100** (the last column is the final library, graph replay on by default):

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

## Status

CUDA: complete (H100 idle, 2026-09-08: full suite for the baseline and HEAD, region timer
and census for each intermediate commit, profiler and epoch modes and examples for the
graph-replay commit). CPU and MPS: the baseline
MPS run is in; HEAD and the ablation on MPS and the whole CPU matrix are re-run
automatically when the Mac load is under 6 and will replace this paragraph.
