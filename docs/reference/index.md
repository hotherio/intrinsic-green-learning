# API reference

Every public symbol the library exposes. Generated from docstrings via
[mkdocstrings](https://mkdocstrings.github.io/).

## Top-level

The flat surface re-exported by `igl`:

- **Configuration** — [`EncoderConfig`][igl.config.EncoderConfig], [`KernelConfig`][igl.config.KernelConfig],
  [`MatryoshkaConfig`][igl.config.MatryoshkaConfig], [`IGLConfig`][igl.config.IGLConfig].
- **Building blocks** — [`MLPEncoder`][igl.core.encoder.MLPEncoder],
  [`LinearEncoder`][igl.core.encoder.LinearEncoder], [`GreenKernel`][igl.core.kernel.GreenKernel],
  [`IGLModule`][igl.nn.module.IGLModule].
- **Training** — [`MatryoshkaTrainer`][igl.core.trainer.MatryoshkaTrainer],
  [`TrainingHistory`][igl.core.trainer.TrainingHistory],
  [`CrossEntropyLoss`][igl.core.loss.CrossEntropyLoss], [`MSELoss`][igl.core.loss.MSELoss],
  [`direct_solve_weights`][igl.core.solver.direct_solve_weights],
  [`normalize_phi`][igl.core.normalization.normalize_phi].
- **Matryoshka & dimension discovery** — [`UniformSampler`][igl.matryoshka.sampler.UniformSampler],
  [`PowerLawSampler`][igl.matryoshka.sampler.PowerLawSampler],
  [`eval_dimension_curve`][igl.matryoshka.dimension_curve.eval_dimension_curve],
  [`detect_elbow`][igl.matryoshka.dimension_curve.detect_elbow],
  [`d_eff_from_curve`][igl.metrics.dimension.d_eff_from_curve],
  [`compare_d_eff`][igl.metrics.dimension.compare_d_eff],
  [`DimensionComparison`][igl.metrics.dimension.DimensionComparison].
- **Kernel registry** — [`Operator`][igl.kernels._registry.Operator],
  [`register_operator`][igl.kernels._registry.register_operator],
  [`get_operator`][igl.kernels._registry.get_operator],
  [`list_operators`][igl.kernels._registry.list_operators].
- **sklearn estimators** — [`IGLClassifier`][igl.models.classifier.IGLClassifier],
  [`IGLRegressor`][igl.models.regressor.IGLRegressor],
  [`IGLAutoencoder`][igl.models.autoencoder.IGLAutoencoder].
- **Types & enums** — [`OperatorName`][igl.types.OperatorName],
  [`SamplingMode`][igl.types.SamplingMode], [`NormalizeMode`][igl.types.NormalizeMode],
  [`NormType`][igl.types.NormType], [`ActivationType`][igl.types.ActivationType],
  [`EncoderKind`][igl.types.EncoderKind], [`SchedulerType`][igl.types.SchedulerType],
  [`LossStrategy`][igl.types.LossStrategy], [`MatryoshkaSampler`][igl.types.MatryoshkaSampler],
  [`OperatorFn`][igl.types.OperatorFn], [`EncoderProtocol`][igl.types.EncoderProtocol].
- **Exceptions** — [`IGLError`][igl.exceptions.IGLError], [`IGLConfigError`][igl.exceptions.IGLConfigError],
  [`IGLConvergenceError`][igl.exceptions.IGLConvergenceError],
  [`IGLDependencyError`][igl.exceptions.IGLDependencyError],
  [`IGLNotFittedError`][igl.exceptions.IGLNotFittedError].

## Subpackages

- [`igl.core`](core.md) — encoder, kernel, solver, trainer, losses.
- [`igl.kernels`](kernels.md) — operator zoo + registration API.
- [`igl.matryoshka`](matryoshka.md) — samplers + dimension curve helpers.
- [`igl.models`](models.md) — sklearn-compatible estimators.
- [`igl.metrics`](metrics.md) — cross-task `d_eff` comparison + elbow detectors.
- [`igl.nn`](nn.md) — bare PyTorch `IGLModule`.
- [`igl.spd`](spd.md) — Riemannian extension (AIRM, log-Eig, orthogonality, reconstruction).
- [`igl.data`](data.md) — synthetic data generators.
- [`igl.viz`](viz.md) — optional matplotlib helpers.

## Source layout

```
src/igl/
├── __init__.py          # flat public surface
├── config.py            # frozen dataclasses
├── exceptions.py        # IGLError + subclasses
├── types.py             # Protocols + StrEnums + Literal companions
├── core/                # encoder, kernel, solver, trainer, losses, normalization
├── kernels/             # 9 log-space operators + registry
├── matryoshka/          # samplers + dimension curve
├── models/              # sklearn estimators
├── metrics/             # d_eff comparison + elbow detectors
├── nn/                  # IGLModule
├── spd/                 # AIRM + log-Eig + orthogonality + reconstruction
├── data/                # synthetic generators
└── viz/                 # matplotlib helpers (gated behind [viz])
```
