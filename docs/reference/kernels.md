# `igl.kernels`

The bundled Green's-function operator zoo and the registration API.

::: igl.kernels._registry.Operator
    options:
      show_root_heading: true
      show_source: false

::: igl.kernels._registry.register_operator
    options:
      show_root_heading: true
      show_source: false

::: igl.kernels._registry.get_operator
    options:
      show_root_heading: true
      show_source: false

::: igl.kernels._registry.list_operators
    options:
      show_root_heading: true
      show_source: false

## Bundled operators

All nine operators are registered eagerly when ``igl.kernels`` is imported.
Each one is a tiny module under ``src/igl/kernels/``; the module names match
the operator names exported via :func:`list_operators`.

| Name | Oscillatory | Form |
|---|---|---|
| `gaussian` | no | `exp(-d² / (2σ²))` |
| `laplacian` | no | `exp(-|d| / σ)` |
| `cauchy` | no | `1 / (1 + d²/σ²)` |
| `yukawa` | no | `exp(-|d|/σ)` (Laplacian shape, separate identity) |
| `multiquadric` | no | `1 / sqrt(1 + d²/σ²)` |
| `helmholtz` | yes | `exp(-|d|/σ) · cos(π d / σ)` |
| `gabor` | yes | `exp(-d²/(2σ²)) · cos(π d / σ)` |
| `mexican_hat` | yes | `(1 - d²/σ²) · exp(-d²/(2σ²))` |
| `soft_box` | no | smooth indicator of `[-σ, σ]` |

To register your own operator, supply a callable matching
[`OperatorFn`][igl.types.OperatorFn] (returns ``(log_abs, sign)``) and pass
``is_oscillatory=True`` if the function can take negative values.
