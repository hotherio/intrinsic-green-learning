# `igl.preprocessing`

Sklearn-compatible preprocessing transformers for EEG / SPD-valued
data. Subpackage gated behind the `[eeg]` extra (which ships
`pyriemann`); plain installs will raise
[`IGLDependencyError`][igl.IGLDependencyError] on import.

```
pip install intrinsic-green-learning[eeg]
```

## Auto covariances

::: igl.preprocessing.covariances.AutoCovariances
    options:
      show_root_heading: true
      show_source: false

::: igl.preprocessing.covariances.CovarianceEstimator
    options:
      show_root_heading: true
      show_source: false
