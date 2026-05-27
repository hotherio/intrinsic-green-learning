# About

Intrinsic Green Learning (IGL) consolidates two research codebases — an
NLP/embedding implementation and an EEG/SPD implementation — into a single
library with a stable public API.

## Origins

The theoretical framework and original prototypes live under
`hother/laplacian` (Knowledge Base + matryoshka implementations). The library
here is the **production-ready consolidation**: strict typing, full test
coverage, sklearn-compatible models, optional Riemannian extension.

## Roadmap

- **0.1.0** — Matryoshka classifier/regressor; SPD reconstruction with AIRM
  loss and orthogonality penalty; synthetic examples demonstrating
  $d_{\text{eff}}(\text{cls}) \le d_{\text{eff}}(\text{reg}) \le d_{\text{eff}}(\text{recon})$.
- **Post-0.1** — EEG/MOABB and NLP/BERT examples; `contrib/` variants (Hard
  Concrete gates, Direction-IGL, Deep IGL, TimeIGL, adaptive kernels).
- **Beyond** — Vision integrations, distributed training, ONNX export.

## Citing

If IGL helps your work, cite via the project's
[`CITATION.cff`](https://github.com/hotherio/intrinsic-green-learning/blob/main/CITATION.cff).
