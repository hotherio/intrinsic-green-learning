# `igl.spd`

Riemannian extension for SPD-valued data. Not flattened into the top-level
``igl`` namespace — import explicitly::

    from igl.spd import AIRMLoss, IGLReconSPDClassifier, LogEigVectorizer

## Linear algebra

::: igl.spd.linalg.matrix_log_sym
    options:
      show_root_heading: true
      show_source: false

::: igl.spd.linalg.matrix_exp_sym
    options:
      show_root_heading: true
      show_source: false

::: igl.spd.linalg.matrix_pow_sym
    options:
      show_root_heading: true
      show_source: false

::: igl.spd.linalg.unpack_sym_vec
    options:
      show_root_heading: true
      show_source: false

## Log-Eig vectorizer

::: igl.spd.log_eig.LogEigVectorizer
    options:
      show_root_heading: true
      show_source: false

## AIRM loss

::: igl.spd.airm.airm_loss
    options:
      show_root_heading: true
      show_source: false

::: igl.spd.airm.AIRMLoss
    options:
      show_root_heading: true
      show_source: false

## Orthogonality penalty

::: igl.spd.orthogonality.OrthogonalityPenalty
    options:
      show_root_heading: true
      show_source: false

::: igl.spd.orthogonality.jacobian
    options:
      show_root_heading: true
      show_source: false

::: igl.spd.orthogonality.pullback_metric
    options:
      show_root_heading: true
      show_source: false

::: igl.spd.orthogonality.orthogonality_loss
    options:
      show_root_heading: true
      show_source: false

::: igl.spd.orthogonality.init_encoder_orthogonal_
    options:
      show_root_heading: true
      show_source: false

## Reconstruction classifier

::: igl.spd.reconstruction.IGLReconSPDClassifier
    options:
      show_root_heading: true
      show_source: false
