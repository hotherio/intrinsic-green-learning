"""Symmetric Positive Definite (SPD) extension: AIRM loss, log-Eig, orthogonality, SPD reconstruction.

Not flattened into the top-level ``igl`` namespace. Import explicitly::

    from igl.spd import AIRMLoss, IGLReconSPDClassifier, LogEigVectorizer
"""

from igl.spd.airm import AIRMLoss, airm_loss
from igl.spd.diagnostics import geometry_report, metric_summary
from igl.spd.linalg import MatrixMethod, matrix_exp_sym, matrix_log_sym, matrix_pow_sym, unpack_sym_vec
from igl.spd.log_eig import LogEigVectorizer
from igl.spd.orthogonality import (
    ConformalityPenalty,
    OrthogonalityPenalty,
    conformal_factor,
    conformality_loss,
    init_encoder_orthogonal_,
    jacobian,
    orthogonality_loss,
    pullback_metric,
)
from igl.spd.reconstruction import IGLReconSPDClassifier

__all__ = [
    "AIRMLoss",
    "ConformalityPenalty",
    "IGLReconSPDClassifier",
    "LogEigVectorizer",
    "MatrixMethod",
    "OrthogonalityPenalty",
    "airm_loss",
    "geometry_report",
    "conformal_factor",
    "conformality_loss",
    "init_encoder_orthogonal_",
    "jacobian",
    "matrix_exp_sym",
    "matrix_log_sym",
    "matrix_pow_sym",
    "metric_summary",
    "orthogonality_loss",
    "pullback_metric",
    "unpack_sym_vec",
]
