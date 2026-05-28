"""scikit-learn-compatible estimators: classifier, regressor, autoencoder."""

from igl.models.autoencoder import IGLAutoencoder
from igl.models.classifier import IGLClassifier
from igl.models.regressor import IGLRegressor

__all__ = ["IGLAutoencoder", "IGLClassifier", "IGLRegressor"]
