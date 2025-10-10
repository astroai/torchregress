"""
Ensemble models for regression tasks with uncertainty estimation.
"""

# Make classes directly accessible from the ensemble module
from .advanced import (
    BayesianModelAveraging,
    DynamicEnsembleWeighting,
    EnsembleCalibration,
    StackingEnsemble,
)
from .base import BaseEnsembleModel
from .layers import BatchEnsembleLinear
from .models import DeepEnsemble, HeteroscedasticBatchEnsembleModel, HeteroscedasticEnsembleModel
from .swag import MultiSWAG, SWAG

__all__ = [
    "BaseEnsembleModel",
    "BatchEnsembleLinear",
    "HeteroscedasticEnsembleModel",
    "DeepEnsemble",
    "HeteroscedasticBatchEnsembleModel",
    "BayesianModelAveraging",
    "StackingEnsemble",
    "DynamicEnsembleWeighting",
    "EnsembleCalibration",
    "SWAG",
    "MultiSWAG",
]
