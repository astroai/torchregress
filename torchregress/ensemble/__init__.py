"""
Ensemble models for regression tasks with uncertainty estimation.
"""

# Make classes directly accessible from the ensemble module
from .combiners import (
    BayesianModelAveraging,
    DynamicEnsembleWeighting,
    StackingEnsemble,
)
from .base import BaseEnsembleModel
from .layers import BatchEnsembleLinear
from .models import DeepEnsemble, HeteroscedasticBatchEnsembleModel, HeteroscedasticEnsembleModel
from .swag import SWAG, MultiSWAG
from .utils import parse_heteroscedastic_output

__all__ = [
    "BaseEnsembleModel",
    "BatchEnsembleLinear",
    "HeteroscedasticEnsembleModel",
    "DeepEnsemble",
    "HeteroscedasticBatchEnsembleModel",
    "BayesianModelAveraging",
    "StackingEnsemble",
    "DynamicEnsembleWeighting",
    "SWAG",
    "MultiSWAG",
    "parse_heteroscedastic_output",
]
