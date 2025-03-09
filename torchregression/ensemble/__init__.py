"""
Ensemble methods for regression.

This module provides implementations of ensemble techniques
for regression models, including deep ensembles, batch ensembles,
and specialized ensembles for uncertainty quantification.
"""

# Import ensemble models
from .ensemble import (
    BatchEnsembleLinear,
    BaseEnsembleModel,
    HeteroscedasticEnsembleModel,
    DeepEnsemble,
    HeteroscedasticBatchEnsembleModel
)

__all__ = [
    'BatchEnsembleLinear',
    'BaseEnsembleModel',
    'HeteroscedasticEnsembleModel',
    'DeepEnsemble',
    'HeteroscedasticBatchEnsembleModel'
]
