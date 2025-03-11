"""
Ensemble models for regression tasks with uncertainty estimation.
"""

# Make classes directly accessible from the ensemble module
from .base import BaseEnsembleModel
from .layers import BatchEnsembleLinear
from .models import (
    HeteroscedasticEnsembleModel,
    DeepEnsemble,
    HeteroscedasticBatchEnsembleModel
)
