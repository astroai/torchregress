"""
Ensemble models for regression tasks with uncertainty estimation.
"""

# Make classes directly accessible from the ensemble module
from .base import BaseEnsembleModel, EnsembleFitConfig
from .bnn import BayesianNeuralNetwork, HeteroscedasticBNN, VariationalLinear
from .combiners import (
    SoftmaxModelCombiner,
    StackingEnsemble,
)
from .layers import BatchEnsembleLinear
from .mc_dropout import MCDropoutWrapper, enable_dropout
from .models import (
    BatchEnsembleMLPBackbone,
    BinnedPDFEnsembleModel,
    CumulativeLinkEnsembleModel,
    HeteroscedasticBatchEnsembleModel,
    HeteroscedasticEnsembleModel,
    MDNEnsembleModel,
    RandomPartitionEnsembleModel,
)
from .packed import (
    BatchEnsembleOutput,
    BatchEnsembleRegressor,
)
from .swag import SWAG, MultiSWAG
from .utils import parse_heteroscedastic_output

__all__ = [
    "BaseEnsembleModel",
    "EnsembleFitConfig",
    "BatchEnsembleLinear",
    "HeteroscedasticEnsembleModel",
    "BatchEnsembleMLPBackbone",
    "BinnedPDFEnsembleModel",
    "CumulativeLinkEnsembleModel",
    "MDNEnsembleModel",
    "RandomPartitionEnsembleModel",
    "HeteroscedasticBatchEnsembleModel",
    "BatchEnsembleRegressor",
    "BatchEnsembleOutput",
    "SoftmaxModelCombiner",
    "StackingEnsemble",
    "SWAG",
    "MultiSWAG",
    "parse_heteroscedastic_output",
    # MC-Dropout
    "MCDropoutWrapper",
    "enable_dropout",
    # Bayesian Neural Networks
    "VariationalLinear",
    "BayesianNeuralNetwork",
    "HeteroscedasticBNN",
]
