"""
Ensemble models for regression tasks with uncertainty estimation.
"""

# Make classes directly accessible from the ensemble module
from .base import BaseEnsembleModel, EnsembleFitConfig
from .bnn import BayesianNeuralNetwork, HeteroscedasticBNN, VariationalLinear
from .combiners import (
    BayesianModelAveraging,
    DynamicEnsembleWeighting,
    StackingEnsemble,
)
from .layers import BatchEnsembleLinear
from .mc_dropout import MCDropoutModel, MCDropoutWrapper, enable_dropout
from .models import (
    BatchEnsembleMLPBackbone,
    BinnedPDFEnsembleModel,
    CumulativeLinkEnsembleModel,
    DeepEnsemble,
    HeteroscedasticBatchEnsembleModel,
    HeteroscedasticEnsembleModel,
    MDNEnsembleModel,
    RandomPartitionEnsembleModel,
)
from .packed import PackedEnsembleOutput, PackedEnsembleRegressor
from .swag import SWAG, MultiSWAG
from .utils import parse_heteroscedastic_output

__all__ = [
    "BaseEnsembleModel",
    "EnsembleFitConfig",
    "BatchEnsembleLinear",
    "HeteroscedasticEnsembleModel",
    "DeepEnsemble",
    "BatchEnsembleMLPBackbone",
    "BinnedPDFEnsembleModel",
    "CumulativeLinkEnsembleModel",
    "MDNEnsembleModel",
    "RandomPartitionEnsembleModel",
    "HeteroscedasticBatchEnsembleModel",
    "PackedEnsembleRegressor",
    "PackedEnsembleOutput",
    "BayesianModelAveraging",
    "StackingEnsemble",
    "DynamicEnsembleWeighting",
    "SWAG",
    "MultiSWAG",
    "parse_heteroscedastic_output",
    # MC-Dropout
    "MCDropoutWrapper",
    "MCDropoutModel",
    "enable_dropout",
    # Bayesian Neural Networks
    "VariationalLinear",
    "BayesianNeuralNetwork",
    "HeteroscedasticBNN",
]
