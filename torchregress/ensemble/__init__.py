"""
Ensemble models for regression tasks with uncertainty estimation.
"""

# Make classes directly accessible from the ensemble module
from .base import BaseEnsembleModel
from .bnn import BayesianNeuralNetwork, HeteroscedasticBNN, VariationalLinear
from .combiners import (
    BayesianModelAveraging,
    DynamicEnsembleWeighting,
    StackingEnsemble,
)
from .layers import BatchEnsembleLinear
from .mc_dropout import MCDropoutModel, MCDropoutWrapper, enable_dropout
from .models import (
    DeepEnsemble,
    HeteroscedasticBatchEnsembleModel,
    HeteroscedasticEnsembleModel,
)
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
    # MC-Dropout
    "MCDropoutWrapper",
    "MCDropoutModel",
    "enable_dropout",
    # Bayesian Neural Networks
    "VariationalLinear",
    "BayesianNeuralNetwork",
    "HeteroscedasticBNN",
]
