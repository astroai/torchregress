"""Strong-default neural networks for tabular regression.

``TabularMLP`` (periodic numeric embeddings, SiLU blocks, dropout),
``TabularPreprocessor`` (robust scaling, smooth clipping, NaN handling) and
``fit_tabular`` (AdamW, one-cycle schedule, early stopping, target standardisation)
work with any torchregress loss.
"""

from .copula_spline import GaussianCopulaSplineHead
from .flow_matching import FlowMatchingMLP, cfm_loss, sample_cfm
from .ft_transformer import FeatureTokenizer, FTTransformer
from .normalizing_flow_map import NormalizingFlowMap
from .preprocessing import TabularPreprocessor
from .tabular_mlp import PeriodicEmbedding, TabularMLP
from .training import TabularEnsembleFit, TabularFit, fit_tabular, fit_tabular_ensemble

__all__ = [
    "FTTransformer",
    "FeatureTokenizer",
    "FlowMatchingMLP",
    "GaussianCopulaSplineHead",
    "NormalizingFlowMap",
    "cfm_loss",
    "sample_cfm",
    "TabularMLP",
    "PeriodicEmbedding",
    "TabularPreprocessor",
    "TabularFit",
    "TabularEnsembleFit",
    "fit_tabular",
    "fit_tabular_ensemble",
]
