"""Strong-default neural networks for tabular regression.

``TabularMLP`` (periodic numeric embeddings, SiLU blocks, dropout),
``TabularPreprocessor`` (robust scaling, smooth clipping, NaN handling) and
``fit_tabular`` (AdamW, one-cycle schedule, early stopping, target standardisation)
work with any torchregress loss.
"""

from .preprocessing import TabularPreprocessor
from .tabular_mlp import PeriodicEmbedding, TabularMLP
from .training import TabularEnsembleFit, TabularFit, fit_tabular, fit_tabular_ensemble

__all__ = [
    "TabularMLP",
    "PeriodicEmbedding",
    "TabularPreprocessor",
    "TabularFit",
    "TabularEnsembleFit",
    "fit_tabular",
    "fit_tabular_ensemble",
]
