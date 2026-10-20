"""Strong-default neural networks for tabular regression.

``TabularMLP`` (periodic numeric embeddings, SiLU blocks, dropout),
``TabularPreprocessor`` (robust scaling, smooth clipping, NaN handling) and
``fit_tabular`` (AdamW, one-cycle schedule, early stopping, target standardisation)
work with any torchregress loss.
"""

from .ft_transformer import FeatureTokenizer, FTTransformer, PZRecipe
from .preprocessing import TabularPreprocessor
from .pz_recipe import PZRecipeFit, fit_pz_recipe
from .tabular_mlp import PeriodicEmbedding, TabularMLP
from .training import TabularEnsembleFit, TabularFit, fit_tabular, fit_tabular_ensemble

__all__ = [
    "FTTransformer",
    "FeatureTokenizer",
    "PZRecipe",
    "PZRecipeFit",
    "fit_pz_recipe",
    "TabularMLP",
    "PeriodicEmbedding",
    "TabularPreprocessor",
    "TabularFit",
    "TabularEnsembleFit",
    "fit_tabular",
    "fit_tabular_ensemble",
]
