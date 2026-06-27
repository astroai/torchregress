import pytest
import torch
import torch.nn as nn

from torchregress.ensemble.combiners import SoftmaxModelCombiner
from torchregress.ensemble.models import BaseEnsembleModel, HeteroscedasticEnsembleModel
from torchregress.ensemble.packed import BatchEnsembleRegressor
from torchregress.ensemble.swag import MultiSWAG


class ToyModel(nn.Module):
    def __init__(self, in_features: int = 2, out_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ToyHeteroModel(nn.Module):
    def __init__(self, in_features: int = 2, out_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_member_factory_and_base_seed():
    # Test using member_factory
    ensemble = BaseEnsembleModel(
        member_factory=lambda idx, seed: ToyModel(2, 1), ensemble_size=3, base_seed=42
    )
    assert len(ensemble.models) == 3
    # Check they have different weights due to different manual seeds
    w0 = ensemble.models[0].linear.weight
    w1 = ensemble.models[1].linear.weight
    assert not torch.equal(w0, w1)


def test_reset_parameters_warning():
    model_instance = ToyModel(2, 1)
    # If reset_parameters=False, copy-construction from instance keeps identical parameters
    # and should raise a UserWarning
    with pytest.warns(UserWarning, match="identical parameter values"):
        BaseEnsembleModel(base_model=model_instance, ensemble_size=2, reset_parameters=False)


def test_predict_correction():
    ensemble = BaseEnsembleModel(
        member_factory=lambda idx, seed: ToyModel(2, 1), ensemble_size=3, base_seed=42
    )
    x = torch.randn(5, 2)
    # Default correction=0
    res0 = ensemble.predict(x, correction=0)
    # correction=1 (Bessel's)
    res1 = ensemble.predict(x, correction=1)

    # Calculate expected variances manually
    preds = ensemble(x)  # [3, 5, 1]
    var0 = torch.var(preds, dim=0, correction=0)
    var1 = torch.var(preds, dim=0, correction=1)

    assert torch.allclose(res0["variance"], var0)
    assert torch.allclose(res1["variance"], var1)


def test_predict_full_covariance_contraction():
    # BaseEnsembleModel full covariance should be [B, D, D]
    ensemble = BaseEnsembleModel(
        member_factory=lambda idx, seed: ToyModel(2, 3),  # output size 3
        ensemble_size=3,
        base_seed=42,
    )
    x = torch.randn(5, 2)
    res = ensemble.predict_full_covariance(x, correction=0)
    assert res["covariance"].shape == (5, 3, 3)  # [batch, out_dim, out_dim]

    # HeteroscedasticEnsembleModel full covariance should be [B, D, D]
    het_ens = HeteroscedasticEnsembleModel(
        member_factory=lambda idx, seed: ToyHeteroModel(2, 3),  # outputs 6 values
        ensemble_size=3,
        base_seed=42,
    )
    res_het = het_ens.predict_full_covariance(x, correction=0)
    assert res_het["epistemic_covariance"].shape == (5, 3, 3)
    assert res_het["aleatoric_covariance"].shape == (5, 3, 3)
    assert res_het["total_covariance"].shape == (5, 3, 3)


def test_multiswag_decomposition():
    base = ToyModel(2, 1)
    mswag = MultiSWAG(base, n_models=2, max_num_models=2)
    # Collect fake snapshots
    for m in mswag.swag_models:
        m.collect_model(base)
        m.collect_model(base)

    x = torch.randn(4, 2)
    mean, epistemic, aleatoric = mswag.predict_with_uncertainty(
        x, n_samples=3, scale=0.5, correction=0
    )
    assert mean.shape == (4, 1)
    assert epistemic.shape == (4, 1)
    assert aleatoric.shape == (4, 1)



