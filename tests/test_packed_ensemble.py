"""Tests for BatchEnsembleRegressor."""

import torch

from torchregress.ensemble import BatchEnsembleMLPBackbone, BatchEnsembleRegressor
from torchregress.ensemble.layers import BatchEnsembleLinear
from torchregress.ensemble.packed import _scale_batch_ensemble_factors


def test_packed_ensemble_heteroscedastic_shapes() -> None:
    bb = BatchEnsembleMLPBackbone(2, 8, ensemble_size=3, hidden_dims=[8])
    m = BatchEnsembleRegressor(
        bb,
        feature_dim=bb.feature_dim,
        output_dim=1,
        ensemble_size=3,
        heteroscedastic=True,
        alpha=1.0,
    )
    x = torch.randn(6, 2)
    d = m(x)
    assert d["means"].shape == (6, 3, 1)
    assert d["log_vars"].shape == (6, 3, 1)
    p = m.predict_output(x)
    assert p.mean.shape == (6, 1)
    assert p.member_means.shape == (6, 3, 1)
    assert p.std_epistemic.shape == (6, 1)
    assert p.aleatoric_variance is not None and p.aleatoric_variance.shape == (6, 1)


def test_packed_ensemble_homoscedastic_no_aleatoric() -> None:
    bb = BatchEnsembleMLPBackbone(1, 4, ensemble_size=2, hidden_dims=[4])
    m = BatchEnsembleRegressor(
        bb,
        feature_dim=bb.feature_dim,
        output_dim=1,
        ensemble_size=2,
        heteroscedastic=False,
        alpha=1.0,
    )
    x = torch.randn(4, 1)
    d = m(x)
    assert set(d.keys()) == {"means"}
    p = m.predict_output(x)
    assert p.aleatoric_variance is None
    assert p.predictive_variance.shape == (4, 1)


def test_alpha_scales_batch_ensemble_vectors() -> None:
    layer = BatchEnsembleLinear(3, 2, ensemble_size=2)
    r0 = layer.r_vectors.detach().clone()
    s0 = layer.s_vectors.detach().clone()
    _scale_batch_ensemble_factors(layer, 2.0)
    assert torch.allclose(layer.r_vectors, r0 * 2.0)
    assert torch.allclose(layer.s_vectors, s0 * 2.0)
