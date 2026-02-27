from __future__ import annotations

import torch

from torchregress.utils import PropensityEstimator, ipw_weights


def test_ipw_weights_basic_behavior() -> None:
    p = torch.tensor([0.2, 0.5, 0.8])
    w = ipw_weights(p, normalize=False)
    expected = torch.tensor([5.0, 2.0, 1.25])
    assert torch.allclose(w, expected)


def test_ipw_weights_with_observed() -> None:
    p = torch.tensor([0.2, 0.7, 0.8])
    obs = torch.tensor([1.0, 0.0, 1.0])
    w = ipw_weights(p, observed=obs, normalize=False)
    expected = torch.tensor([5.0, 1.0 / 0.3, 1.25])
    assert torch.allclose(w, expected)


def test_propensity_estimator_fit_predict() -> None:
    torch.manual_seed(0)
    x = torch.randn(200, 3)
    logits = 0.8 * x[:, 0] - 0.4 * x[:, 1]
    p = torch.sigmoid(logits)
    observed = torch.bernoulli(p).long()

    est = PropensityEstimator()
    est.fit(x, observed)
    pred = est.predict_proba(x)
    assert pred.shape == observed.shape
    assert torch.all(pred > 0.0)
    assert torch.all(pred < 1.0)
