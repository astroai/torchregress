from __future__ import annotations

import torch

from torchregress.losses._legacy_args import (
    is_legacy_mask_argument,
    resolve_legacy_cov_mask_weights,
    resolve_legacy_log_variance_kwarg,
)
from torchregress.utils.distributions import normal_cdf
from torchregress.utils.gaussian_output import (
    parse_heteroscedastic_output,
    split_mean_log_variance,
    variance_from_logvar,
)
from torchregress.utils.numpy_stats import subsample_rows, winsorize


def test_split_mean_log_variance_tuple_and_concat() -> None:
    mean = torch.tensor([[1.0, 2.0]])
    log_var = torch.tensor([[0.1, -0.2]])
    m1, lv1 = split_mean_log_variance((mean, log_var))
    assert torch.allclose(m1, mean)
    assert torch.allclose(lv1, log_var)

    concat = torch.cat([mean, log_var], dim=-1)
    m2, lv2 = split_mean_log_variance(concat)
    assert torch.allclose(m2, mean)
    assert torch.allclose(lv2, log_var)


def test_split_mean_log_variance_mean_only_zeros() -> None:
    mean = torch.tensor([[3.0]])
    m, lv = split_mean_log_variance(mean, mean_only_log_var="zeros")
    assert torch.allclose(m, mean)
    assert torch.allclose(lv, torch.zeros_like(mean))


def test_parse_heteroscedastic_output_dict() -> None:
    mean = torch.randn(4, 2)
    log_var = torch.randn(4, 2)
    m, lv = parse_heteroscedastic_output({"means": mean, "log_vars": log_var})
    assert torch.allclose(m, mean)
    assert torch.allclose(lv, log_var)


def test_variance_from_logvar_clamps() -> None:
    log_var = torch.tensor([[-20.0, 10.0]])
    var = variance_from_logvar(log_var)
    assert var.min().item() >= 1e-8


def test_legacy_mask_resolution() -> None:
    legacy_mask = torch.ones(3, dtype=torch.bool)
    weights = torch.tensor([1.0, 2.0, 3.0])
    mask, w, cov = resolve_legacy_cov_mask_weights(legacy_mask, weights, None)
    assert mask is legacy_mask
    assert w is weights
    assert cov is None
    assert is_legacy_mask_argument(legacy_mask, None)


def test_legacy_log_variance_kwarg_resolution() -> None:
    legacy_mask = torch.ones(3, dtype=torch.bool)
    weights = torch.tensor([1.0, 2.0, 3.0])
    mask, w, log_var = resolve_legacy_log_variance_kwarg(legacy_mask, weights, False)
    assert mask is legacy_mask
    assert w is weights
    assert log_var is False

    mask2, w2, log_var2 = resolve_legacy_log_variance_kwarg(None, None, None)
    assert mask2 is None and w2 is None and log_var2 is None


def test_normal_cdf_matches_erf() -> None:
    z = torch.tensor([-1.0, 0.0, 1.5])
    cdf = normal_cdf(z)
    expected = 0.5 * (1.0 + torch.erf(z / torch.sqrt(torch.tensor(2.0))))
    assert torch.allclose(cdf, expected)


def test_numpy_stats_subsample_and_winsorize() -> None:
    import numpy as np

    X = np.arange(20, dtype=float).reshape(10, 2)
    small = subsample_rows(X, 4, random_state=0)
    assert small.shape[0] == 4
    clipped = winsorize(X, 0.1)
    assert clipped.shape == X.shape
