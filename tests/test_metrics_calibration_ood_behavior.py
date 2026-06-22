from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch

from torchregress.metrics import (
    EntropyScore,
    KernelDensityScore,
    MahalanobisDistance,
    TypicalityScore,
    calibration_score,
    entropy_score,
    expected_calibration_error,
    kernel_density_score,
    mahalanobis_distance,
    ood_metrics_report,
    typicality_score,
)


def test_calibration_score_matches_expected_calibration_error_construction():
    y_true = torch.linspace(-1, 1, 25).unsqueeze(-1)
    pred_mean = y_true.clone()
    pred_std = torch.full_like(y_true, 0.25)

    score = calibration_score(y_true, pred_mean, pred_std, n_levels=5)

    levels = torch.linspace(0.05, 0.95, 5)
    standard = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))
    quantiles = {float(q.item()): pred_mean + standard.icdf(q) * pred_std for q in levels}
    expected = expected_calibration_error(quantiles, y_true)

    for key in (
        "mean_absolute_calibration_error",
        "root_mean_squared_calibration_error",
        "maximum_calibration_error",
    ):
        assert torch.allclose(
            torch.as_tensor(score[key]),
            torch.as_tensor(expected[key]),
            atol=1e-6,
            rtol=1e-6,
        )


def test_ood_metric_classes_match_functional_variants_for_deterministic_inputs():
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    mean = torch.tensor([0.5, 0.5])
    # Pin dtype/device to ``x`` so the fixture doesn't implicitly rely on
    # the metric module handling dtype/device of input fixtures internally.
    cov = torch.eye(2, device=x.device, dtype=x.dtype) * 0.5

    md_metric = MahalanobisDistance()
    md_metric.update(x, mean, cov)
    md_class = md_metric.compute()
    md_func = mahalanobis_distance(x, mean, cov, reduction="mean")
    assert torch.allclose(md_class, md_func, atol=1e-6)

    x_ref = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    kde_metric = KernelDensityScore(bandwidth=0.5)
    kde_metric.update(x, x_ref)
    kde_class = kde_metric.compute()
    kde_func = kernel_density_score(x, x_ref, bandwidth=0.5, reduction="mean")
    assert torch.allclose(kde_class, kde_func, atol=1e-6)

    samples = torch.randn(16, 3, 1)
    ent_metric = EntropyScore(n_bins=8)
    ent_metric.update(samples)
    ent_class = ent_metric.compute()
    ent_func = entropy_score(samples, n_bins=8, reduction="mean")
    assert torch.allclose(ent_class, ent_func, atol=1e-6)


def test_typicality_score_accepts_dict_and_tuple_and_report_returns_expected_keys():
    torch.manual_seed(0)
    mean = torch.zeros(6, 1)
    var = torch.ones(6, 1) * 0.25
    x = torch.randn(6, 1)

    tuple_scores = typicality_score((mean, var), x=x, reduction="none")
    dict_scores = typicality_score({"mean": mean, "variance": var}, x=x, reduction="none")
    assert tuple_scores.shape == (6,)
    assert torch.allclose(tuple_scores, dict_scores, atol=1e-6)

    torch.manual_seed(0)
    metric = TypicalityScore(n_samples=8)
    metric.update({"mean": mean, "variance": var})
    metric_value = metric.compute()
    assert metric_value.ndim == 0
    assert torch.isfinite(metric_value)

    report = ood_metrics_report(
        model_output={"mean": mean, "variance": var},
        x_test=x,
        x_reference=torch.randn(10, 1),
        mean=torch.tensor([0.0]),
        # Pin dtype/device to ``mean`` so the fixture doesn't implicitly rely
        # on the metric module handling dtype/device of input fixtures internally.
        cov=torch.eye(1, device=mean.device, dtype=mean.dtype),
        samples=torch.randn(12, 6, 1),
    )
    assert {"typicality_score", "kernel_density", "mahalanobis_distance", "entropy"} <= set(report)


def test_calibration_score_supports_multitarget_tiny_std_and_numpy_output() -> None:
    y_true = torch.tensor([[0.0, 1.0], [0.5, -0.5], [1.0, 0.0]])
    pred_mean = y_true + 0.05
    pred_std = torch.full_like(y_true, 1e-12)

    score_t = calibration_score(y_true, pred_mean, pred_std, n_levels=4)
    score_np = calibration_score(
        y_true.numpy(), pred_mean.numpy(), pred_std.numpy(), n_levels=4, as_numpy=True
    )

    for key, value in score_t.items():
        assert torch.isfinite(torch.as_tensor(value))
        assert isinstance(score_np[key], (float, np.ndarray))


def test_expected_calibration_error_as_numpy_returns_numpy_diagnostics() -> None:
    y_true = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    y_pred_quantiles = {
        0.1: np.array([[-0.5], [0.5], [1.5]], dtype=np.float32),
        0.5: np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        0.9: np.array([[0.5], [1.5], [2.5]], dtype=np.float32),
    }
    # cast to avoid dict variance issues in mypy
    result = expected_calibration_error(
        cast(dict[float, Any], y_pred_quantiles), y_true, return_diagnostics=True, as_numpy=True
    )
    assert isinstance(result["mean_absolute_calibration_error"], float)
    assert isinstance(result["bin_errors"], np.ndarray)
    assert result["bin_errors"].shape == (3,)


def test_ood_metrics_handle_singular_covariance_zero_range_entropy_and_multivariate_shapes() -> (
    None
):
    x = torch.tensor([[0.0, 0.0], [1.0, -1.0], [0.5, 0.2]])
    mean = torch.tensor([0.0, 0.0])
    singular_cov = torch.tensor([[1.0, 1.0], [1.0, 1.0]])  # rank-deficient -> fallback path

    md = mahalanobis_distance(x, mean, singular_cov, reduction="none")
    assert md.shape == (3,)
    assert torch.all(torch.isfinite(md))

    x_ref = torch.zeros(5, 2)
    kde = kernel_density_score(x, x_ref, bandwidth=0.3, reduction="none")
    assert kde.shape == (3,)
    assert torch.all(torch.isfinite(kde))

    constant_samples = torch.ones(7, 3, 2) * 4.2
    ent = entropy_score(constant_samples, n_bins=8, reduction="none")
    assert ent.shape == (3,)
    assert torch.all(torch.isfinite(ent))
    assert torch.all(ent >= 0)

    typical = typicality_score(
        {"mean": torch.zeros(3, 2), "variance": torch.ones(3, 2) * 0.2},
        x=torch.zeros(3, 2),
        reduction="none",
    )
    assert typical.shape == (3,)
    assert torch.all(torch.isfinite(typical))
