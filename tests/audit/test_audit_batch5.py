"""Regression tests for audit batch 5 (PRD-*, UTL-*, VIZ-*) of the 0.3.0 release prep.

Each test reproduces one audit finding and pins the fixed behaviour. References come from
closed forms, :mod:`scipy` (core) or :mod:`sklearn` / :mod:`matplotlib` (``test`` / ``viz``
extras).  PRD-001 (truncation semantics) is resolved: tail mass is kept on the grid margins.
"""

from __future__ import annotations

import importlib
import math
import pathlib
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from scipy import special, stats  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

from torchregress.comparison import compute_point_metrics  # noqa: E402
from torchregress.prediction import quantiles_to_density_grid  # noqa: E402
from torchregress.utils import (  # noqa: E402
    BoxCoxTransform,
    BSplineDensityBasis,
    YeoJohnsonTransform,
    ipw_weights,
    masked_mean,
    masked_reduction,
    masked_sum,
    split_mean_log_variance,
    validate_positive,
)
from torchregress.utils import openml_relaxed as om  # noqa: E402
from torchregress.utils.distributions import normal_cdf  # noqa: E402
from torchregress.utils.gaussian_output import parse_heteroscedastic_output  # noqa: E402
from torchregress.utils.ordinal import labels_to_levels  # noqa: E402
from torchregress.utils.tensor_ops import calculate_gaussian_nll  # noqa: E402
from torchregress.utils.validation import validate_quantile, validate_range  # noqa: E402
from torchregress.viz import (  # noqa: E402
    plot_calibration_curve,
    plot_causal_uplift_qini,
    plot_gaussian_reliability_diagram,
    plot_performance_comparison,
    plot_pit_histogram,
    plot_prediction_intervals,
    plot_qq_plot,
    plot_reliability_diagram,
    plot_residual_histogram,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------- PRD-001
# PRD-001: quantiles_to_density_grid does not preserve the CDF at the quantile knots.


def _cdf_at(support, density, x):
    s = support.numpy()
    d = density.numpy()
    F = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(s))])
    return np.interp(x, s, F)


def test_PRD_001_cdf_preserved_at_knots_asymmetric_levels():
    levels = [0.05, 0.5, 0.75]
    q = torch.tensor([[stats.norm.ppf(t) for t in levels]], dtype=torch.float64)
    s, d = quantiles_to_density_grid(q, levels, n_support=4001, range_margin=0.5)
    np.testing.assert_allclose(_cdf_at(s[0], d[0], q[0].numpy()), levels, atol=0.01)


def test_PRD_001_tail_mass_on_margins():
    # Resolved: tau_0 and 1 - tau_K sit uniformly on the margins (no truncation).
    levels = [0.05, 0.5, 0.75]
    q = torch.tensor([[stats.norm.ppf(t) for t in levels]], dtype=torch.float64)
    s, d = quantiles_to_density_grid(q, levels, n_support=4001, range_margin=0.5)
    F = _cdf_at(s[0], d[0], s[0].numpy())
    np.testing.assert_allclose(F[-1], 1.0, atol=1e-6)
    np.testing.assert_allclose(_cdf_at(s[0], d[0], q[0, 0].item()), 0.05, atol=0.01)
    assert "truncated" not in (quantiles_to_density_grid.__doc__ or "")


def test_PRD_001_density_matches_piecewise_uniform_value():
    # Linear-interpolated quantile function -> density (tau_{k+1}-tau_k)/(q_{k+1}-q_k).
    levels = [0.1, 0.5, 0.9]
    q = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float64)
    s, d = quantiles_to_density_grid(q, levels, n_support=2001)
    inside = (s[0] > 0.1) & (s[0] < 1.9)
    torch.testing.assert_close(d[0][inside], torch.full_like(d[0][inside], 0.4), rtol=1e-2, atol=0)


# ---------------------------------------------------------------------- PRD-002
# PRD-002: quantiles_to_density_grid repairs crossing quantiles with cummax instead of the


def test_PRD_002_crossing_quantiles_equal_sorted_quantiles():
    levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    crossed = torch.tensor([[0.0, 1.0, 3.0, 2.0, 4.0]], dtype=torch.float64)
    rearranged = torch.sort(crossed, dim=1).values
    s1, d1 = quantiles_to_density_grid(crossed, levels, n_support=501)
    s2, d2 = quantiles_to_density_grid(rearranged, levels, n_support=501)
    torch.testing.assert_close(s1, s2)
    torch.testing.assert_close(d1, d2)


# ---------------------------------------------------------------------- PRD-003
# PRD-003: comparison.compute_point_metrics broadcasts [N, 1] predictions against [N]


def test_PRD_003_column_predictions_vs_flat_targets():
    g = torch.Generator().manual_seed(0)
    y_true = torch.randn(50, generator=g, dtype=torch.float64)
    y_pred = (y_true + 0.1 * torch.randn(50, generator=g, dtype=torch.float64)).unsqueeze(1)
    out = compute_point_metrics(y_pred, y_true)
    yp, yt = y_pred.numpy().ravel(), y_true.numpy()
    np.testing.assert_allclose(out["MSE"], mean_squared_error(yt, yp), rtol=1e-6)
    np.testing.assert_allclose(out["MAE"], mean_absolute_error(yt, yp), rtol=1e-6)
    np.testing.assert_allclose(out["R2"], r2_score(yt, yp), rtol=1e-6)


# ---------------------------------------------------------------------- PRD-004
# PRD-004: compute_point_metrics pools R2 over all outputs around the *global* mean


def test_PRD_004_multioutput_r2_matches_sklearn():
    g = torch.Generator().manual_seed(0)
    y_true = torch.stack(
        [
            torch.randn(200, generator=g, dtype=torch.float64),
            100.0 + torch.randn(200, generator=g, dtype=torch.float64),
        ],
        dim=1,
    )
    y_pred = y_true.mean(dim=0, keepdim=True).expand_as(y_true)  # no skill at all
    out = compute_point_metrics(y_pred, y_true)
    np.testing.assert_allclose(out["R2"], r2_score(y_true.numpy(), y_pred.numpy()), atol=1e-6)


# ---------------------------------------------------------------------- UTL-001
# UTL-001: masked_mean / masked_sum / masked_reduction propagate NaN from masked-out entries.


def _masked_data():
    x = torch.tensor([1.0, float("nan"), 3.0, float("inf")], dtype=torch.float64)
    m = torch.tensor([True, False, True, False])
    return x, m


def test_UTL_001_masked_mean_ignores_masked_nan():
    x, m = _masked_data()
    assert torch.equal(masked_mean(x, m), torch.tensor(2.0, dtype=torch.float64))


def test_UTL_001_masked_sum_ignores_masked_nan():
    x, m = _masked_data()
    assert torch.equal(masked_sum(x, m), torch.tensor(4.0, dtype=torch.float64))


def test_UTL_001_masked_reduction_mean_ignores_masked_nan():
    x, m = _masked_data()
    assert torch.equal(masked_reduction(x, m, "mean"), torch.tensor(2.0, dtype=torch.float64))


# ---------------------------------------------------------------------- UTL-002
# UTL-002: ipw_weights and labels_to_levels hard-code float32 (ignore float64 inputs /


def test_UTL_002_ipw_weights_keeps_float64():
    p = torch.tensor([0.123456789012345, 0.5, 0.987654321], dtype=torch.float64)
    w = ipw_weights(p, clip_min=1e-4, clip_max=1 - 1e-4, normalize=False)
    assert w.dtype == torch.float64
    torch.testing.assert_close(w, 1.0 / p, rtol=1e-12, atol=0)


def test_UTL_002_labels_to_levels_follows_default_dtype():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        lv = labels_to_levels(torch.tensor([0, 2, 1]), 3)
    finally:
        torch.set_default_dtype(old)
    assert lv.dtype == torch.float64


# ---------------------------------------------------------------------- UTL-003
# UTL-003: calculate_gaussian_nll accepts var with dim == residuals.dim() - 1


def _ref(r, v):
    r = r.numpy()
    v = v.numpy()
    return np.array(
        [-stats.norm.logpdf(r[i], 0.0, math.sqrt(v[i])).sum() for i in range(r.shape[0])]
    )


def test_UTL_003_per_sample_variance_B_ne_D():
    g = torch.Generator().manual_seed(0)
    r = torch.randn(5, 2, generator=g, dtype=torch.float64)
    v = torch.rand(5, generator=g, dtype=torch.float64) + 0.5
    out = calculate_gaussian_nll(r, v, eps=0.0)
    np.testing.assert_allclose(out.numpy(), _ref(r, v), rtol=1e-10)


def test_UTL_003_per_sample_variance_B_eq_D_silently_wrong():
    r = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], dtype=torch.float64)
    v = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
    out = calculate_gaussian_nll(r, v, eps=0.0)
    np.testing.assert_allclose(out.numpy(), _ref(r, v), rtol=1e-10)


# ---------------------------------------------------------------------- UTL-004
# UTL-004: parse_heteroscedastic_output splits any ndim >= 2 tensor along dim 1


def test_UTL_004_3d_output_split_on_last_dim():
    y = torch.arange(4 * 2 * 6, dtype=torch.float64).reshape(4, 2, 6)
    mean, log_var = parse_heteroscedastic_output(y)
    ref_mean, ref_lv = split_mean_log_variance(y)
    assert mean.shape == (4, 2, 3)
    torch.testing.assert_close(mean, ref_mean)
    torch.testing.assert_close(log_var, ref_lv)


# ---------------------------------------------------------------------- UTL-005
# UTL-005: BoxCoxTransform / YeoJohnsonTransform use (x**lam - 1) / lam and


X_POS = np.array([0.05, 0.5, 2.0, 5.0, 50.0])


def test_UTL_005_boxcox_forward_float32_small_lambda():
    lam = 1e-6
    out = BoxCoxTransform(lam=lam, eps=0.0)(torch.tensor(X_POS, dtype=torch.float32))
    np.testing.assert_allclose(out.numpy(), special.boxcox(X_POS, lam), rtol=1e-4)


def test_UTL_005_boxcox_inverse_float32_small_lambda():
    lam = 1e-6
    y = special.boxcox(X_POS, lam)
    out = BoxCoxTransform(lam=lam, eps=0.0).inverse(torch.tensor(y, dtype=torch.float32))
    np.testing.assert_allclose(out.numpy(), X_POS, rtol=1e-4)


def test_UTL_005_yeojohnson_forward_float32_small_lambda():
    lam = 1e-6
    x = np.concatenate([X_POS, -X_POS])
    out = YeoJohnsonTransform(lam=lam)(torch.tensor(x, dtype=torch.float32))
    np.testing.assert_allclose(out.numpy(), stats.yeojohnson(x, lmbda=lam), rtol=1e-4)


# ---------------------------------------------------------------------- UTL-006
# UTL-006: normal_cdf = 0.5 * (1 + erf(z / sqrt 2)) loses all relative precision in


def test_UTL_006_normal_cdf_lower_tail_float64():
    z = np.array([-6.0, -8.0, -10.0, -20.0, -30.0])
    out = normal_cdf(torch.tensor(z, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(out, stats.norm.cdf(z), rtol=1e-10)


def test_UTL_006_normal_cdf_lower_tail_float32():
    z = np.array([-4.0, -5.0, -6.0, -8.0])
    out = normal_cdf(torch.tensor(z, dtype=torch.float32)).numpy()
    np.testing.assert_allclose(out, stats.norm.cdf(z), rtol=1e-4)


# ---------------------------------------------------------------------- UTL-007
# UTL-007: validate_quantile / validate_positive / validate_range accept NaN


@pytest.mark.parametrize(
    "call",
    [
        lambda: validate_quantile(math.nan),
        lambda: validate_quantile(torch.tensor([0.1, math.nan, 0.9])),
        lambda: validate_positive(math.nan, "scale"),
        lambda: validate_positive(torch.tensor([1.0, math.nan]), "scale"),
        lambda: validate_range(math.nan, 0.0, 1.0, "alpha"),
        lambda: validate_range(torch.tensor([0.5, math.nan]), 0.0, 1.0, "alpha"),
    ],
)
def test_UTL_007_validators_reject_nan(call):
    with pytest.raises(ValueError):
        call()


# ---------------------------------------------------------------------- UTL-008
# UTL-008: BSplineDensityBasis.bin_integrals always returns the *cached* float64 CPU


def test_UTL_008_bin_integrals_follow_edges_dtype():
    basis = BSplineDensityBasis.from_uniform(0.0, 3.0, n_intervals=6)
    edges = torch.linspace(0.0, 3.0, 7, dtype=torch.float32)
    coeffs = torch.softmax(torch.randn(4, basis.n_basis), dim=-1)  # float32, as from a head
    T = basis.bin_integrals(edges)
    assert T.dtype == edges.dtype
    masses = coeffs @ T.T  # documented usage (docs/losses/density_basis.md)
    torch.testing.assert_close(masses.sum(-1), torch.ones(4), rtol=1e-5, atol=1e-5)


def test_UTL_008_bin_integrals_cache_not_aliased():
    basis = BSplineDensityBasis.from_uniform(0.0, 3.0, n_intervals=6)
    edges = [0.0, 1.0, 2.0, 3.0]
    T = basis.bin_integrals(edges)
    T.mul_(2.0)  # caller-side in-place op, e.g. scaling to counts
    T2 = basis.bin_integrals(edges)
    torch.testing.assert_close(T2.sum(0), torch.ones(basis.n_basis, dtype=T2.dtype))


# ---------------------------------------------------------------------- UTL-009
# UTL-009: public names missing from docs/api/ and documented names that do not exist.


DOCS = REPO_ROOT / "docs" / "api"
TEXT = "\n".join(p.read_text() for p in DOCS.glob("*.md"))

PUBLIC = {
    "torchregress.utils.tensor_ops": ["float_dtype"],
    "torchregress.utils.openml_relaxed": [
        "fetch_openml_regression_frame_skip_checksum",
        "fetch_openml_regression_with_sklearn_fallback",
    ],
    "torchregress.method_catalog": [
        "list_methods",
        "get_method_metadata",
        "list_task_recommendations",
        "list_decision_workflow_steps",
        "list_comparative_evidence_rows",
        "MethodMetadata",
    ],
    "torchregress.health": ["check_health"],
}


def test_UTL_009_public_names_documented():
    undocumented = []
    for mod, names in PUBLIC.items():
        m = importlib.import_module(mod)
        for n in names:
            assert hasattr(m, n)
            if not re.search(r"`" + re.escape(n) + r"[`(\s]", TEXT):
                undocumented.append(f"{mod}.{n}")
    assert undocumented == [], undocumented


def test_UTL_009_no_phantom_symbols_in_utils_md():
    text = (DOCS / "utils.md").read_text()
    documented = set(re.findall(r"^\|\s*`([A-Za-z_][A-Za-z0-9_]*)", text, re.M))
    mods = [
        importlib.import_module(m)
        for m in [
            "torchregress.utils",
            "torchregress.prediction",
            "torchregress.utils.tensor_ops",
            "torchregress.utils.augment",
            "torchregress.utils.openml_relaxed",
            "torchregress.utils.reduction",
            # utils.md lists the internal helpers (not re-exported) under their submodules.
            "torchregress.utils.distributions",
            "torchregress.utils.gaussian_output",
            "torchregress.utils.numpy_stats",
            "torchregress.utils.ordinal",
            "torchregress.utils.pytorch_compat",
            "torchregress.utils.quantile",
            "torchregress.utils.security",
            "torchregress.utils.validation",
            # utils.md also documents these two modules (method catalog API, health check).
            "torchregress.method_catalog",
            "torchregress.health",
        ]
    ]
    phantoms = sorted(s for s in documented if not any(hasattr(m, s) for m in mods))
    assert phantoms == [], phantoms


# ---------------------------------------------------------------------- UTL-010
# UTL-010: openml_relaxed decodes nominal (byte-string) ARFF columns into str and writes


ARFF = b"""@relation toy
@attribute x1 numeric
@attribute flag {0,1}
@attribute target numeric
@data
1.0,0,2.0
2.0,1,3.0
3.0,1,5.0
"""


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(
        om,
        "_openml_dataset_description",
        lambda data_id, timeout=0: {"format": "ARFF", "url": "https://example.org/x.arff"},
    )
    monkeypatch.setattr(
        om,
        "_openml_feature_list",
        lambda data_id, timeout=0: [
            {"name": "x1"},
            {"name": "flag"},
            {"name": "target", "is_target": "true"},
        ],
    )
    monkeypatch.setattr(om, "_download_bytes", lambda url, timeout=0: ARFF)


def test_UTL_010_numeric_coded_nominal_feature_kept(offline):
    X, y, _ = om.fetch_openml_regression_frame_skip_checksum(data_id=1)
    np.testing.assert_array_equal(X, np.array([[1, 0], [2, 1], [3, 1]], dtype=np.float32))
    np.testing.assert_array_equal(y, np.array([2, 3, 5], dtype=np.float32))


# ---------------------------------------------------------------------- VIZ-001
# VIZ-001: plot_pit_histogram calls ax.hist(pit, bins=n_bins) without range=(0, 1), so


def _bars(fig):
    ax = fig.axes[0]
    rects = [p for p in ax.patches if isinstance(p, matplotlib.patches.Rectangle)]
    lefts = np.array([r.get_x() for r in rects])
    widths = np.array([r.get_width() for r in rects])
    heights = np.array([r.get_height() for r in rects])
    return lefts, widths, heights


def test_VIZ_001_pit_bins_span_unit_interval():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=400)
    mu = np.zeros(400)
    sigma = np.full(400, 2.0)  # underconfident: PIT concentrated in ~[0.2, 0.8]
    n_bins = 10
    fig = plot_pit_histogram(mu, sigma, y_true, n_bins=n_bins, return_figure=True)
    lefts, widths, heights = _bars(fig)
    plt.close(fig)
    np.testing.assert_allclose(lefts[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(lefts[-1] + widths[-1], 1.0, atol=1e-12)
    pit = stats.norm.cdf((y_true - mu) / sigma)
    ref, _ = np.histogram(pit, bins=n_bins, range=(0.0, 1.0), density=True)
    np.testing.assert_allclose(heights, ref, rtol=1e-6)


# ---------------------------------------------------------------------- VIZ-002
# VIZ-002: plot_causal_uplift_qini masks single-arm prefixes as NaN (the first prefix of


def test_VIZ_002_qini_area_is_finite():
    rng = np.random.default_rng(0)
    n = 400
    t = rng.integers(0, 2, n)
    tau = rng.normal(size=n)
    y = rng.normal(size=n) + t * tau
    fig = plot_causal_uplift_qini(tau, t, y, return_figure=True)
    texts = [c.get_text() for c in fig.axes[0].texts]
    plt.close(fig)
    line = next(s for s in texts if "Qini Area Metric" in s)
    value = float(line.split(":")[1])
    assert np.isfinite(value), line


# ---------------------------------------------------------------------- VIZ-003
# VIZ-003: plot_calibration_curve's RMSCE averages over *all* bins, empty bins included


def test_VIZ_003_rmsce_ignores_empty_bins():
    # 100 predictions at p = 0.75, observed frequency 0.25 -> |acc - conf| = 0.5 everywhere.
    probs = np.full(100, 0.75)
    y = np.r_[np.ones(25), np.zeros(75)]
    fig, diag = plot_calibration_curve(
        probs, y, n_bins=10, return_figure=True, return_diagnostics=True
    )
    plt.close(fig)
    np.testing.assert_allclose(diag["mean_calibration_error"], 0.5, atol=1e-12)
    np.testing.assert_allclose(diag["root_mean_squared_calibration_error"], 0.5, atol=1e-12)


# ---------------------------------------------------------------------- VIZ-004
# VIZ-004: plot_residual_histogram and plot_qq_plot compute (y_true - y_pred).flatten()


def _residual_data():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=60)
    y_pred = (y_true + 0.1 * rng.normal(size=60))[:, None]  # [N, 1] model output
    return torch.tensor(y_pred), torch.tensor(y_true)


def test_VIZ_004_residual_histogram_stats():
    y_pred, y_true = _residual_data()
    fig = plot_residual_histogram(y_pred, y_true, show_kde=False, return_figure=True)
    text = "\n".join(c.get_text() for c in fig.axes[0].texts)
    plt.close(fig)
    res = y_true.numpy() - y_pred.numpy().ravel()
    std = float(text.split("Std:")[1].split()[0])
    np.testing.assert_allclose(std, np.std(res), atol=1e-3)


def test_VIZ_004_qq_plot_point_count():
    y_pred, y_true = _residual_data()
    fig = plot_qq_plot(y_pred, y_true, return_figure=True)
    offsets = fig.axes[0].collections[0].get_offsets()
    plt.close(fig)
    assert len(offsets) == 60


# ---------------------------------------------------------------------- VIZ-005
# VIZ-005: plot_performance_comparison picks the 'best' model with the substring registry


@pytest.mark.parametrize(
    "metric", ["interval_score", "nmad", "mad", "mpiw", "mean_width", "train_s"]
)
def test_VIZ_005_lower_is_better_metric_highlights_minimum(metric):
    metrics = {"good": {metric: 0.1}, "bad": {metric: 5.0}}
    fig = plot_performance_comparison(metrics, plot_type="bar", return_figure=True)
    containers = fig.axes[0].containers
    widths = {c.get_label(): c.patches[0].get_linewidth() for c in containers}
    plt.close(fig)
    assert widths["good"] == 2 and widths["bad"] != 2, widths


# ---------------------------------------------------------------------- extra coverage


def test_PRD_003_flat_predictions_vs_column_targets():
    g = torch.Generator().manual_seed(1)
    y_true = torch.randn(40, 1, generator=g, dtype=torch.float64)
    y_pred = (y_true + 0.2 * torch.randn(40, 1, generator=g, dtype=torch.float64)).squeeze(1)
    out = compute_point_metrics(y_pred, y_true)
    yt, yp = y_true.numpy().ravel(), y_pred.numpy()
    np.testing.assert_allclose(out["MSE"], mean_squared_error(yt, yp), rtol=1e-6)
    np.testing.assert_allclose(out["R2"], r2_score(yt, yp), rtol=1e-6)


def test_UTL_001_masked_mean_broadcast_mask_and_dim():
    x = torch.tensor([[1.0, float("nan"), 3.0], [4.0, 5.0, float("nan")]], dtype=torch.float64)
    m = torch.tensor([[True, False, True], [True, True, False]])
    torch.testing.assert_close(
        masked_mean(x, m, dim=1), torch.tensor([2.0, 4.5], dtype=torch.float64)
    )
    torch.testing.assert_close(
        masked_sum(x, m, dim=0, keepdim=True), torch.tensor([[5.0, 5.0, 3.0]], dtype=torch.float64)
    )
    # A mask that broadcasts (one flag per row) counts every broadcast element.
    row_mask = torch.tensor([[True], [False]])
    torch.testing.assert_close(
        masked_mean(torch.ones(2, 3, dtype=torch.float64), row_mask, dim=1),
        torch.tensor([1.0, 0.0], dtype=torch.float64),
    )


def test_UTL_008_bin_integrals_explicit_dtype_and_device_for_sequences():
    basis = BSplineDensityBasis.from_uniform(0.0, 3.0, n_intervals=6)
    assert basis.bin_integrals([0.0, 1.5, 3.0]).dtype == torch.float64
    T32 = basis.bin_integrals([0.0, 1.5, 3.0], dtype=torch.float32, device="cpu")
    assert T32.dtype == torch.float32 and T32.device.type == "cpu"
    torch.testing.assert_close(T32.sum(0), torch.ones(basis.n_basis), rtol=1e-6, atol=1e-6)


def test_UTL_010_non_numeric_nominal_column_is_still_dropped(monkeypatch):
    arff = b"""@relation toy
@attribute x1 numeric
@attribute colour {red,green}
@attribute target numeric
@data
1.0,red,2.0
2.0,green,3.0
"""
    monkeypatch.setattr(
        om,
        "_openml_dataset_description",
        lambda data_id, timeout=0: {"format": "ARFF", "url": "https://example.org/x.arff"},
    )
    monkeypatch.setattr(
        om,
        "_openml_feature_list",
        lambda data_id, timeout=0: [
            {"name": "x1"},
            {"name": "colour"},
            {"name": "target", "is_target": "true"},
        ],
    )
    monkeypatch.setattr(om, "_download_bytes", lambda url, timeout=0: arff)
    with pytest.raises(ValueError, match="empty"):
        om.fetch_openml_regression_frame_skip_checksum(data_id=1)


def test_VIZ_002_qini_area_matches_trapezoid_over_two_arm_prefixes():
    rng = np.random.default_rng(3)
    n = 200
    t = rng.integers(0, 2, n)
    tau = rng.normal(size=n)
    y = rng.normal(size=n) + t * tau
    fig = plot_causal_uplift_qini(tau, t, y, return_figure=True)
    ax = fig.axes[0]
    model, rand = ax.get_lines()[0], ax.get_lines()[1]
    x = np.asarray(model.get_xdata(), dtype=float)
    diff = np.asarray(model.get_ydata(), dtype=float) - np.asarray(rand.get_ydata(), dtype=float)
    ok = np.isfinite(diff)
    expected = np.sum(0.5 * (diff[ok][1:] + diff[ok][:-1]) * np.diff(x[ok]))
    text = next(c.get_text() for c in ax.texts if "Qini Area Metric" in c.get_text())
    np.testing.assert_allclose(float(text.split(":")[1]), expected, rtol=1e-3, atol=1e-3)


def test_VIZ_002_qini_area_is_nan_without_a_two_arm_prefix():
    n = 20
    fig = plot_causal_uplift_qini(
        np.linspace(1, -1, n), np.ones(n, dtype=int), np.arange(n, dtype=float), return_figure=True
    )
    text = next(c.get_text() for c in fig.axes[0].texts if "Qini Area Metric" in c.get_text())
    assert np.isnan(float(text.split(":")[1]))


def _lifecycle_calls():
    rng = np.random.default_rng(0)
    n = 30
    y_true = rng.normal(size=n)
    y_pred = y_true + 0.1 * rng.normal(size=n)
    std = np.full(n, 0.5)
    quantiles = {0.1: y_pred - 0.5, 0.5: y_pred, 0.9: y_pred + 0.5}
    probs = rng.uniform(size=n)
    labels = (rng.uniform(size=n) < probs).astype(float)
    return {
        "reliability": lambda **kw: plot_reliability_diagram(quantiles, y_true, **kw),
        "intervals": lambda **kw: plot_prediction_intervals(
            y_pred, y_pred - 1, y_pred + 1, y_true, **kw
        ),
        "qq": lambda **kw: plot_qq_plot(y_pred, y_true, **kw),
        "residual_hist": lambda **kw: plot_residual_histogram(y_pred, y_true, **kw),
        "calibration": lambda **kw: plot_calibration_curve(probs, labels, **kw),
        "pit": lambda **kw: plot_pit_histogram(y_pred, std, y_true, **kw),
        "gaussian_reliability": lambda **kw: plot_gaussian_reliability_diagram(
            y_pred, std, y_true, **kw
        ),
    }


@pytest.mark.parametrize("name", list(_lifecycle_calls()))
def test_VIZ_figure_lifecycle_closes_only_figures_it_created(name):
    call = _lifecycle_calls()[name]
    # ax=None: the function creates, shows and closes its own figure.
    assert call() is None
    assert plt.get_fignums() == []
    # return_figure=True: the caller owns the figure.
    fig = call(return_figure=True)
    assert plt.get_fignums() == [fig.number]
    plt.close(fig)
    # A caller-provided axes is never closed.
    fig, ax = plt.subplots()
    assert call(ax=ax) is None
    assert plt.get_fignums() == [fig.number]
    plt.close(fig)


def test_VIZ_005_registry_keeps_higher_is_better_metrics():
    from torchregress.viz.utils import is_lower_better

    for name in ("r2", "accuracy", "coverage", "picp", "ev_score"):
        assert not is_lower_better(name)
    for name in ("interval_score", "nmad", "mad", "mpiw", "mean_width", "train_s", "eval_s"):
        assert is_lower_better(name)
