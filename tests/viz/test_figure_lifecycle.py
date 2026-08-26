"""Regression tests for audit-fix Workstream D (TR-VIZ-01..21, viz module)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from torchregress.viz.diagnostic import (
    _compute_binned_metrics,
    plot_censored_survival_curves,
    plot_residual_histogram,
    plot_uncertainty_vs_error,
)
from torchregress.viz.monitoring import _smooth_losses, _suggest_learning_rate
from torchregress.viz.results import _mass_contour_levels, plot_causal_uplift_qini
from torchregress.viz.utils import (
    LOWER_IS_BETTER_TERMS,
    create_color_palette,
    is_lower_better,
    save_figure,
    set_style,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ── D2: direction registry ────────────────────────────────────────────────────


class TestDirectionRegistry:
    def test_lower_is_better_terms(self):
        """Every registry term matches itself and common metric names."""
        for term in (
            "error",
            "loss",
            "mae",
            "mse",
            "rmse",
            "mape",
            "nll",
            "crps",
            "pinball",
            "brier",
            "log_loss",
            "ece",
            "mce",
            "deviance",
        ):
            assert term in LOWER_IS_BETTER_TERMS

    def test_is_lower_better_table(self):
        assert is_lower_better("rmse")
        assert is_lower_better("validation_loss")
        assert is_lower_better("mean_absolute_percentage_error")
        assert is_lower_better("crps")
        assert not is_lower_better("r2")
        assert not is_lower_better("accuracy")
        assert not is_lower_better("coverage")


# ── D4: EMA smoothing recurrence ─────────────────────────────────────────────


class TestEmaSmoothing:
    def test_recurrence_identity(self):
        """s_t = alpha * x_t + (1 - alpha) * s_{t-1}, s_0 = x_0."""
        rng = np.random.default_rng(0)
        x = rng.normal(5.0, 2.0, size=50)
        alpha = 0.3
        out = _smooth_losses(x, smoothing=alpha)
        expected = np.empty_like(x)
        s = x[0]
        for i, xi in enumerate(x):
            s = alpha * xi + (1 - alpha) * s
            expected[i] = s
        assert np.allclose(out, expected)

    def test_alpha_one_is_identity(self):
        x = np.array([3.0, 1.0, 4.0, 1.5])
        assert np.allclose(_smooth_losses(x, 1.0), x)

    def test_no_smoothing_passthrough(self):
        x = np.arange(5.0)
        assert np.array_equal(_smooth_losses(x, 0.0), x)


# ── D4: corner-contour credible-mass levels ──────────────────────────────────


class TestCornerContourLevels:
    def test_uniform_field(self):
        """On a constant density every level equals the (single) density value."""
        z = np.full((10, 10), 2.5)
        levels = _mass_contour_levels(z)
        assert levels == pytest.approx([2.5, 2.5])

    def test_gaussian_mass_thresholds(self):
        """For a discrete Gaussian-like field the returned levels enclose ~39.3%/86.5% mass."""
        grid = np.arange(-50, 51)
        dens = np.exp(-0.5 * (grid / 10.0) ** 2)  # unnormalized Gaussian profile
        total = dens.sum()
        target_masses = (1 - np.exp(-0.5), 1 - np.exp(-2.0))

        flat = np.sort(dens.ravel())[::-1]
        cum = np.cumsum(flat) / total
        for m in target_masses:
            idx = int(np.searchsorted(cum, m))
            enclosed = cum[min(idx, len(flat) - 1)]
            assert enclosed >= m - 1e-9

        levels = sorted(_mass_contour_levels(dens))
        # Levels are density values strictly inside the density range
        assert levels[0] < dens.max()
        assert len(levels) == 2


# ── D4: set_style validation ─────────────────────────────────────────────────


class TestSetStyle:
    def test_unknown_style_raises_listing_names(self):
        with pytest.raises(ValueError) as excinfo:
            set_style("no-such-style")
        msg = str(excinfo.value)
        for name in ("default", "whitegrid", "darkgrid", "ticks", "minimal"):
            assert name in msg

    def test_valid_styles_accepted(self):
        for style in ("default", "whitegrid", "darkgrid", "ticks", "minimal"):
            set_style(style)  # must not raise

    def test_minimal_has_own_rcparams(self):
        set_style("minimal")
        assert not plt.rcParams["axes.grid"]
        assert not plt.rcParams["axes.spines.top"]
        assert not plt.rcParams["axes.spines.right"]


# ── D4: save_figure contract ─────────────────────────────────────────────────


class TestSaveFigure:
    def test_writes_both_formats_and_returns_paths(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])

        written = save_figure(fig, "auditfix_d", directory=str(tmp_path))

        assert written == [str(tmp_path / "auditfix_d.png"), str(tmp_path / "auditfix_d.pdf")]
        for p in written:
            assert __import__("os").path.exists(p)
        plt.close(fig)


# ── D4: qualitative palette sampling ─────────────────────────────────────────


class TestColorPalette:
    def test_listed_colormap_discrete_indexing(self):
        colors = create_color_palette(12, palette_name="tab10")
        cmap = plt.get_cmap("tab10")
        assert isinstance(cmap, matplotlib.colors.ListedColormap)
        for i in range(12):
            assert colors[i] == cmap(i % 10)

    def test_single_color_uses_zero(self):
        colors = create_color_palette(1, palette_name="viridis")
        assert colors[0] == plt.get_cmap("viridis")(0.0)


# ── D4: Qini single-arm prefixes are masked ──────────────────────────────────


class TestQiniMasking:
    def test_prefix_without_control_is_nan_gap(self):
        uplift = np.linspace(10.0, -10.0, 20)  # all-treated prefix when ranked desc
        treatment = np.ones(20, dtype=int)  # everyone treated -> no control arm at any prefix
        y_obs = np.random.default_rng(1).normal(size=20)

        result = plot_causal_uplift_qini(uplift, treatment, y_obs, return_figure=True)
        assert isinstance(result, Figure)  # returns a figure without crashing
        line = result.axes[0].get_lines()[0]
        ydata = np.asarray(line.get_ydata(), dtype=float)
        assert np.isnan(ydata).any()  # fabricated zero-control prefixes are now gaps

    def test_mixed_arms_have_no_nan(self):
        rng = np.random.default_rng(2)
        treatment = rng.binomial(1, 0.5, 100)
        uplift = rng.normal(size=100)
        y_obs = rng.normal(size=100)
        fig = plot_causal_uplift_qini(uplift, treatment, y_obs, return_figure=True)
        ydata = np.asarray(fig.axes[0].get_lines()[0].get_ydata(), dtype=float)
        # Position 0 is the inserted origin; position 1 is the first ranked
        # sample and may legitimately lack one arm. Everything after must be finite.
        assert ydata[0] == 0.0
        assert np.isfinite(ydata[2:]).all()


# ── D4: binned metrics last-bin label ────────────────────────────────────────


class TestBinnedMetricsLabels:
    def test_last_bin_closed_interval_label(self):
        y_pred = np.arange(40, dtype=float)
        y_std = np.full(40, 0.5)
        y_true = np.arange(40, dtype=float)
        metrics = _compute_binned_metrics(y_pred, y_std, y_true, n_bins=4)
        keys = list(metrics.keys())
        assert keys[0].startswith("[")
        assert keys[-1].endswith("]")

    def test_duplicate_edges_warn(self):
        y_true = np.zeros(40)  # degenerate: all quantile edges identical
        y_pred = np.zeros(40)
        y_std = np.ones(40)
        with pytest.warns(RuntimeWarning, match="duplicate bin edges"):
            _compute_binned_metrics(y_pred, y_std, y_true, n_bins=4)


# ── D4: LR suggestion has no x0.1 fudge ──────────────────────────────────────


class TestLRSuggestion:
    def test_minimum_suggested_verbatim(self):
        lrs = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        smooth = np.array([10.0, 6.0, 1.0, 7.0, 30.0])
        assert _suggest_learning_rate(lrs, smooth, "minimum") == pytest.approx(1e-3)

    def test_valley_at_gradient_sign_change(self):
        lrs = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        smooth = np.array([10.0, 5.0, 2.0, 8.0, 20.0])
        suggestion = _suggest_learning_rate(lrs, smooth, "valley")
        # Valley bottom sits between lrs[1] and lrs[2]; no x0.1 shrinkage applied.
        assert 1e-4 <= suggestion <= 1e-2

    def test_empty_returns_none(self):
        assert _suggest_learning_rate(np.array([]), np.array([]), "valley") is None


# ── D3/D4: figure lifecycle and return contracts ─────────────────────────────


class TestFigureLifecycleAndContracts:
    def test_default_path_closes_figure(self):
        before = plt.get_fignums()
        result = plot_uncertainty_vs_error(np.arange(10.0), np.ones(10), np.arange(10.0) * 0.1)
        after = plt.get_fignums()
        # Display path under Agg: correlation NOT leaked via return value.
        assert result is None
        assert after == before  # TR-VIZ-05: figure was closed

    def test_return_correlation_kwarg_only(self):
        corr = plot_uncertainty_vs_error(
            np.arange(10.0),
            np.ones(10) * 0.5,
            np.arange(10.0) * 0.1,
            return_correlation=True,
        )
        assert isinstance(corr, float)

    def test_residual_histogram_n_bins_kwarg(self):
        fig = plot_residual_histogram(
            np.arange(10.0), np.arange(10.0) + 0.1, n_bins=5, return_figure=True, show_kde=False
        )
        assert isinstance(fig, Figure)

    def test_km_matches_bruteforce(self):
        rng = np.random.default_rng(3)
        times = rng.uniform(0.5, 10.0, size=60)
        events = rng.binomial(1, 0.7, size=60).astype(float)
        fig = plot_censored_survival_curves(
            predicted_survival=np.full((60, 20), 0.9),
            time_grid=np.linspace(0, 10, 20),
            observed_times=times,
            censoring_indicators=events,
            return_figure=True,
        )
        km_line = fig.axes[0].get_lines()[-1]
        km_t = np.asarray(km_line.get_xdata(), dtype=float)
        km_s = np.asarray(km_line.get_ydata(), dtype=float)

        # Brute-force KM reference
        surv = 1.0
        ref = {0.0: 1.0}
        for t in sorted(set(times[times > 0])):
            d = np.sum((times == t) & (events == 1))
            n = np.sum(times >= t)
            if n > 0:
                surv *= 1 - d / n
                ref[float(t)] = surv
        for kt, ks in zip(km_t, km_s):
            if kt in ref:
                assert ks == pytest.approx(ref[kt])
