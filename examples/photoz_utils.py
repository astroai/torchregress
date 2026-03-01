"""
Visualization and analysis utilities for photometric redshift estimation.

This module provides helper functions for:
- Calibration diagnostics (reliability diagrams, PIT histograms)
- Uncertainty quality assessment
- Tail/edge performance analysis
"""

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_reliability_diagram(
    predictions: np.ndarray,
    pred_stds: np.ndarray,
    targets: np.ndarray,
    ax: Optional[plt.Axes] = None,
    n_bins: int = 10,
    label: Optional[str] = None,
    color: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plot a reliability diagram (calibration curve) for probabilistic predictions.

    A well-calibrated model should have observed coverage matching expected coverage
    at all confidence levels. Points on the diagonal indicate perfect calibration.

    Args:
        predictions: Predicted mean values, shape (n_samples,) or (n_samples, 1)
        pred_stds: Predicted standard deviations, shape (n_samples,) or (n_samples, 1)
        targets: True values, shape (n_samples,) or (n_samples, 1)
        ax: Matplotlib axes to plot on (creates new figure if None)
        n_bins: Number of confidence levels to evaluate
        label: Label for the plot legend
        color: Color for the plot line

    Returns:
        Tuple of (expected_coverage, observed_coverage) arrays

    Physics interpretation:
        Think of this as testing whether your error bars are honest.
        If you claim 68% confidence intervals, do 68% of true values fall inside?
    """
    predictions = np.asarray(predictions).flatten()
    pred_stds = np.asarray(pred_stds).flatten()
    targets = np.asarray(targets).flatten()

    # Confidence levels to check
    confidence_levels = np.linspace(0.1, 0.99, n_bins)

    expected_coverage = []
    observed_coverage = []

    for conf in confidence_levels:
        # For Gaussian: z-score for given confidence level
        z = stats.norm.ppf((1 + conf) / 2)

        # Compute interval bounds
        lower = predictions - z * pred_stds
        upper = predictions + z * pred_stds

        # Check coverage
        covered = (targets >= lower) & (targets <= upper)
        observed = np.mean(covered)

        expected_coverage.append(conf)
        observed_coverage.append(observed)

    expected_coverage = np.array(expected_coverage)
    observed_coverage = np.array(observed_coverage)

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.5)
    ax.plot(expected_coverage, observed_coverage, "o-", label=label, color=color, markersize=6)
    ax.set_xlabel("Expected Coverage")
    ax.set_ylabel("Observed Coverage")
    ax.set_title("Reliability Diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()

    return expected_coverage, observed_coverage


def plot_pit_histogram(
    predictions: np.ndarray,
    pred_stds: np.ndarray,
    targets: np.ndarray,
    ax: Optional[plt.Axes] = None,
    n_bins: int = 20,
    label: Optional[str] = None,
    color: Optional[str] = None,
) -> np.ndarray:
    """
    Plot Probability Integral Transform (PIT) histogram.

    For a well-calibrated probabilistic model, the PIT values should be
    uniformly distributed. Deviations indicate miscalibration:
    - U-shaped: underconfident (variances too large)
    - Inverse U-shaped: overconfident (variances too small)
    - Skewed: biased predictions

    Args:
        predictions: Predicted mean values
        pred_stds: Predicted standard deviations
        targets: True values
        ax: Matplotlib axes to plot on
        n_bins: Number of histogram bins
        label: Label for the plot
        color: Color for the histogram

    Returns:
        PIT values array

    Physics interpretation:
        If your probability distributions are correct, transforming each
        observation through its predictive CDF should give Uniform(0,1).
        Like checking if your PDF integrates to the right values.
    """
    predictions = np.asarray(predictions).flatten()
    pred_stds = np.asarray(pred_stds).flatten()
    targets = np.asarray(targets).flatten()

    # Compute PIT: CDF of target under predicted Gaussian
    z_scores = (targets - predictions) / (pred_stds + 1e-8)
    pit_values = stats.norm.cdf(z_scores)

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(
        pit_values,
        bins=n_bins,
        density=True,
        alpha=0.7,
        label=label,
        color=color,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axhline(y=1.0, color="red", linestyle="--", label="Uniform (ideal)")
    ax.set_xlabel("PIT Value")
    ax.set_ylabel("Density")
    ax.set_title("PIT Histogram (Uniform = Well-Calibrated)")
    ax.set_xlim(0, 1)
    ax.legend()

    return pit_values


def plot_uncertainty_vs_error(
    predictions: np.ndarray,
    pred_stds: np.ndarray,
    targets: np.ndarray,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
    max_points: int = 1000,
) -> float:
    """
    Plot predicted uncertainty vs absolute error.

    A good uncertainty estimator should show positive correlation:
    when the model predicts high uncertainty, errors should actually be larger.

    Args:
        predictions: Predicted mean values
        pred_stds: Predicted standard deviations
        targets: True values
        ax: Matplotlib axes to plot on
        label: Label for the plot
        color: Color for the scatter points
        max_points: Maximum number of points to plot (for performance)

    Returns:
        Spearman correlation coefficient between pred_std and |error|

    Physics interpretation:
        This tests whether your uncertainty estimates are informative.
        Like checking if larger stated uncertainties actually correspond
        to larger measurement errors.
    """
    predictions = np.asarray(predictions).flatten()
    pred_stds = np.asarray(pred_stds).flatten()
    targets = np.asarray(targets).flatten()

    # Compute absolute errors
    abs_errors = np.abs(predictions - targets)

    # Compute Spearman correlation
    correlation, p_value = stats.spearmanr(pred_stds, abs_errors)

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Subsample if too many points
    if len(predictions) > max_points:
        idx = np.random.choice(len(predictions), max_points, replace=False)
        pred_stds_plot = pred_stds[idx]
        abs_errors_plot = abs_errors[idx]
    else:
        pred_stds_plot = pred_stds
        abs_errors_plot = abs_errors

    ax.scatter(pred_stds_plot, abs_errors_plot, alpha=0.3, s=10, label=label, color=color)

    # Add trend line
    z = np.polyfit(pred_stds_plot, abs_errors_plot, 1)
    p = np.poly1d(z)
    x_line = np.linspace(pred_stds_plot.min(), pred_stds_plot.max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f"Trend (ρ={correlation:.3f})")

    ax.set_xlabel("Predicted Uncertainty (σ)")
    ax.set_ylabel("Absolute Error |ŷ - y|")
    ax.set_title("Uncertainty vs Error (should correlate)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return correlation


def compute_binned_metrics(
    predictions: np.ndarray,
    pred_stds: np.ndarray,
    targets: np.ndarray,
    bin_edges: Optional[np.ndarray] = None,
    n_bins: int = 5,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics in bins of the target variable.

    This reveals whether model performance degrades in certain regions,
    particularly at the tails of the distribution (e.g., high-z galaxies).

    Args:
        predictions: Predicted mean values
        pred_stds: Predicted standard deviations
        targets: True values
        bin_edges: Custom bin edges (if None, uses quantiles)
        n_bins: Number of bins if bin_edges not provided

    Returns:
        Dictionary with metrics for each bin

    Physics interpretation:
        Checks if your model works equally well across the parameter space.
        Important for detecting biases at extremes (e.g., faint sources,
        high redshifts, edge cases).
    """
    predictions = np.asarray(predictions).flatten()
    pred_stds = np.asarray(pred_stds).flatten()
    targets = np.asarray(targets).flatten()

    # Create bins if not provided
    if bin_edges is None:
        bin_edges = np.quantile(targets, np.linspace(0, 1, n_bins + 1))

    results = {}

    for i in range(len(bin_edges) - 1):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (targets >= low) & (targets < high)

        if i == len(bin_edges) - 2:  # Include right edge for last bin
            mask = (targets >= low) & (targets <= high)

        if mask.sum() < 10:  # Skip bins with too few samples
            continue

        bin_preds = predictions[mask]
        bin_stds = pred_stds[mask]
        bin_targets = targets[mask]

        # Compute metrics
        errors = bin_preds - bin_targets
        abs_errors = np.abs(errors)

        # RMSE
        rmse = np.sqrt(np.mean(errors**2))

        # MAE
        mae = np.mean(abs_errors)

        # Bias
        bias = np.mean(errors)

        # NMAD (normalized median absolute deviation)
        delta_z_norm = errors / (1 + bin_targets)
        nmad = 1.48 * np.median(np.abs(delta_z_norm - np.median(delta_z_norm)))

        # PICP at 95%
        z = stats.norm.ppf(0.975)
        lower = bin_preds - z * bin_stds
        upper = bin_preds + z * bin_stds
        picp = np.mean((bin_targets >= lower) & (bin_targets <= upper))

        # Mean prediction interval width
        mpiw = np.mean(upper - lower)

        bin_name = f"[{low:.3f}, {high:.3f})"
        results[bin_name] = {
            "n_samples": int(mask.sum()),
            "rmse": float(rmse),
            "mae": float(mae),
            "bias": float(bias),
            "nmad": float(nmad),
            "picp_95": float(picp),
            "mpiw_95": float(mpiw),
        }

    return results


def plot_binned_metrics(
    binned_metrics: Dict[str, Dict[str, float]],
    metric_name: str = "rmse",
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
) -> None:
    """
    Plot a metric across bins.

    Args:
        binned_metrics: Output from compute_binned_metrics
        metric_name: Which metric to plot ('rmse', 'mae', 'nmad', 'picp_95', etc.)
        ax: Matplotlib axes to plot on
        label: Label for the plot
        color: Color for the bars
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    bins = list(binned_metrics.keys())
    values = [binned_metrics[b][metric_name] for b in bins]
    n_samples = [binned_metrics[b]["n_samples"] for b in bins]

    x = np.arange(len(bins))
    bars = ax.bar(x, values, alpha=0.7, label=label, color=color)

    # Add sample counts as text
    for i, (bar, n) in enumerate(zip(bars, n_samples)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(bins, rotation=45, ha="right")
    ax.set_xlabel("Target Bin (z_spec)")
    ax.set_ylabel(metric_name.upper())
    ax.set_title(f"{metric_name.upper()} by Redshift Bin")
    ax.grid(True, alpha=0.3, axis="y")

    if label:
        ax.legend()


def compare_calibration(
    results: Dict[str, Dict],
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """
    Create comprehensive calibration comparison plot for multiple models.

    Args:
        results: Dictionary of model results, each containing:
            - 'predictions': array
            - 'pred_stds': array
            - 'targets': array
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # 1. Reliability diagram (top-left)
    ax = axes[0, 0]
    for (name, res), color in zip(results.items(), colors):
        if "pred_stds" not in res or res["pred_stds"] is None:
            continue
        plot_reliability_diagram(
            res["predictions"], res["pred_stds"], res["targets"], ax=ax, label=name, color=color
        )
    ax.legend(fontsize=8)

    # 2. PIT histograms (top-right) - show just a few key models
    ax = axes[0, 1]
    key_models = ["GaussianNLL", "Quantile", "MDN", "Evidential"]
    for name in key_models:
        if name in results and "pred_stds" in results[name]:
            res = results[name]
            color = colors[list(results.keys()).index(name)]
            ax.hist(
                stats.norm.cdf(
                    (res["targets"].flatten() - res["predictions"].flatten())
                    / (res["pred_stds"].flatten() + 1e-8)
                ),
                bins=20,
                density=True,
                alpha=0.5,
                label=name,
                color=color,
            )
    ax.axhline(y=1.0, color="red", linestyle="--", label="Uniform")
    ax.set_xlabel("PIT Value")
    ax.set_ylabel("Density")
    ax.set_title("PIT Histograms")
    ax.legend(fontsize=8)

    # 3. Uncertainty vs Error correlation (bottom-left)
    ax = axes[1, 0]
    correlations = {}
    for (name, res), color in zip(results.items(), colors):
        if "pred_stds" not in res or res["pred_stds"] is None:
            continue
        preds = res["predictions"].flatten()
        stds = res["pred_stds"].flatten()
        targets = res["targets"].flatten()
        abs_err = np.abs(preds - targets)
        corr, _ = stats.spearmanr(stds, abs_err)
        correlations[name] = corr

    # Bar chart of correlations
    names = list(correlations.keys())
    corrs = list(correlations.values())
    ax.bar(range(len(names)), corrs, color=[colors[list(results.keys()).index(n)] for n in names])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Spearman Correlation")
    ax.set_title("Uncertainty-Error Correlation (higher = better)")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # 4. Summary metrics table (bottom-right)
    ax = axes[1, 1]
    ax.axis("off")

    # Create summary table
    headers = ["Model", "CRPS", "PICP@95%", "MPIW", "MCE"]
    table_data = []
    for name, res in results.items():
        if "crps" in res:
            row = [
                name,
                f"{res.get('crps', 0):.4f}",
                f"{res.get('picp_95', 0):.3f}",
                f"{res.get('mpiw_95', 0):.4f}",
                f"{res.get('mce', 0):.4f}" if not np.isnan(res.get("mce", float("nan"))) else "N/A",
            ]
            table_data.append(row)

    if table_data:
        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            cellLoc="center",
            loc="center",
            colWidths=[0.25, 0.15, 0.15, 0.15, 0.15],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax.set_title("Probabilistic Metrics Summary", pad=20)

    plt.tight_layout()
    return fig


def print_metrics_table(
    results: Dict[str, Dict],
    include_point: bool = True,
    include_prob: bool = True,
) -> None:
    """
    Print formatted metrics tables to console.

    Args:
        results: Dictionary of model results
        include_point: Include point prediction metrics
        include_prob: Include probabilistic metrics
    """
    if include_point:
        print("\n" + "=" * 70)
        print("POINT PREDICTION METRICS")
        print("=" * 70)
        print(f"{'Model':<20} {'MAE':<10} {'RMSE':<10} {'Bias':<10} {'NMAD':<10}")
        print("-" * 70)
        for name, res in results.items():
            if "mae" in res:
                print(
                    f"{name:<20} {res['mae']:<10.4f} {res['rmse']:<10.4f} "
                    f"{res['bias']:<10.4f} {res['nmad']:<10.4f}"
                )

    if include_prob:
        print("\n" + "=" * 70)
        print("PROBABILISTIC METRICS")
        print("=" * 70)
        print(f"{'Model':<20} {'CRPS':<10} {'PICP@95%':<10} {'MPIW':<10} {'MCE':<10}")
        print("-" * 70)
        for name, res in results.items():
            if "crps" in res:
                mce_str = (
                    f"{res['mce']:.4f}" if not np.isnan(res.get("mce", float("nan"))) else "N/A"
                )
                print(
                    f"{name:<20} {res['crps']:<10.4f} {res['picp_95']:<10.3f} "
                    f"{res['mpiw_95']:<10.4f} {mce_str:<10}"
                )


def print_comprehensive_metrics_table(
    results: Dict[str, Dict],
    n_tail_bins: int = 5,
) -> None:
    """
    Print a comprehensive metrics table covering all evaluation categories.

    Categories:
    - Point: MAE, RMSE, Bias, NMAD
    - Probabilistic: CRPS, PICP@95%, MPIW, MCE
    - Calibration: Uncertainty-Error ρ, MACE
    - Tail/Imbalance: Low-z RMSE, High-z RMSE, Tail PICP
    - Uncertainty Decomposition: Aleatoric, Epistemic (where available)

    Args:
        results: Dictionary of model results with metrics
        n_tail_bins: Number of bins for tail analysis
    """
    print("\n" + "=" * 140)
    print("COMPREHENSIVE METRICS COMPARISON")
    print("=" * 140)

    # Collect all metrics for each model
    all_metrics = {}

    for name, res in results.items():
        metrics = {}

        # Basic metrics (already in results)
        metrics["mae"] = res.get("mae", float("nan"))
        metrics["rmse"] = res.get("rmse", float("nan"))
        metrics["bias"] = res.get("bias", float("nan"))
        metrics["nmad"] = res.get("nmad", float("nan"))
        metrics["crps"] = res.get("crps", float("nan"))
        metrics["picp_95"] = res.get("picp_95", float("nan"))
        metrics["mpiw"] = res.get("mpiw_95", float("nan"))
        metrics["mce"] = res.get("mce", float("nan"))

        # Compute uncertainty-error correlation
        if "predictions" in res and "pred_stds" in res and "targets" in res:
            preds = np.asarray(res["predictions"]).flatten()
            stds = np.asarray(res["pred_stds"]).flatten()
            targets = np.asarray(res["targets"]).flatten()
            abs_err = np.abs(preds - targets)

            if len(stds) > 0 and np.std(stds) > 1e-8:
                corr, _ = stats.spearmanr(stds, abs_err)
                metrics["unc_err_rho"] = corr
            else:
                metrics["unc_err_rho"] = float("nan")

            # Compute tail metrics (first and last quintile)
            try:
                bin_edges = np.quantile(targets, [0, 0.2, 0.8, 1.0])

                # Low-z bin (first 20%)
                low_mask = targets <= bin_edges[1]
                if low_mask.sum() > 10:
                    low_rmse = np.sqrt(np.mean((preds[low_mask] - targets[low_mask]) ** 2))
                    metrics["low_z_rmse"] = low_rmse
                else:
                    metrics["low_z_rmse"] = float("nan")

                # High-z bin (last 20%)
                high_mask = targets >= bin_edges[2]
                if high_mask.sum() > 10:
                    high_rmse = np.sqrt(np.mean((preds[high_mask] - targets[high_mask]) ** 2))
                    # Tail PICP
                    z = stats.norm.ppf(0.975)
                    lower = preds[high_mask] - z * stds[high_mask]
                    upper = preds[high_mask] + z * stds[high_mask]
                    tail_picp = np.mean(
                        (targets[high_mask] >= lower) & (targets[high_mask] <= upper)
                    )
                    metrics["high_z_rmse"] = high_rmse
                    metrics["tail_picp"] = tail_picp
                else:
                    metrics["high_z_rmse"] = float("nan")
                    metrics["tail_picp"] = float("nan")

                # Imbalance ratio (high-z / low-z RMSE)
                if not np.isnan(metrics["low_z_rmse"]) and not np.isnan(metrics["high_z_rmse"]):
                    metrics["imbal_ratio"] = metrics["high_z_rmse"] / (metrics["low_z_rmse"] + 1e-8)
                else:
                    metrics["imbal_ratio"] = float("nan")
            except Exception:
                metrics["low_z_rmse"] = float("nan")
                metrics["high_z_rmse"] = float("nan")
                metrics["tail_picp"] = float("nan")
                metrics["imbal_ratio"] = float("nan")
        else:
            metrics["unc_err_rho"] = float("nan")
            metrics["low_z_rmse"] = float("nan")
            metrics["high_z_rmse"] = float("nan")
            metrics["tail_picp"] = float("nan")
            metrics["imbal_ratio"] = float("nan")

        # Uncertainty decomposition (if available)
        if "aleatoric_std" in res:
            metrics["ale_std"] = float(np.mean(np.asarray(res["aleatoric_std"])))
        else:
            metrics["ale_std"] = float("nan")

        if "epistemic_std" in res:
            metrics["epi_std"] = float(np.mean(np.asarray(res["epistemic_std"])))
        else:
            metrics["epi_std"] = float("nan")

        all_metrics[name] = metrics

    # Print tables by category

    # 1. Point Prediction Metrics
    print("\n┌" + "─" * 80 + "┐")
    print("│" + " POINT PREDICTION METRICS".center(80) + "│")
    print("├" + "─" * 20 + "┬" + "─" * 14 + "┬" + "─" * 14 + "┬" + "─" * 14 + "┬" + "─" * 14 + "┤")
    print(f"│ {'Model':<18} │ {'MAE':^12} │ {'RMSE':^12} │ {'Bias':^12} │ {'NMAD':^12} │")
    print("├" + "─" * 20 + "┼" + "─" * 14 + "┼" + "─" * 14 + "┼" + "─" * 14 + "┼" + "─" * 14 + "┤")
    for name, m in all_metrics.items():
        print(
            f"│ {name:<18} │ {m['mae']:^12.4f} │ {m['rmse']:^12.4f} │ "
            f"{m['bias']:^12.4f} │ {m['nmad']:^12.4f} │"
        )
    print("└" + "─" * 20 + "┴" + "─" * 14 + "┴" + "─" * 14 + "┴" + "─" * 14 + "┴" + "─" * 14 + "┘")

    # 2. Probabilistic Metrics
    print("\n┌" + "─" * 80 + "┐")
    print("│" + " PROBABILISTIC METRICS".center(80) + "│")
    print("├" + "─" * 20 + "┬" + "─" * 14 + "┬" + "─" * 14 + "┬" + "─" * 14 + "┬" + "─" * 14 + "┤")
    print(f"│ {'Model':<18} │ {'CRPS':^12} │ {'PICP@95%':^12} │ {'MPIW':^12} │ {'MCE':^12} │")
    print("├" + "─" * 20 + "┼" + "─" * 14 + "┼" + "─" * 14 + "┼" + "─" * 14 + "┼" + "─" * 14 + "┤")
    for name, m in all_metrics.items():
        mce_str = f"{m['mce']:.4f}" if not np.isnan(m["mce"]) else "N/A"
        print(
            f"│ {name:<18} │ {m['crps']:^12.4f} │ {m['picp_95']:^12.3f} │ "
            f"{m['mpiw']:^12.4f} │ {mce_str:^12} │"
        )
    print("└" + "─" * 20 + "┴" + "─" * 14 + "┴" + "─" * 14 + "┴" + "─" * 14 + "┴" + "─" * 14 + "┘")

    # 3. Calibration & Uncertainty Quality
    print("\n┌" + "─" * 52 + "┐")
    print("│" + " CALIBRATION & UNCERTAINTY QUALITY".center(52) + "│")
    print("├" + "─" * 20 + "┬" + "─" * 14 + "┬" + "─" * 14 + "┤")
    print(f"│ {'Model':<18} │ {'Unc-Err ρ':^12} │ {'|PICP-0.95|':^12} │")
    print("├" + "─" * 20 + "┼" + "─" * 14 + "┼" + "─" * 14 + "┤")
    for name, m in all_metrics.items():
        rho_str = f"{m['unc_err_rho']:.3f}" if not np.isnan(m["unc_err_rho"]) else "N/A"
        picp_dev = abs(m["picp_95"] - 0.95) if not np.isnan(m["picp_95"]) else float("nan")
        picp_dev_str = f"{picp_dev:.3f}" if not np.isnan(picp_dev) else "N/A"
        print(f"│ {name:<18} │ {rho_str:^12} │ {picp_dev_str:^12} │")
    print("└" + "─" * 20 + "┴" + "─" * 14 + "┴" + "─" * 14 + "┘")

    # 4. Tail/Imbalance Performance
    print("\n┌" + "─" * 82 + "┐")
    print("│" + " TAIL/IMBALANCE PERFORMANCE (first vs last 20%)".center(82) + "│")
    print("├" + "─" * 20 + "┬" + "─" * 14 + "┬" + "─" * 14 + "┬" + "─" * 14 + "┬" + "─" * 16 + "┤")
    print(
        f"│ {'Model':<18} │ {'Low-z RMSE':^12} │ {'High-z RMSE':^12} │ {'Tail PICP':^12} │ {'Imbal. Ratio':^14} │"
    )
    print("├" + "─" * 20 + "┼" + "─" * 14 + "┼" + "─" * 14 + "┼" + "─" * 14 + "┼" + "─" * 16 + "┤")
    for name, m in all_metrics.items():
        low_str = f"{m['low_z_rmse']:.4f}" if not np.isnan(m["low_z_rmse"]) else "N/A"
        high_str = f"{m['high_z_rmse']:.4f}" if not np.isnan(m["high_z_rmse"]) else "N/A"
        tail_str = f"{m['tail_picp']:.3f}" if not np.isnan(m["tail_picp"]) else "N/A"
        imbal_str = f"{m['imbal_ratio']:.2f}x" if not np.isnan(m["imbal_ratio"]) else "N/A"
        print(f"│ {name:<18} │ {low_str:^12} │ {high_str:^12} │ {tail_str:^12} │ {imbal_str:^14} │")
    print("└" + "─" * 20 + "┴" + "─" * 14 + "┴" + "─" * 14 + "┴" + "─" * 14 + "┴" + "─" * 16 + "┘")

    # 5. Uncertainty Decomposition (for models that support it)
    decomp_models = {
        k: v
        for k, v in all_metrics.items()
        if not np.isnan(v["ale_std"]) or not np.isnan(v["epi_std"])
    }
    if decomp_models:
        print("\n┌" + "─" * 68 + "┐")
        print("│" + " UNCERTAINTY DECOMPOSITION".center(68) + "│")
        print("├" + "─" * 20 + "┬" + "─" * 14 + "┬" + "─" * 14 + "┬" + "─" * 16 + "┤")
        print(
            f"│ {'Model':<18} │ {'Aleatoric σ':^12} │ {'Epistemic σ':^12} │ {'Epi/(Ale+Epi)':^14} │"
        )
        print("├" + "─" * 20 + "┼" + "─" * 14 + "┼" + "─" * 14 + "┼" + "─" * 16 + "┤")
        for name, m in decomp_models.items():
            ale_str = f"{m['ale_std']:.4f}" if not np.isnan(m["ale_std"]) else "N/A"
            epi_str = f"{m['epi_std']:.4f}" if not np.isnan(m["epi_std"]) else "N/A"
            if not np.isnan(m["ale_std"]) and not np.isnan(m["epi_std"]):
                epi_frac = m["epi_std"] ** 2 / (m["ale_std"] ** 2 + m["epi_std"] ** 2 + 1e-8)
                frac_str = f"{epi_frac:.1%}"
            else:
                frac_str = "N/A"
            print(f"│ {name:<18} │ {ale_str:^12} │ {epi_str:^12} │ {frac_str:^14} │")
        print("└" + "─" * 20 + "┴" + "─" * 14 + "┴" + "─" * 14 + "┴" + "─" * 16 + "┘")

    # Print summary recommendations
    print("\n" + "─" * 80)
    print("SUMMARY RECOMMENDATIONS:")
    print("─" * 80)

    # Find best models by category
    valid_models = {k: v for k, v in all_metrics.items() if not np.isnan(v["rmse"])}
    if valid_models:
        best_point = min(valid_models, key=lambda x: valid_models[x]["rmse"])
        print(f"  Best point prediction (RMSE):      {best_point}")

    valid_prob = {k: v for k, v in all_metrics.items() if not np.isnan(v["crps"])}
    if valid_prob:
        best_prob = min(valid_prob, key=lambda x: valid_prob[x]["crps"])
        print(f"  Best probabilistic (CRPS):         {best_prob}")

    valid_calib = {
        k: v
        for k, v in all_metrics.items()
        if not np.isnan(v["picp_95"]) and abs(v["picp_95"] - 0.95) < 0.2
    }
    if valid_calib:
        best_calib = min(valid_calib, key=lambda x: abs(valid_calib[x]["picp_95"] - 0.95))
        print(f"  Best calibration (PICP@95%):       {best_calib}")

    valid_tail = {k: v for k, v in all_metrics.items() if not np.isnan(v["imbal_ratio"])}
    if valid_tail:
        best_tail = min(valid_tail, key=lambda x: valid_tail[x]["imbal_ratio"])
        print(f"  Best tail balance (Imbal. Ratio):  {best_tail}")

    valid_unc = {
        k: v
        for k, v in all_metrics.items()
        if not np.isnan(v["unc_err_rho"]) and v["unc_err_rho"] > 0
    }
    if valid_unc:
        best_unc = max(valid_unc, key=lambda x: valid_unc[x]["unc_err_rho"])
        print(f"  Best uncertainty quality (ρ):      {best_unc}")


def print_expected_metrics_guide() -> None:
    """
    Print a guide explaining what metric values to expect.
    """
    print("\n" + "=" * 80)
    print("EXPECTED METRICS GUIDE")
    print("=" * 80)
    print(
        """
For a well-calibrated probabilistic regression model:

POINT METRICS:
  MAE, RMSE, NMAD  - Lower is better. Compare to baseline (e.g., mean prediction).
  Bias             - Should be ~0. Positive = overprediction, negative = underprediction.

PROBABILISTIC METRICS:
  CRPS     - Lower is better. Combines calibration and sharpness.
             Typical range: 0.01-0.1 for normalized data.

  PICP@95% - Should be ~0.95 for 95% confidence intervals.
             < 0.95: Underconfident (intervals too narrow)
             > 0.95: Overconfident (intervals too wide)

  MPIW     - Mean interval width. Narrower is better IF PICP is correct.
             Trade-off: wider intervals = higher PICP but less informative.

  MCE      - Marginal Calibration Error. Should be < 0.05 for well-calibrated.
             Measures average deviation from perfect calibration curve.

DIAGNOSTIC PLOTS:
  Reliability Diagram  - Points should follow the diagonal.
  PIT Histogram       - Should be uniform (flat at height 1).
  Uncertainty vs Error - Should show positive correlation.

COMMON ISSUES:
  ┌─────────────────────────┬────────────────────────┬─────────────────────────┐
  │ Symptom                 │ Likely Cause           │ Fix                     │
  ├─────────────────────────┼────────────────────────┼─────────────────────────┤
  │ PICP << 0.95            │ Underconfident         │ Check variance scaling  │
  │ PICP >> 0.95            │ Overconfident          │ May need variance floor │
  │ PIT peaks at edges      │ Distribution mismatch  │ Try MDN or quantile     │
  │ High-z RMSE >> low-z    │ Tail undersampling     │ Use LDSLoss or weights  │
  │ Low uncertainty-error ρ │ Uninformative uncert.  │ Check heteroscedastic   │
  └─────────────────────────┴────────────────────────┴─────────────────────────┘
"""
    )
