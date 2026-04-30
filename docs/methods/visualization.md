# Visualization Guide

torchregress provides powerful visualization tools for regression analysis and uncertainty estimation. These visualizations help you understand model behavior, diagnose issues, and communicate results effectively.

## Training Visualizations

### Learning Curves

Monitor your model's training progress with learning curves:

```python
from torchregress.viz import plot_learning_curves

# Assuming you've tracked losses during training
plot_learning_curves(
    train_history={'loss': train_losses, 'rmse': train_rmse},
    val_history={'loss': val_losses, 'rmse': val_rmse},
    log_scale=['loss'],  # Use log scale for loss
    smoothing=0.2        # Apply light smoothing to curves
)
```

This plot helps you identify overfitting (gap between train and validation growing) or underfitting (both curves plateau at high values).

### Learning Rate Finder

Find the optimal learning rate for your training:

```python
from torchregress.viz import plot_lr_find_results

# After running learning rate finder
plot_lr_find_results(
    learning_rates=lr_values,
    losses=loss_values,
    suggestion_method='valley'  # 'valley', 'steepest', or 'minimum'
)
```

The plot identifies the learning rate where the loss decreases most rapidly, helping you select an optimal value.

## Diagnostic Visualizations

### Residual Analysis

Analyze model residuals to diagnose systematic errors:

```python
from torchregress.viz import plot_residuals, plot_residual_histogram, plot_qq_plot

# Basic residual plot
plot_residuals(y_pred, y_true, clip_outliers=True)

# Distribution of residuals
plot_residual_histogram(y_pred, y_true, show_kde=True)

# Check if residuals follow normal distribution
plot_qq_plot(y_pred, y_true)
```

These plots help you identify heteroscedasticity (non-constant variance), non-linearity, and other violations of regression assumptions.

### Uncertainty Visualization

Visualize prediction intervals and uncertainty:

```python
from torchregress.viz import plot_prediction_intervals

# For models that predict uncertainty
plot_prediction_intervals(
    y_pred=mean_predictions,
    y_lower=lower_bounds,    # E.g., mean - 1.96 * std_dev for 95% interval
    y_upper=upper_bounds,    # E.g., mean + 1.96 * std_dev for 95% interval
    y_true=y_test
)
```

### Uncertainty Calibration

Check if your uncertainty estimates are well-calibrated:

```python
from torchregress.viz import plot_reliability_diagram

# For models that predict quantiles
plot_reliability_diagram(
    y_pred_quantiles={0.1: q10_preds, 0.5: q50_preds, 0.9: q90_preds},
    y_true=y_test
)
```

The closer the plot is to the diagonal line, the better calibrated your uncertainty estimates are.

## Results Visualization

### Model Comparison

Compare multiple models across different metrics:

```python
from torchregress.viz import plot_performance_comparison

# Dictionary of models and their metrics
metrics = {
    'Model A': {'rmse': 0.25, 'mae': 0.18, 'r2': 0.85},
    'Model B': {'rmse': 0.22, 'mae': 0.16, 'r2': 0.88},
    'Model C': {'rmse': 0.28, 'mae': 0.20, 'r2': 0.81}
}

plot_performance_comparison(
    metrics=metrics,
    highlight_best=True,
    plot_type='bar'  # 'bar', 'radar', or 'heatmap'
)
```

### Feature Importance

Visualize which features are most important for your model:

```python
from torchregress.viz import plot_feature_importance

plot_feature_importance(
    feature_names=feature_names,
    importance_values=importance_scores,
    sort_values=True,
    horizontal=True,
    top_n=10  # Show only top 10 features
)
```

## Advanced Visualization: Predictive Distributions

For probabilistic regression models:

```python
from torchregress.viz import plot_distribution_comparison

# For models that generate samples
plot_distribution_comparison(
    predicted_samples=model_samples,  # Shape: [n_samples, batch_size]
    y_true=y_test,
    n_samples_to_show=4,
    credible_interval=0.95
)
```

This visualization shows the full predicted distribution for selected samples, helping you understand prediction uncertainty on an individual sample level.

## Customization Options

All visualization functions support customization:

```python
from torchregress.viz import set_style, save_figure

# Set consistent style for all plots
set_style(style="whitegrid", context="talk", font_scale=1.2)

# Create plot (any torchregress plot)
fig = plot_residuals(y_pred, y_true, return_figure=True)

# Save in multiple formats
save_figure(fig, "residual_analysis", formats=["png", "pdf"])
```
