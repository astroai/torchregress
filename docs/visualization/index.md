# Visualization Guide

torchregress includes a comprehensive visualization suite (`torchregress.viz`) for diagnostic analysis, training monitoring, and results presentation.

## Overview

The visualization module provides three categories of plots:

1. **[Diagnostic Plots](diagnostic-plots.md)** - Analyze model behavior and identify issues
2. **[Training Monitoring](training-monitoring.md)** - Track training progress and convergence
3. **[Results Presentation](results-presentation.md)** - Present and compare model performance

## Quick Start

```python
import torch
import torchregress as tr
import torchregress.viz as viz

# Train a model
model = create_model()
# ... training code ...

# Make predictions
with torch.no_grad():
    y_pred = model(X_test)

# Visualize results
viz.plot_residuals(y_pred, y_test)
viz.plot_qq_plot(y_pred, y_test)
```

## Installation

The visualization module requires matplotlib:

```bash
pip install torchregress[viz]
# or
pip install matplotlib seaborn
```

## Complete Function Reference

### Diagnostic Plots

| Function | Purpose | When to Use |
|----------|---------|-------------|
| `plot_residuals()` | Residual scatter plot | Check for patterns, heteroscedasticity |
| `plot_residual_histogram()` | Distribution of residuals | Check normality assumption |
| `plot_qq_plot()` | Quantile-quantile plot | Verify normality of residuals |
| `plot_prediction_intervals()` | Intervals with actual values | Assess uncertainty calibration |
| `plot_calibration_curve()` | Calibration assessment | Check if uncertainties are well-calibrated |
| `plot_reliability_diagram()` | Reliability plot | Validate quantile predictions |
| `plot_distribution_comparison()` | Predicted vs true distributions | Compare full distributions |

### Training Monitoring

| Function | Purpose | When to Use |
|----------|---------|-------------|
| `plot_learning_curves()` | Train/val loss curves | Monitor convergence, detect overfitting |
| `plot_validation_metrics()` | Multiple metrics over time | Track various performance measures |
| `plot_early_stopping()` | Early stopping visualization | Understand when/why training stopped |
| `plot_lr_find_results()` | Learning rate finder | Choose optimal learning rate |

### Results Presentation

| Function | Purpose | When to Use |
|----------|---------|-------------|
| `plot_performance_comparison()` | Compare multiple models | Model selection, benchmarking |
| `plot_feature_importance()` | Feature importance bars | Understand feature contributions |
| `plot_parameter_sensitivity()` | Sensitivity analysis | Hyperparameter tuning insights |
| `plot_model_ensemble_contributions()` | Ensemble member contributions | Analyze ensemble diversity |

### Utilities

| Function | Purpose |
|----------|---------|
| `set_style()` | Set consistent plot style |
| `save_figure()` | Save in multiple formats |
| `create_grid_figure()` | Create multi-panel figures |
| `add_identity_line()` | Add reference lines |
| `create_color_palette()` | Generate color schemes |
| `enable_latex_rendering()` | Use LaTeX in plots |
| `format_metric_label()` | Pretty metric names |

## Common Workflows

### Workflow 1: Model Diagnosis

```python
import torchregress.viz as viz

# Set consistent style
viz.set_style(style='whitegrid', context='paper')

# 1. Check residuals
viz.plot_residuals(y_pred, y_test,
                   title='Residual Analysis',
                   figsize=(10, 6))

# 2. Check normality
viz.plot_qq_plot(y_pred, y_test,
                title='Q-Q Plot')

# 3. Check distribution
viz.plot_residual_histogram(y_pred, y_test,
                            show_kde=True)
```

### Workflow 2: Training Monitoring

```python
# During training, collect metrics
train_losses = []
val_losses = []

for epoch in range(n_epochs):
    # Training...
    train_losses.append(train_loss)
    val_losses.append(val_loss)

# Visualize
viz.plot_learning_curves(
    train_history={'loss': train_losses},
    val_history={'loss': val_losses},
    log_scale=['loss']
)
```

### Workflow 3: Uncertainty Validation

```python
# For models with uncertainty
mean, std = model.predict_with_uncertainty(X_test)
lower = mean - 1.96 * std
upper = mean + 1.96 * std

# Check calibration
viz.plot_prediction_intervals(
    y_pred=mean,
    y_lower=lower,
    y_upper=upper,
    y_true=y_test,
    figsize=(12, 6)
)

viz.plot_calibration_curve(mean, std, y_test)
```

### Workflow 4: Model Comparison

```python
# Compare multiple models
results = {
    'Baseline': {'RMSE': 0.25, 'MAE': 0.18, 'R²': 0.85},
    'Improved': {'RMSE': 0.22, 'MAE': 0.16, 'R²': 0.88},
    'Best': {'RMSE': 0.20, 'MAE': 0.15, 'R²': 0.90}
}

viz.plot_performance_comparison(
    results,
    plot_type='bar',
    highlight_best=True
)
```

## Customization

All plotting functions support extensive customization:

```python
# Customize figure
viz.plot_residuals(
    y_pred, y_test,
    figsize=(12, 8),
    title='Custom Residual Plot',
    xlabel='Custom X Label',
    ylabel='Custom Y Label',
    color='steelblue',
    alpha=0.6,
    s=50,  # marker size
    return_figure=True
)

# Save in multiple formats
fig = viz.plot_residuals(y_pred, y_test, return_figure=True)
viz.save_figure(fig, 'residuals', formats=['png', 'pdf', 'svg'])
```

## Styling

### Global Style

```python
# Set publication-quality style
viz.set_style(
    style='whitegrid',
    context='paper',  # or 'notebook', 'talk', 'poster'
    font_scale=1.2
)
```

### Custom Colors

```python
# Create colorblind-friendly palette
colors = viz.create_color_palette(
    n_colors=5,
    palette='colorblind'  # or 'viridis', 'husl', etc.
)
```

### LaTeX Rendering

```python
# Enable LaTeX for professional plots
viz.enable_latex_rendering()

# Now use LaTeX in labels
plt.xlabel(r'$\hat{y}$ (Predicted)')
plt.ylabel(r'$y - \hat{y}$ (Residual)')
plt.title(r'Residual Plot ($n=1000$)')
```

## Best Practices

### 1. Always Set Style First

```python
import torchregress.viz as viz

# At the beginning of your script
viz.set_style(style='whitegrid', context='paper')

# Then create all plots
```

### 2. Use Consistent Colors

```python
# Define colors once
primary_color = 'steelblue'
secondary_color = 'coral'

# Use throughout
viz.plot_residuals(y_pred, y_test, color=primary_color)
```

### 3. Save for Publications

```python
# High-resolution figures
fig = viz.plot_residuals(y_pred, y_test,
                         figsize=(10, 6),
                         return_figure=True)

viz.save_figure(fig, 'residuals',
               formats=['png', 'pdf'],
               dpi=300)
```

### 4. Create Multi-Panel Figures

```python
fig, axes = viz.create_grid_figure(nrows=2, ncols=2, figsize=(12, 10))

# Plot on each axis
viz.plot_residuals(y_pred, y_test, ax=axes[0, 0])
viz.plot_qq_plot(y_pred, y_test, ax=axes[0, 1])
viz.plot_residual_histogram(y_pred, y_test, ax=axes[1, 0])
viz.plot_calibration_curve(mean, std, y_test, ax=axes[1, 1])

plt.tight_layout()
plt.show()
```

## Examples

See detailed examples in:

- [Diagnostic Plots Guide](diagnostic-plots.md) - Complete diagnostic workflow
- [Training Monitoring Guide](training-monitoring.md) - Track training effectively
- [Results Presentation Guide](results-presentation.md) - Publication-quality figures
- [Cookbook](cookbook.md) - Common visualization recipes

## API Reference

For complete parameter documentation:

- [API: Visualization](../api/viz.md)

## Troubleshooting

### Plot doesn't show

```python
# Make sure to call plt.show()
import matplotlib.pyplot as plt

viz.plot_residuals(y_pred, y_test)
plt.show()  # Add this!
```

### Figure too small

```python
# Increase figsize
viz.plot_residuals(y_pred, y_test, figsize=(12, 8))
```

### Need to modify plot

```python
# Get figure object
fig = viz.plot_residuals(y_pred, y_test, return_figure=True)

# Modify
ax = fig.axes[0]
ax.set_ylim(-10, 10)
ax.grid(True, alpha=0.3)

plt.show()
```

### Plot on existing axes

```python
fig, ax = plt.subplots(figsize=(10, 6))

# Pass ax to plot function
viz.plot_residuals(y_pred, y_test, ax=ax)

# Add more customization
ax.set_title('My Custom Title')
plt.show()
```

## Next Steps

- Start with [Diagnostic Plots](diagnostic-plots.md) for model analysis
- Learn [Training Monitoring](training-monitoring.md) for effective training
- Master [Results Presentation](results-presentation.md) for publications
- Check the [Cookbook](cookbook.md) for common patterns
