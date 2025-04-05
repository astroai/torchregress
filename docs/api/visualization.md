# Visualization API

TorchRegression provides a comprehensive suite of visualization tools to help you understand, diagnose, and communicate your regression models. These visualizations are designed to work seamlessly with PyTorch models and can be easily customized to fit your needs.

## Diagnostic Visualizations

### Residual Analysis

```python
from torchregress.viz import plot_residuals

plot_residuals(
    y_pred, 
    y_true, 
    clip_outliers=True,
    show_trend=True
)
```

**Parameters:**

- **y_pred** : `torch.Tensor` or `numpy.ndarray`  
  Predicted values from your model
- **y_true** : `torch.Tensor` or `numpy.ndarray`  
  Ground truth values
- **clip_outliers** : `bool`, default=`False`  
  Whether to clip outlier residuals for better visualization
- **clip_percentile** : `float`, default=`99.0`  
  Percentile to clip outliers (if `clip_outliers` is True)
- **show_trend** : `bool`, default=`True`  
  Whether to show a trend line for residuals
- **return_figure** : `bool`, default=`False`  
  If True, returns the figure object instead of displaying

**Returns:**

- `matplotlib.figure.Figure` or `None`  
  If `return_figure=True`, returns the figure object

**Usage Notes:**

- Residual plots help identify heteroscedasticity (non-constant variance) and non-linearity
- A good model should show residuals randomly scattered around zero
- Systematic patterns in residuals indicate model deficiencies
- For large datasets, use `downsample=True` to improve visualization performance

---

### Prediction Intervals

```python
from torchregress.viz import plot_prediction_intervals

plot_prediction_intervals(
    y_pred=mean_predictions,
    y_lower=lower_bounds,
    y_upper=upper_bounds,
    y_true=y_test
)
```

**Parameters:**

- **y_pred** : `torch.Tensor` or `numpy.ndarray`  
  Predicted values (mean or median predictions)
- **y_lower** : `torch.Tensor` or `numpy.ndarray`  
  Lower bounds of prediction intervals
- **y_upper** : `torch.Tensor` or `numpy.ndarray`  
  Upper bounds of prediction intervals
- **y_true** : `torch.Tensor` or `numpy.ndarray`, optional  
  Ground truth values
- **sorted_by_pred** : `bool`, default=`False`  
  If True, sorts all values by predicted values for better visualization
- **return_figure** : `bool`, default=`False`  
  If True, returns the figure object instead of displaying

**Returns:**

- `matplotlib.figure.Figure` or `None`  
  If `return_figure=True`, returns the figure object

**Usage Notes:**

- Provides visual assessment of uncertainty and prediction intervals
- Automatically calculates and displays coverage if ground truth is provided
- Perfect calibration would show around 95% coverage for 95% prediction intervals
- Useful for communicating model uncertainty to stakeholders

---

### Reliability Diagram

```python
from torchregress.viz import plot_reliability_diagram

plot_reliability_diagram(
    y_pred_quantiles={0.1: q10_preds, 0.5: q50_preds, 0.9: q90_preds},
    y_true=y_test
)
```

**Parameters:**

- **y_pred_quantiles** : `Dict[float, torch.Tensor]`  
  Dictionary mapping quantile levels to predictions
- **y_true** : `torch.Tensor` or `numpy.ndarray`  
  Ground truth values
- **show_diagonal** : `bool`, default=`True`  
  Whether to show diagonal line (perfect calibration)
- **return_figure** : `bool`, default=`False`  
  If True, returns the figure object instead of displaying

**Returns:**

- `matplotlib.figure.Figure` or `None`  
  If `return_figure=True`, returns the figure object

**Usage Notes:**

- Assesses calibration of uncertainty estimates
- Perfectly calibrated models will follow the diagonal line
- Points below the diagonal indicate overconfidence
- Points above the diagonal indicate underconfidence
- Useful for tuning uncertainty estimation methods

---

### Distribution Comparison

```python
from torchregress.viz import plot_distribution_comparison

plot_distribution_comparison(
    predicted_samples=model_samples,  # Shape: [n_samples, batch_size]
    y_true=y_test,
    n_samples_to_show=4,
    credible_interval=0.95
)
```

**Parameters:**

- **predicted_samples** : `torch.Tensor` or `numpy.ndarray`  
  Samples from predicted distributions [n_samples, batch_size]
- **y_true** : `torch.Tensor` or `numpy.ndarray`  
  Ground truth values [batch_size]
- **n_samples_to_show** : `int`, default=`5`  
  Number of examples to display
- **plot_type** : `str`, default=`'kde'`  
  Plot type ('kde', 'histogram', or 'both')
- **credible_interval** : `float`, default=`0.95`  
  Credible interval for shading (e.g., 0.95 for 95% CI)

**Returns:**

- `matplotlib.figure.Figure` or `None`  
  If `return_figure=True`, returns the figure object

**Usage Notes:**

- Provides detailed view of full predictive distributions for selected samples
- Helps understand shape, spread, and potential multi-modality of predictions
- Compares true values against predicted distributions
- Essential for deep probabilistic regression models like MDNs

## Training Monitoring

### Learning Curves

```python
from torchregress.viz import plot_learning_curves

plot_learning_curves(
    train_history={'loss': train_losses, 'rmse': train_rmse},
    val_history={'loss': val_losses, 'rmse': val_rmse},
    log_scale=['loss'],
    smoothing=0.2
)
```

**Parameters:**

- **train_history** : `Dict[str, List[float]]`  
  Dictionary mapping metric names to lists of values (training)
- **val_history** : `Dict[str, List[float]]`, optional  
  Dictionary mapping metric names to lists of validation values
- **metrics_to_plot** : `List[str]`, optional  
  List of metric names to plot (defaults to all metrics)
- **smoothing** : `float`, default=`0.0`  
  Smoothing factor for the curves (0.0 = no smoothing, 0.9 = high smoothing)
- **log_scale** : `List[str]`, optional  
  List of metrics to display with log scale
- **show_annotations** : `bool`, default=`True`  
  Whether to show best value annotations

**Returns:**

- `matplotlib.figure.Figure` or `None`  
  If `return_figure=True`, returns the figure object

**Usage Notes:**

- Essential for monitoring training progress and convergence
- Helps identify overfitting (gap between train and validation growing)
- Helps identify underfitting (both curves plateau at high values)
- Use `log_scale=['loss']` for better visualization of loss curves
- Apply smoothing to reduce noise in plots for clearer trends

---

### Early Stopping Visualization

```python
from torchregress.viz import plot_early_stopping

plot_early_stopping(
    train_losses=train_loss_history,
    val_losses=val_loss_history,
    patience=10,
    delta=0.001
)
```

**Parameters:**

- **train_losses** : `List[float]`  
  List of training loss values
- **val_losses** : `List[float]`  
  List of validation loss values
- **patience** : `int`, default=`10`  
  Patience parameter used
- **delta** : `float`, default=`0.0`  
  Minimum change to qualify as improvement

**Returns:**

- `matplotlib.figure.Figure` or `None`  
  If `return_figure=True`, returns the figure object

**Usage Notes:**

- Visualizes early stopping behavior to understand training dynamics
- Highlights best model point and early stopping point
- Shows patience window where model was waiting for improvement
- Useful for tuning patience and delta hyperparameters

---

### Learning Rate Finder

```python
from torchregress.viz import plot_lr_find_results

lr, suggested_lr = plot_lr_find_results(
    learning_rates=lr_values,
    losses=loss_values,
    suggestion_method='valley',
    return_figure=True
)
```

**Parameters:**

- **learning_rates** : `List[float]`  
  List of learning rates tested
- **losses** : `List[float]`  
  Corresponding loss values
- **smoothing** : `float`, default=`0.05`  
  Smoothing factor for the loss curve
- **suggestion_method** : `str`, default=`'valley'`  
  Method to suggest learning rate ('valley', 'steepest', or 'minimum')

**Returns:**

- `matplotlib.figure.Figure` or `None`  
  If `return_figure=True`, returns the figure object
- `float` or `None`  
  If `return_figure=True`, also returns the suggested learning rate

**Usage Notes:**

- Helps find optimal learning rate for your optimizer
- 'valley' method looks for the learning rate just before loss starts increasing
- 'steepest' method finds the point of steepest descent on loss curve
- 'minimum' method simply suggests the minimum loss point
- Using a learning rate slightly lower than suggested often gives better stability

## Results Visualization

### Model Comparison

```python
from torchregress.viz import plot_performance_comparison

plot_performance_comparison(
    metrics={
        'Model A': {'rmse': 0.25, 'mae': 0.18, 'r2': 0.85},
        'Model B': {'rmse': 0.22, 'mae': 0.16, 'r2': 0.88},
        'Model C': {'rmse': 0.28, 'mae': 0.20, 'r2': 0.81}
    },
    highlight_best=True,
    plot_type='bar'  # 'bar', 'radar', or 'heatmap'
)
```

**Parameters:**

- **metrics** : `Dict[str, Dict[str, float]]`  
  Dictionary mapping model names to dictionaries of metrics
- **highlight_best** : `bool`, default=`True`  
  Whether to highlight the best model for each metric
- **higher_is_better** : `Dict[str, bool]`, optional  
  Dictionary specifying for each metric if higher values are better
- **plot_type** : `str`, default=`'bar'`  
  Type of plot ('bar', 'radar', or 'heatmap')

**Returns:**

- `matplotlib.figure.Figure` or `None`  
  If `return_figure=True`, returns the figure object

**Usage Notes:**

- Provides clear visual comparison of multiple models across metrics
- Bar plots give detailed comparison of specific metrics
- Radar plots provide holistic view of model performance
- Heatmaps are useful for comparing many models and metrics
- Automatically determines if higher or lower is better for common metrics

---

### Feature Importance

```python
from torchregress.viz import plot_feature_importance

plot_feature_importance(
    feature_names=feature_names,
    importance_values=importance_scores,
    importance_errors=importance_std_devs,  # Optional
    sort_values=True,
    horizontal=True,
    top_n=10
)
```

**Parameters:**

- **feature_names** : `List[str]`  
  Names of features
- **importance_values** : `torch.Tensor` or `numpy.ndarray` or `List[float]`  
  Importance score for each feature
- **importance_errors** : `torch.Tensor` or `numpy.ndarray` or `List[float]`, optional  
  Error/uncertainty for importance values
- **sort_values** : `bool`, default=`True`  
  Whether to sort features by importance
- **horizontal** : `bool`, default=`True`  
  Whether to create a horizontal bar chart
- **top_n** : `int`, optional  
  Optionally limit to top N features

**Returns:**

- `matplotlib.figure.Figure` or `None`  
  If `return_figure=True`, returns the figure object

**Usage Notes:**

- Essential for model interpretability
- Works with importance scores from various methods:
  - Permutation importance
  - SHAP values
  - Feature attributions from gradient-based methods
  - Model-specific importance (e.g., tree-based models)
- Including error bars provides insight into feature importance stability
- Horizontal layout often works better for models with many features

---

### Parameter Sensitivity

```python
from torchregress.viz import plot_parameter_sensitivity

plot_parameter_sensitivity(
    parameter_values={'learning_rate': [0.001, 0.01, 0.1], 'dropout': [0.0, 0.2, 0.5]},
    metric_values={'rmse': [0.24, 0.22, 0.28], 'r2': [0.82, 0.85, 0.79]}
)
```

**Parameters:**

- **parameter_values** : `Dict[str, List[Union[float, int, str]]]`  
  Dictionary mapping parameter names to lists of values tested
- **metric_values** : `Dict[str, List[float]]`  
  Dictionary mapping metric names to lists of resulting values
- **highlight_best** : `bool`, default=`True`  
  Whether to highlight the best parameter value for each metric
- **plot_type** : `str`, default=`'line'`  
  Type of plot ('line' or 'bar')

**Returns:**

- `matplotlib.figure.Figure` or `None`  
  If `return_figure=True`, returns the figure object

**Usage Notes:**

- Helps understand how model parameters affect performance
- Useful for hyperparameter tuning and sensitivity analysis
- Line plots show trends across parameter values
- Bar plots emphasize discrete parameter options
- Automatically determines if higher or lower is better for common metrics

## Customization and Utilities

### Setting Plot Style

```python
from torchregress.viz import set_style

# Set style for all subsequent plots
set_style(
    style="whitegrid",      # 'default', 'whitegrid', 'darkgrid', 'ticks', 'minimal'
    context="talk",         # 'paper', 'notebook', 'talk', 'poster'
    font_scale=1.2
)
```

**Parameters:**

- **style** : `str`, default=`'default'`  
  Style name ('default', 'whitegrid', 'darkgrid', 'ticks', or 'minimal')
- **context** : `str`, default=`'notebook'`  
  Context name ('paper', 'notebook', 'talk', or 'poster')
- **font_scale** : `float`, default=`1.0`  
  Scale factor for font sizes
- **rc** : `Dict[str, Any]`, optional  
  Dictionary of rc parameter mappings to override

**Usage Notes:**

- Set at the beginning of your notebook/script for consistent styling
- 'paper' context uses small fonts suitable for academic papers
- 'talk' and 'poster' contexts use larger fonts suitable for presentations
- Uses seaborn styles if available, falls back to matplotlib styles

---

### Saving Figures

```python
from torchregress.viz import save_figure

# Create any plot with return_figure=True
fig = plot_residuals(y_pred, y_true, return_figure=True)

# Save in multiple formats
save_figure(
    fig, 
    filename="residual_analysis",
    directory="./figures",
    formats=["png", "pdf", "svg"],
    dpi=300
)
```

**Parameters:**

- **fig** : `matplotlib.figure.Figure`  
  Matplotlib figure to save
- **filename** : `str`  
  Base filename (without extension)
- **directory** : `str`, default=`'./figures'`  
  Output directory
- **formats** : `List[str]`, default=`['png', 'pdf']`  
  List of formats to save (e.g., ["png", "pdf", "svg"])
- **dpi** : `int`, default=`300`  
  Resolution for raster formats
- **transparent** : `bool`, default=`False`  
  Whether to use transparent background

**Usage Notes:**

- Creates output directory if it doesn't exist
- Saves the same figure in multiple formats
- PNG files are good for web/screens
- PDF files are vector-based and good for publications
- Higher DPI values produce larger but clearer images
