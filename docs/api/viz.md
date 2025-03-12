# Visualization API Reference

This page documents the visualization functions available in TorchRegression.

## Diagnostic Plots

### `plot_predictions`

```python
def plot_predictions(X, y_true, y_pred, y_std=None, y_lower=None, y_upper=None, 
                    x_label='X', y_label='y', title='Predictions', figsize=(10, 6),
                    sort_x=True, alpha=0.3, show=True, ax=None)
```

Plot predictions with optional uncertainty visualization.

**Parameters:**

- `X` (torch.Tensor or numpy.ndarray): Input features (typically 1D for visualization)
- `y_true` (torch.Tensor or numpy.ndarray): True target values
- `y_pred` (torch.Tensor or numpy.ndarray): Predicted values
- `y_std` (torch.Tensor or numpy.ndarray, optional): Standard deviation of predictions. Default: None
- `y_lower` (torch.Tensor or numpy.ndarray, optional): Lower bound of prediction interval. Default: None
- `y_upper` (torch.Tensor or numpy.ndarray, optional): Upper bound of prediction interval. Default: None
- `x_label` (str, optional): Label for x-axis. Default: 'X'
- `y_label` (str, optional): Label for y-axis. Default: 'y'
- `title` (str, optional): Plot title. Default: 'Predictions'
- `figsize` (tuple, optional): Figure size. Default: (10, 6)
- `sort_x` (bool, optional): Whether to sort X values for better visualization. Default: True
- `alpha` (float, optional): Transparency for scatter points. Default: 0.3
- `show` (bool, optional): Whether to display the plot. Default: True
- `ax` (matplotlib.axes.Axes, optional): Axes to plot on. Default: None

**Returns:**

- matplotlib.figure.Figure: The figure object
- matplotlib.axes.Axes: The axes object

**Example:**

```python
fig, ax = tr.viz.plot_predictions(X, y_true, y_pred, y_std=y_std)
```

### `plot_residuals`

```python
def plot_residuals(y_pred, y_true, title='Residuals', figsize=(10, 6), 
                  alpha=0.5, show=True, ax=None)
```

Plot residuals (y_true - y_pred).

**Parameters:**

- `y_pred` (torch.Tensor or numpy.ndarray): Predicted values
- `y_true` (torch.Tensor or numpy.ndarray): True target values
- `title` (str, optional): Plot title. Default: 'Residuals'
- `figsize` (tuple, optional): Figure size. Default: (10, 6)
- `alpha` (float, optional): Transparency for scatter points. Default: 0.5
- `show` (bool, optional): Whether to display the plot. Default: True
- `ax` (matplotlib.axes.Axes, optional): Axes to plot on. Default: None

**Returns:**

- matplotlib.figure.Figure: The figure object
- matplotlib.axes.Axes: The axes object

**Example:**

```python
fig, ax = tr.viz.plot_residuals(y_pred, y_true)
```

### `plot_qq`

```python
def plot_qq(y_pred, y_true, title='Q-Q Plot', figsize=(8, 8),
           show=True, ax=None)
```

Create a quantile-quantile plot to assess normality of residuals.

**Parameters:**

- `y_pred` (torch.Tensor or numpy.ndarray): Predicted values
- `y_true` (torch.Tensor or numpy.ndarray): True target values
- `title` (str, optional): Plot title. Default: 'Q-Q Plot'
- `figsize` (tuple, optional): Figure size. Default: (8, 8)
- `show` (bool, optional): Whether to display the plot. Default: True
- `ax` (matplotlib.axes.Axes, optional): Axes to plot on. Default: None

**Returns:**

- matplotlib.figure.Figure: The figure object
- matplotlib.axes.Axes: The axes object

**Example:**

```python
fig, ax = tr.viz.plot_qq(y_pred, y_true)
```

### `plot_calibration_curve`

```python
def plot_calibration_curve(y_pred, y_std, y_true, n_bins=10, title='Calibration Curve',
                         figsize=(8, 8), show=True, ax=None)
```

Plot calibration curve for uncertainty estimates.

**Parameters:**

- `y_pred` (torch.Tensor or numpy.ndarray): Predicted mean values
- `y_std` (torch.Tensor or numpy.ndarray): Predicted standard deviation values
- `y_true` (torch.Tensor or numpy.ndarray): True target values
- `n_bins` (int, optional): Number of bins for calibration. Default: 10
- `title` (str, optional): Plot title. Default: 'Calibration Curve'
- `figsize` (tuple, optional): Figure size. Default: (8, 8)
- `show` (bool, optional): Whether to display the plot. Default: True
- `ax` (matplotlib.axes.Axes, optional): Axes to plot on. Default: None

**Returns:**

- matplotlib.figure.Figure: The figure object
- matplotlib.axes.Axes: The axes object

**Example:**

```python
fig, ax = tr.viz.plot_calibration_curve(mean, std, y_true)
```

### `plot_reliability_diagram`

```python
def plot_reliability_diagram(y_lower, y_upper, y_true, title='Reliability Diagram',
                           figsize=(8, 8), show=True, ax=None)
```

Plot reliability diagram for prediction intervals.

**Parameters:**

- `y_lower` (torch.Tensor or numpy.ndarray): Lower bound of prediction intervals
- `y_upper` (torch.Tensor or numpy.ndarray): Upper bound of prediction intervals
- `y_true` (torch.Tensor or numpy.ndarray): True target values
- `title` (str, optional): Plot title. Default: 'Reliability Diagram'
- `figsize` (tuple, optional): Figure size. Default: (8, 8)
- `show` (bool, optional): Whether to display the plot. Default: True
- `ax` (matplotlib.axes.Axes, optional): Axes to plot on. Default: None

**Returns:**

- matplotlib.figure.Figure: The figure object
- matplotlib.axes.Axes: The axes object

**Example:**

```python
fig, ax = tr.viz.plot_reliability_diagram(lower, upper, y_true)
```

## Training Monitoring

### `plot_learning_curve`

```python
def plot_learning_curve(train_losses, val_losses=None, metrics=None, 
                      title='Learning Curve', figsize=(10, 6), show=True, ax=None)
```

Plot learning curves during training.

**Parameters:**

- `train_losses` (list or numpy.ndarray): Training loss values
- `val_losses` (list or numpy.ndarray, optional): Validation loss values. Default: None
- `metrics` (dict, optional): Dictionary of metric names to values. Default: None
- `title` (str, optional): Plot title. Default: 'Learning Curve'
- `figsize` (tuple, optional): Figure size. Default: (10, 6)
- `show` (bool, optional): Whether to display the plot. Default: True
- `ax` (matplotlib.axes.Axes, optional): Axes to plot on. Default: None

**Returns:**

- matplotlib.figure.Figure: The figure object
- matplotlib.axes.Axes: The axes object (or dict of axes if metrics are provided)

**Example:**

```python
fig, axes = tr.viz.plot_learning_curve(
    train_losses, val_losses, 
    metrics={'RMSE': rmse_values, 'MAE': mae_values}
)
```

### `plot_early_stopping`

```python
def plot_early_stopping(val_losses, patience=10, title='Early Stopping',
                      figsize=(10, 6), show=True, ax=None)
```

Visualize early stopping behavior.

**Parameters:**

- `val_losses` (list or numpy.ndarray): Validation loss values
- `patience` (int, optional): Early stopping patience. Default: 10
- `title` (str, optional): Plot title. Default: 'Early Stopping'
- `figsize` (tuple, optional): Figure size. Default: (10, 6)
- `show` (bool, optional): Whether to display the plot. Default: True
- `ax` (matplotlib.axes.Axes, optional): Axes to plot on. Default: None

**Returns:**

- matplotlib.figure.Figure: The figure object
- matplotlib.axes.Axes: The axes object
- int: The optimal early stopping epoch

**Example:**

```python
fig, ax, best_epoch = tr.viz.plot_early_stopping(val_losses, patience=5)
```

## Results Visualization

### `plot_comparison`

```python
def plot_comparison(true_values, predictions_dict, title='Model Comparison', 
                  figsize=(12, 8), show=True, ax=None)
```

Compare predictions from multiple models.

**Parameters:**

- `true_values` (torch.Tensor or numpy.ndarray): True target values
- `predictions_dict` (dict): Dictionary mapping model names to their predictions
- `title` (str, optional): Plot title. Default: 'Model Comparison'
- `figsize` (tuple, optional): Figure size. Default: (12, 8)
- `show` (bool, optional): Whether to display the plot. Default: True
- `ax` (matplotlib.axes.Axes, optional): Axes to plot on. Default: None

**Returns:**

- matplotlib.figure.Figure: The figure object
- matplotlib.axes.Axes: The axes object

**Example:**

```python
fig, ax = tr.viz.plot_comparison(
    y_true, 
    {
        'MSE': mse_preds, 
        'Huber': huber_preds, 
        'Quantile': quantile_preds
    }
)
```

### `plot_parameter_sensitivity`

```python
def plot_parameter_sensitivity(parameter_values, metrics_dict, 
                             param_name='Parameter', title='Parameter Sensitivity',
                             figsize=(10, 6), show=True, ax=None)
```

Plot sensitivity of model performance to hyperparameter values.

**Parameters:**

- `parameter_values` (list or numpy.ndarray): Values of the parameter
- `metrics_dict` (dict): Dictionary mapping metric names to lists of values
- `param_name` (str, optional): Name of the parameter. Default: 'Parameter'
- `title` (str, optional): Plot title. Default: 'Parameter Sensitivity'
- `figsize` (tuple, optional): Figure size. Default: (10, 6)
- `show` (bool, optional): Whether to display the plot. Default: True
- `ax` (matplotlib.axes.Axes, optional): Axes to plot on. Default: None

**Returns:**

- matplotlib.figure.Figure: The figure object
- matplotlib.axes.Axes: The axes object

**Example:**

```python
fig, ax = tr.viz.plot_parameter_sensitivity(
    [0.01, 0.1, 1.0, 10.0],
    {'RMSE': [0.5, 0.4, 0.3, 0.6], 'MAE': [0.4, 0.3, 0.2, 0.5]},
    param_name='Huber Delta'
)
```

### `plot_feature_importance`

```python
def plot_feature_importance(feature_names, importance_values, title='Feature Importance',
                          figsize=(10, 8), show=True, ax=None)
```

Plot feature importance values.

**Parameters:**

- `feature_names` (list): Names of features
- `importance_values` (torch.Tensor or numpy.ndarray): Importance values for each feature
- `title` (str, optional): Plot title. Default: 'Feature Importance'
- `figsize` (tuple, optional): Figure size. Default: (10, 8)
- `show` (bool, optional): Whether to display the plot. Default: True
- `ax` (matplotlib.axes.Axes, optional): Axes to plot on. Default: None

**Returns:**

- matplotlib.figure.Figure: The figure object
- matplotlib.axes.Axes: The axes object

**Example:**

```python
fig, ax = tr.viz.plot_feature_importance(
    ['feature1', 'feature2', 'feature3'], 
    [0.5, 0.3, 0.2]
)
```
