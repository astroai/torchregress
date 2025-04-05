# Examples

This section contains practical examples of using TorchRegression for various regression tasks. The examples are designed to demonstrate key features and use cases of the library.

## Basic Examples

### [Basic Usage](basic_usage.md)

This tutorial covers the fundamental usage patterns of TorchRegression:

- Setting up simple regression models
- Choosing and applying loss functions
- Evaluating models with appropriate metrics
- Basic visualization of results

```python
import torch
import torchregress as tr

# Example of basic TorchRegression usage
X_train, y_train = load_data()
model = MyRegressionModel()
loss_fn = tr.losses.HuberLoss()

# Train model
# ...

# Evaluate
predictions = model(X_test)
rmse = tr.metrics.rmse(predictions, y_test)
```

### [Loss Comparison](loss_comparison.md)

This example compares different loss functions on the same dataset:

- Comparing traditional vs. robust losses
- Analyzing how different losses handle outliers
- Visualizing the effect of loss choice on predictions
- Determining which loss is best for different data characteristics

```python
import torchregress as tr

# Dictionary of loss functions to compare
losses = {
    "MSE": tr.losses.MSELoss(),
    "MAE": tr.losses.L1Loss(),
    "Huber": tr.losses.HuberLoss(),
    "LogCosh": tr.losses.LogCoshLoss(),
}

# Compare losses and visualize results
# ...
```

## Advanced Examples

### [Photometric Redshift Estimation](photoz.md)

A real-world application for astronomy:

- Implementing uncertainty-aware regression for photometric redshift estimation
- Using specialized loss functions for astronomical data
- Creating calibrated prediction intervals
- Evaluating results with domain-specific metrics

```python
import torchregress as tr

# Load astronomical dataset
# ...

# Setup specialized model for photometric redshift estimation
model = PhotozModel()
loss_fn = tr.losses.GaussianNLLLoss()

# Train and evaluate
# ...

# Visualize results with specialized plots
tr.viz.plot_redshift_comparison(z_pred, z_spec)
tr.viz.plot_calibration_curve(z_pred, z_err, z_spec)
```

## Planned Examples

We are working on additional examples that will be added soon:

1. **Time Series Forecasting with Uncertainty**
   - Predicting future values with prediction intervals
   - Handling temporal dependencies
   - Evaluating forecasts with specialized metrics

2. **Medical Outcome Prediction**
   - Reliable uncertainty estimation for clinical decision support
   - Risk-aware regression for patient outcomes
   - Calibrated prediction intervals for medical applications

3. **Financial Risk Modeling**
   - Predicting asset returns with uncertainty
   - Value-at-Risk estimation with different loss functions
   - Evaluating prediction intervals for financial applications

## Running the Examples

All examples can be run directly from the repository:

```bash
# Clone the repository
git clone https://github.com/username/torchregress.git
cd torchregress

# Install dependencies
pip install -e '.[examples]'

# Run a specific example
python examples/basic_usage.py
```

For questions or issues with the examples, please [open an issue](https://github.com/username/torchregress/issues) in the repository.
