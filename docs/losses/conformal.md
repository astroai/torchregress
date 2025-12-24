# Conformal Prediction

The `torchregress.losses.conformal` module provides a unified interface for state-of-the-art conformal prediction methods for regression, leveraging the `torchcp` library as a backend.

## Installation

Conformal prediction relies on `torchcp` and its dependencies. Install the extra:

```bash
uv pip install -e ".[conformal]"
```

`torch-geometric` and its compiled dependencies may require the PyG wheel index; see the
official installation guide if your environment needs it.

## Unified Conformal Prediction Module

The `ConformalLoss` class is a wrapper around various conformal prediction methods. It provides a consistent API for training, calibrating, and generating prediction intervals.

### Supported Methods

- **Split Conformal Prediction (`'split'`)**: A simple and general method for conformal prediction.
- **Conformalized Quantile Regression (`'cqr'`)**: A method that combines quantile regression with conformal prediction to produce more efficient prediction intervals.
- **Adaptive Conformal Inference (`'aci'`)**: A method that adapts the prediction intervals to the difficulty of the input data.

### Usage

To use the unified conformal prediction module, you create a `ConformalLoss` object, specifying the desired `method` and the target miscoverage level `alpha`.

For `'split'` and `'cqr'`:
```python
from torchregress.losses.conformal import ConformalLoss

# Create a CQR loss with a target coverage of 90%
loss_fn = ConformalLoss(method='cqr', alpha=0.1)
```

For `'aci'`, you also need to pass the model to the constructor:
```python
# Create an ACI loss with a target coverage of 90%
loss_fn = ConformalLoss(method='aci', alpha=0.1, model=my_model)
```

During training, the `ConformalLoss` object can be used like any other loss function in PyTorch.

After training, you need to calibrate the conformal predictor on a hold-out calibration set.

```python
# Calibrate the predictor on the calibration set
loss_fn.calibrate(cal_preds, y_cal)
```

Finally, you can use the calibrated predictor to generate prediction intervals for new data.

```python
# Get prediction intervals for the test set
lower, upper = loss_fn.predict_interval(test_preds)
```

### Choosing the Right Method

- **`'split'`**: Use this method when you have a simple model and you want a general-purpose conformal prediction method.
- **`'cqr'`**: Use this method when you are using a quantile regression model and you want to produce more efficient prediction intervals.
- **`'aci'`**: Use this method when you have a model that can estimate the difficulty of the input data and you want to produce adaptive prediction intervals.
