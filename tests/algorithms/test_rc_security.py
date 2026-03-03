import pytest
import torch

from torchregress.algorithms.rc import RegressionCalibration


def test_rc_fit_nan_input_rejection():
    """Test that RegressionCalibration fit rejects NaN inputs."""
    x = torch.randn(10, 2)
    x[0, 0] = float("nan")

    rc = RegressionCalibration(sigma_u=0.1)

    with pytest.raises(ValueError, match="contains NaN values"):
        rc.fit(x)


def test_rc_fit_inf_input_rejection():
    """Test that RegressionCalibration fit rejects Inf inputs."""
    x = torch.randn(10, 2)
    x[0, 0] = float("inf")

    rc = RegressionCalibration(sigma_u=0.1)

    with pytest.raises(ValueError, match="contains infinite values"):
        rc.fit(x)


def test_rc_transform_nan_input_rejection():
    """Test that RegressionCalibration transform rejects NaN inputs."""
    x = torch.randn(10, 2)

    rc = RegressionCalibration(sigma_u=0.1)
    rc.fit(x)

    x_test = torch.randn(10, 2)
    x_test[0, 0] = float("nan")

    with pytest.raises(ValueError, match="contains NaN values"):
        rc.transform(x_test)
