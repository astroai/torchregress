"""Tests for API consistency across all loss functions in torchregress.

This module ensures that all loss functions in the library follow consistent
API patterns, including parameter ordering, inheritance structure, and
reduction behavior.
"""

import importlib
import inspect
import pkgutil
from typing import List, Type

import pytest
import torch

# Import base classes
from torchregress.losses.base import BaseLoss

# Alias for tests expecting a ReductionLoss base
ReductionLoss = BaseLoss


def get_all_loss_classes() -> List[Type[BaseLoss]]:
    """Dynamically discover all loss classes in the torchregress.losses package.

    Returns:
        List of all loss classes that inherit from the base Loss class.
    """
    loss_classes = []

    # Import the losses package
    import torchregress.losses as losses_pkg

    # List of modules to skip due to dependency issues
    skip_modules = ["rag"]

    # Iterate through all modules in the losses package
    for _, module_name, _ in pkgutil.iter_modules(losses_pkg.__path__):
        if (
            module_name == "base" or module_name in skip_modules
        ):  # Skip the base module and problematic modules
            continue

        try:
            # Import the module
            module = importlib.import_module(f"torchregress.losses.{module_name}")

            # Find all classes in the module that inherit from Loss
            for name, obj in module.__dict__.items():
                if isinstance(obj, type) and issubclass(obj, BaseLoss) and obj != BaseLoss:
                    loss_classes.append(obj)
        except ImportError:
            print(f"Skipping module {module_name} due to import error")
            continue

    return loss_classes


def test_parameter_ordering():
    """Test that all loss classes have y_pred as the first parameter in forward()."""
    # Special cases that have different parameter patterns
    special_cases = {
        "CoTeachingLoss": "y_pred1",  # Takes two predictions
        "RENTLoss": "ensemble_preds",  # Takes ensemble predictions
    }

    for loss_class in get_all_loss_classes():
        # Get the signature of the forward method
        sig = inspect.signature(loss_class.forward)
        params = list(sig.parameters.keys())

        # Skip 'self' parameter
        if params[0] == "self":
            params = params[1:]

        # Check for special cases
        if loss_class.__name__ in special_cases:
            expected_param = special_cases[loss_class.__name__]
            assert params[0] == expected_param, (
                f"{loss_class.__name__}.forward() should have {expected_param} "
                f"as first parameter, got {params[0]}"
            )
        else:
            # Check that y_pred is the first parameter
            assert params[0] == "y_pred", (
                f"{loss_class.__name__}.forward() should have y_pred as first parameter, "
                f"got {params[0]}"
            )


def test_base_class_inheritance():
    """Test that all loss classes inherit from the base Loss class."""
    for loss_class in get_all_loss_classes():
        assert issubclass(
            loss_class, BaseLoss
        ), f"{loss_class.__name__} should inherit from BaseLoss"


def test_reduction_behavior():
    """Test that reduction behavior is consistent across all ReductionLoss classes."""
    # Test data - ensure all values are positive for classes that need positive data
    y_pred = torch.abs(torch.randn(10, 1))
    y_true = torch.abs(torch.randn(10, 1))

    reduction_values = ["mean", "sum", "none"]

    # List of classes that need special initialization and shouldn't be tested this way
    skip_classes = [
        "BaseEIVLoss",
        "DistributionLoss",
        "BaseLoss",
        "RegressionLoss",
        "FunctionalEIVLoss",
        "StructuralEIVLoss",
        "OrthogonalDistanceRegressionLoss",
        "EnsembleEIVLoss",
        "GaussianNLLLoss",
        "LowRankGaussianLoss",
        "MultivariateGaussianLoss",
        "MixtureDensityLoss",
        "EvidentialRegressionLoss",
        "MultiExpectileLoss",
        "MultiQuantileLoss",
        "QuantileLoss",
        "ExpectileLoss",
        "ExpectileCrossoverLoss",
        "PoissonDevianceLoss",
        "PoissonLikelihoodRatioLoss",
        "ZeroInflatedPoissonNLLLoss",
        "NegativeBinomialNLLLoss",
        "PoissonGaussianMixtureLoss",
        "EnhancedPoissonGaussianMixtureLoss",
        "PoissonGaussianLikelihoodRatioLoss",
        "QuantileCrossoverLoss",
        "ConformalLoss",  # Needs special input shape
        "MultiDimensionalConformalLoss",  # Needs special input shape
        "DiagonalGaussianNLL",  # Needs variance input
        "WeightedLossWrapper",  # Wrapper class
        "NormalizingFlowLoss",  # Needs flow parameters
        "CoTeachingLoss",  # Needs two predictions
        "RENTLoss",  # Needs ensemble predictions
        "DensityWeightedLoss",  # Needs to be fitted first
        "LDSLoss",  # Needs to be fitted first
        "NoiseAdaptiveLoss",  # Needs n_samples parameter
    ]

    for loss_class in get_all_loss_classes():
        # Skip classes that need special initialization
        if loss_class.__name__ in skip_classes:
            continue

        # Create default constructor args
        default_args = _get_default_args(loss_class)

        # Test each reduction mode
        for reduction in reduction_values:
            try:
                args = {**default_args, "reduction": reduction}
                loss_fn = loss_class(**args)

                # Some losses may need more inputs
                forward_args = _get_minimal_forward_args(loss_class, y_pred, y_true)
                result = loss_fn(**forward_args)

                if reduction == "none":
                    assert len(result.shape) >= 1, (
                        f"{loss_class.__name__} with reduction='none' "
                        "should return non-scalar tensor"
                    )
                elif reduction in ("mean", "sum"):
                    assert (
                        result.ndim == 0
                    ), f"{loss_class.__name__} with reduction='{reduction}' should return scalar"

            except Exception as e:
                pytest.fail(
                    f"Reduction test failed for {loss_class.__name__} with {reduction}: {str(e)}"
                )


def test_consistent_init_parameters():
    """Test that common init parameters are consistently named across loss classes."""
    common_params = ["reduction"]

    for loss_class in get_all_loss_classes():
        if not issubclass(loss_class, ReductionLoss):
            continue

        # Get the signature of the constructor
        sig = inspect.signature(loss_class.__init__)
        param_names = list(sig.parameters.keys())

        # Skip 'self' parameter
        if param_names[0] == "self":
            param_names = param_names[1:]

        # Check that common parameters are present with expected names
        for param in common_params:
            assert (
                param in param_names
            ), f"{loss_class.__name__}.__init__() should have parameter named '{param}'"


def _get_default_args(cls):
    """Get default arguments for class constructor."""
    signature = inspect.signature(cls.__init__)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty and k != "self"
    }


def _get_minimal_forward_args(loss_class, y_pred, y_true):
    """Create minimal arguments for forward method based on its signature."""
    sig = inspect.signature(loss_class.forward)
    args = {"y_pred": y_pred}

    # Most loss functions have a second parameter for true values
    # The name might vary (y_true, target, targets)
    params = list(sig.parameters.keys())[1:]  # Skip self
    if len(params) >= 2:
        second_param = params[1]  # Second parameter after y_pred
        args[second_param] = y_true

    return args
