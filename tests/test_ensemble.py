"""
Tests for ensemble models and associated components.
"""

import torch
import pytest
from torch import nn

from torchregress.ensemble.base import BaseEnsembleModel
from torchregress.ensemble.models import (
    DeepEnsemble,
    HeteroscedasticEnsembleModel,
    HeteroscedasticBatchEnsembleModel,
)
from torchregress.ensemble.layers import BatchEnsembleLinear
from torchregress.ensemble.utils import (
    run_ensemble_model,
    run_heteroscedastic_ensemble_model,
    generate_prediction_samples,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super().__init__()
        self.layer = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer(x))
        return self.output(x)


class HeteroscedasticModel(nn.Module):
    """Simple heteroscedastic model for testing."""

    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super().__init__()
        self.layer = nn.Linear(input_size, hidden_size)
        self.mean_output = nn.Linear(hidden_size, output_size)
        self.logvar_output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer(x))
        mean = self.mean_output(x)
        logvar = self.logvar_output(x)
        return mean, logvar


class DropoutModel(nn.Module):
    """Model with dropout for MC Dropout testing."""

    def __init__(self, input_size=10, hidden_size=20, output_size=1, dropout_rate=0.2):
        super().__init__()
        self.layer = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer(x))
        x = self.dropout(x)
        return self.output(x)


class BatchEnsembleBackbone(nn.Module):
    """Simple backbone for testing BatchEnsemble."""

    def __init__(self, input_size=10, hidden_size=20):
        super().__init__()
        self.layer = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        return torch.relu(self.layer(x))


class TestBaseEnsembleModel:
    """Tests for BaseEnsembleModel."""

    def test_initialization(self):
        # Test initialization with class
        model = BaseEnsembleModel(
            base_model=SimpleModel, ensemble_size=3, input_size=10, hidden_size=20, output_size=1
        )
        assert len(model.models) == 3

        # Test initialization with instance
        base_instance = SimpleModel(input_size=5, hidden_size=15, output_size=2)
        model = BaseEnsembleModel(base_model=base_instance, ensemble_size=4)
        assert len(model.models) == 4

        # Check that models are different (not just references to the same model)
        for i in range(len(model.models)):
            for j in range(i + 1, len(model.models)):
                # Parameters should be different objects
                for p1, p2 in zip(model.models[i].parameters(), model.models[j].parameters()):
                    assert p1 is not p2

    def test_forward(self):
        model = BaseEnsembleModel(
            base_model=SimpleModel, ensemble_size=3, input_size=10, hidden_size=20, output_size=1
        )

        # Create input tensor
        x = torch.randn(5, 10)  # [batch_size, input_size]

        # Get predictions
        predictions = model(x)

        # Check predictions
        assert len(predictions) == 3  # One prediction per ensemble member
        assert all(
            pred.shape == (5, 1) for pred in predictions
        )  # Each prediction should have shape [batch_size, output_size]

    def test_predict(self):
        model = BaseEnsembleModel(
            base_model=SimpleModel, ensemble_size=3, input_size=10, hidden_size=20, output_size=1
        )

        # Create input tensor
        x = torch.randn(5, 10)  # [batch_size, input_size]

        # Get predictions with uncertainty
        result = model.predict(x)

        # Check result
        assert "mean" in result
        assert "variance" in result
        assert result["mean"].shape == (5, 1)  # [batch_size, output_size]
        assert result["variance"].shape == (5, 1)  # [batch_size, output_size]


class TestDeepEnsemble:
    """Tests for DeepEnsemble."""

    def test_initialization(self):
        model = DeepEnsemble(
            base_model=SimpleModel, ensemble_size=3, input_size=10, hidden_size=20, output_size=1
        )
        assert len(model.models) == 3

    def test_fit_and_predict(self):
        model = DeepEnsemble(
            base_model=SimpleModel,
            ensemble_size=2,  # Small ensemble for quick testing
            input_size=10,
            hidden_size=20,
            output_size=1,
        )

        # Create dummy data
        X_train = torch.randn(20, 10)
        y_train = torch.randn(20, 1)
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5)

        # Create criterion
        criterion = nn.MSELoss()

        # Train model (very briefly for testing purposes)
        history = model.fit(
            train_loader=train_loader, criterion=criterion, epochs=2, lr=0.01, verbose=False
        )

        # Check history
        assert "member_histories" in history
        assert len(history["member_histories"]) == 2

        # Check prediction
        x = torch.randn(3, 10)
        result = model.predict(x)

        assert "mean" in result
        assert "variance" in result
        assert result["mean"].shape == (3, 1)
        assert result["variance"].shape == (3, 1)


class TestHeteroscedasticEnsembleModel:
    """Tests for HeteroscedasticEnsembleModel."""

    def test_initialization(self):
        model = HeteroscedasticEnsembleModel(
            base_model=HeteroscedasticModel,
            ensemble_size=3,
            input_size=10,
            hidden_size=20,
            output_size=1,
        )
        assert len(model.models) == 3

    def test_predict(self):
        model = HeteroscedasticEnsembleModel(
            base_model=HeteroscedasticModel,
            ensemble_size=3,
            input_size=10,
            hidden_size=20,
            output_size=1,
        )

        # Create input tensor
        x = torch.randn(5, 10)  # [batch_size, input_size]

        # Get predictions with uncertainty
        result = model.predict(x)

        # Check result
        assert "mean" in result
        assert "variance" in result
        assert "epistemic_variance" in result
        assert "aleatoric_variance" in result
        assert result["mean"].shape == (5, 1)  # [batch_size, output_size]
        assert result["variance"].shape == (5, 1)  # [batch_size, output_size]
        assert result["epistemic_variance"].shape == (5, 1)  # [batch_size, output_size]
        assert result["aleatoric_variance"].shape == (5, 1)  # [batch_size, output_size]

        # Verify that total variance is sum of epistemic and aleatoric
        assert torch.allclose(
            result["variance"], result["epistemic_variance"] + result["aleatoric_variance"]
        )


class TestBatchEnsembleLinear:
    """Tests for BatchEnsembleLinear layer."""

    def test_initialization(self):
        layer = BatchEnsembleLinear(in_features=10, out_features=5, ensemble_size=4)

        # Check parameters
        assert layer.weight.shape == (5, 10)
        assert layer.r_vectors.shape == (4, 10)
        assert layer.s_vectors.shape == (4, 5)
        assert layer.bias.shape == (5,)

    def test_forward_2d_input(self):
        layer = BatchEnsembleLinear(in_features=10, out_features=5, ensemble_size=4)

        # 2D input [batch_size, in_features]
        x = torch.randn(8, 10)

        # Forward pass
        out = layer(x)

        # Check output shape [batch_size, ensemble_size, out_features]
        assert out.shape == (8, 4, 5)

    def test_forward_3d_input(self):
        layer = BatchEnsembleLinear(in_features=10, out_features=5, ensemble_size=4)

        # 3D input [batch_size, ensemble_size, in_features]
        x = torch.randn(8, 4, 10)

        # Forward pass
        out = layer(x)

        # Check output shape [batch_size, ensemble_size, out_features]
        assert out.shape == (8, 4, 5)

    def test_invalid_input(self):
        layer = BatchEnsembleLinear(in_features=10, out_features=5, ensemble_size=4)

        # 1D input - should raise error
        x = torch.randn(10)

        with pytest.raises(ValueError):
            out = layer(x)

        # 3D input with wrong ensemble size - should raise error
        x = torch.randn(8, 3, 10)  # Ensemble size 3 != 4

        with pytest.raises(ValueError):
            out = layer(x)


class TestHeteroscedasticBatchEnsembleModel:
    """Tests for HeteroscedasticBatchEnsembleModel."""

    def test_initialization(self):
        backbone = BatchEnsembleBackbone(input_size=10, hidden_size=20)
        model = HeteroscedasticBatchEnsembleModel(
            backbone=backbone,
            input_size=20,  # Output size of backbone
            output_size=1,
            ensemble_size=4,
        )

        # Check that output layer is BatchEnsembleLinear
        assert isinstance(model.output_layer, BatchEnsembleLinear)
        assert model.output_layer.in_features == 20
        assert model.output_layer.out_features == 2  # 2*output_size
        assert model.output_layer.ensemble_size == 4

    def test_forward(self):
        backbone = BatchEnsembleBackbone(input_size=10, hidden_size=20)
        model = HeteroscedasticBatchEnsembleModel(
            backbone=backbone, input_size=20, output_size=1, ensemble_size=4
        )

        # Create input tensor
        x = torch.randn(5, 10)  # [batch_size, input_size]

        # Forward pass
        result = model(x)

        # Check result
        assert isinstance(result, dict)
        assert "means" in result
        assert "log_vars" in result
        assert result["means"].shape == (5, 4, 1)  # [batch_size, ensemble_size, output_size]
        assert result["log_vars"].shape == (5, 4, 1)  # [batch_size, ensemble_size, output_size]

    def test_predict(self):
        backbone = BatchEnsembleBackbone(input_size=10, hidden_size=20)
        model = HeteroscedasticBatchEnsembleModel(
            backbone=backbone, input_size=20, output_size=1, ensemble_size=4
        )

        # Create input tensor
        x = torch.randn(5, 10)  # [batch_size, input_size]

        # Predict with uncertainty
        result = model.predict(x)

        # Check result
        assert "mean" in result
        assert "variance" in result
        assert "epistemic_variance" in result
        assert "aleatoric_variance" in result
        assert result["mean"].shape == (5, 1)  # [batch_size, output_size]
        assert result["variance"].shape == (5, 1)  # [batch_size, output_size]
        assert result["epistemic_variance"].shape == (5, 1)  # [batch_size, output_size]
        assert result["aleatoric_variance"].shape == (5, 1)  # [batch_size, output_size]

        # Verify that total variance is sum of epistemic and aleatoric
        assert torch.allclose(
            result["variance"], result["epistemic_variance"] + result["aleatoric_variance"]
        )


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_run_ensemble_model(self):
        # Create a simple model function
        def model_fn(x):
            return x * 2 + torch.randn_like(x) * 0.1

        # Create inputs
        single_input = torch.randn(10, 5)  # [batch_size, features]
        inputs_list = [
            torch.randn(10, 5) for _ in range(5)
        ]  # 5 samples, each [batch_size, features]
        inputs_stacked = torch.stack(inputs_list)  # [n_samples, batch_size, features]

        # Test with list input
        result1 = run_ensemble_model(model_fn, inputs_list)

        # Test with stacked tensor input
        result2 = run_ensemble_model(model_fn, inputs_stacked)

        # Check results
        assert "mean" in result1
        assert "variance" in result1
        assert result1["mean"].shape == (10, 5)  # [batch_size, features]
        assert result1["variance"].shape == (10, 5)  # [batch_size, features]

        # Test with return_individual=True
        result3 = run_ensemble_model(model_fn, inputs_list, return_individual=True)
        assert "individual_preds" in result3
        assert result3["individual_preds"].shape == (5, 10, 5)  # [n_samples, batch_size, features]

    def test_run_heteroscedastic_ensemble_model(self):
        # Create a simple heteroscedastic model function
        def model_fn(x):
            mean = x * 2
            log_var = torch.zeros_like(x) - 1.0
            return mean, log_var

        # Create inputs
        inputs_list = [
            torch.randn(10, 5) for _ in range(5)
        ]  # 5 samples, each [batch_size, features]
        inputs_stacked = torch.stack(inputs_list)  # [n_samples, batch_size, features]

        # Test with stacked tensor input
        result = run_heteroscedastic_ensemble_model(model_fn, inputs_stacked)

        # Check results
        assert "mean" in result
        assert "variance" in result
        assert "epistemic_variance" in result
        assert "aleatoric_variance" in result
        assert result["mean"].shape == (10, 5)  # [batch_size, features]
        assert result["variance"].shape == (10, 5)  # [batch_size, features]
        assert result["epistemic_variance"].shape == (10, 5)  # [batch_size, features]
        assert result["aleatoric_variance"].shape == (10, 5)  # [batch_size, features]

        # Verify that total variance is sum of epistemic and aleatoric
        assert torch.allclose(
            result["variance"], result["epistemic_variance"] + result["aleatoric_variance"]
        )

    def test_generate_prediction_samples(self):
        # Create a dropout model
        model = DropoutModel(input_size=10, hidden_size=20, output_size=2, dropout_rate=0.2)

        # Create input
        x = torch.randn(5, 10)  # [batch_size, input_size]

        # Generate samples
        result = generate_prediction_samples(model, x, n_samples=10)

        # Check results
        assert "mean" in result
        assert "variance" in result
        assert result["mean"].shape == (5, 2)  # [batch_size, output_size]
        assert result["variance"].shape == (5, 2)  # [batch_size, output_size]

        # With return_samples=True
        result = generate_prediction_samples(model, x, n_samples=10, return_samples=True)
        assert "samples" in result
        assert result["samples"].shape == (10, 5, 2)  # [n_samples, batch_size, output_size]
