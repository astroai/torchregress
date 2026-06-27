import pytest
import torch
import torch.nn as nn

from torchregress.ensemble import (
    SWAG,
    BayesianNeuralNetwork,
    HeteroscedasticBNN,
    MCDropoutWrapper,
    MultiSWAG,
)


def _dropout_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(16, 1),
    )


def test_mc_dropout_wrapper_uncertainty_shapes_and_intervals():
    torch.manual_seed(0)
    model = MCDropoutWrapper(_dropout_model(), n_samples=8)
    x = torch.randn(6, 4)

    mean, std = model.predict_with_uncertainty(x)
    lower, upper = model.predict_interval(x, confidence=0.9)

    assert mean.shape == (6, 1)
    assert std.shape == (6, 1)
    assert lower.shape == (6, 1)
    assert upper.shape == (6, 1)
    assert torch.all(std >= 0)
    assert torch.all(lower <= upper)


def test_swag_collect_sample_and_forward_shape():
    torch.manual_seed(0)
    base = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    swag = SWAG(base, max_num_models=4)
    x = torch.randn(5, 4)

    # Collect a few snapshots with small perturbations to create non-zero variance.
    for i in range(3):
        with torch.no_grad():
            for p in base.parameters():
                p.add_(0.01 * (i + 1))
        swag.collect_model(base)

    swag.sample(scale=0.5)
    y = swag(x)
    assert y.shape == (5, 1)


def test_swag_sample_raises_before_any_collection():
    base = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1))
    swag = SWAG(base)

    with pytest.raises(ValueError, match="No models collected yet"):
        swag.sample()


def test_multi_swag_predict_with_uncertainty_shapes():
    torch.manual_seed(0)
    base = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    multi_swag = MultiSWAG(base, n_models=2, max_num_models=4)
    x = torch.randn(4, 4)

    for model_idx, swag_model in enumerate(multi_swag.swag_models):
        for snap in range(2):
            with torch.no_grad():
                for p in swag_model.base_model.parameters():
                    p.add_(0.01 * (model_idx + 1) * (snap + 1))
            swag_model.collect_model(swag_model.base_model)

    mean, epistemic_var, aleatoric_var = multi_swag.predict_with_uncertainty(x, n_samples=2)
    assert mean.shape == (4, 1)
    assert epistemic_var.shape == (4, 1)
    assert aleatoric_var.shape == (4, 1)
    assert torch.all(epistemic_var >= 0)
    assert torch.all(aleatoric_var >= 0)


def test_multi_swag_predict_with_samples_shape_contract():
    torch.manual_seed(0)
    base = nn.Sequential(nn.Linear(2, 6), nn.ReLU(), nn.Linear(6, 1))
    multi_swag = MultiSWAG(base, n_models=2, max_num_models=3)
    x = torch.randn(3, 2)

    for swag_model in multi_swag.swag_models:
        for i in range(2):
            with torch.no_grad():
                for p in swag_model.base_model.parameters():
                    p.add_(0.01 * (i + 1))
            swag_model.collect_model(swag_model.base_model)

    samples = multi_swag.predict_with_samples(x, n_samples=3, scale=0.5)
    assert samples.shape == (2 * 3, 3, 1)
    assert torch.all(torch.isfinite(samples))


def test_mc_dropout_model_interval_monotonicity():
    torch.manual_seed(0)
    model = MCDropoutWrapper(nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(0.2), nn.Linear(8, 1)), n_samples=8)
    x = torch.randn(5, 4)

    lower, upper = model.predict_interval(x, confidence=0.9, n_samples=6)
    assert lower.shape == (5, 1)
    assert upper.shape == (5, 1)
    assert torch.all(lower <= upper)


def test_bnn_predict_with_uncertainty_and_intervals_shapes():
    torch.manual_seed(0)
    model = BayesianNeuralNetwork(input_dim=4, hidden_dims=[8], output_dim=1, n_samples=5)
    x = torch.randn(7, 4)

    mean, std = model.predict_with_uncertainty(x, n_samples=4)
    lower, upper = model.predict_interval(x, confidence=0.8, n_samples=4)

    assert mean.shape == (7, 1)
    assert std.shape == (7, 1)
    assert lower.shape == (7, 1)
    assert upper.shape == (7, 1)
    assert torch.all(std >= 0)
    assert torch.all(lower <= upper)


def test_heteroscedastic_bnn_uncertainty_decomposition_matches_total_std():
    torch.manual_seed(0)
    model = HeteroscedasticBNN(input_dim=4, hidden_dims=[8], output_dim=1, n_samples=5)
    x = torch.randn(5, 4)

    torch.manual_seed(123)
    mean, aleatoric_var, epistemic_var = model.predict_with_decomposition(x, n_samples=4)
    torch.manual_seed(123)
    mean2, total_std = model.predict_with_uncertainty(x, n_samples=4)

    assert mean.shape == (5, 1)
    assert aleatoric_var.shape == (5, 1)
    assert epistemic_var.shape == (5, 1)
    assert total_std.shape == (5, 1)
    assert torch.all(aleatoric_var >= 0)
    assert torch.all(epistemic_var >= 0)
    assert torch.allclose(mean, mean2, atol=1e-5, rtol=1e-4)
    assert torch.allclose(total_std**2, aleatoric_var + epistemic_var, atol=1e-4, rtol=1e-3)
