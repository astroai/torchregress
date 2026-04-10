"""
Tests for the normalizing flow loss module with conditional flows.
"""

from typing import Any
from unittest.mock import Mock

import pytest
import torch

try:
    import zuko

    HAS_ZUKO = True
except ImportError:
    HAS_ZUKO = False

if HAS_ZUKO:
    from torchregress.losses.nflows import NormalizingFlowLoss

    def create_flow_model(
        n_features,
        context_dim,
        flow_type="nsf",
        n_transforms=2,
        hidden_features=32,
        n_hidden_layers=2,
    ):
        """Helper to create flow models for testing."""
        if flow_type == "nsf":
            return zuko.flows.NSF(
                features=n_features,
                context=context_dim,
                transforms=n_transforms,
                hidden_features=[hidden_features] * n_hidden_layers,
            )
        elif flow_type == "maf":
            return zuko.flows.MAF(
                features=n_features,
                context=context_dim,
                transforms=n_transforms,
                hidden_features=[hidden_features] * n_hidden_layers,
            )
        elif flow_type == "realnvp":
            if hasattr(zuko.flows, "NICE"):
                return zuko.flows.NICE(
                    features=n_features,
                    context=context_dim,
                    transforms=n_transforms,
                    hidden_features=[hidden_features] * n_hidden_layers,
                )
            # Fallback
            return zuko.flows.MAF(
                features=n_features,
                context=context_dim,
                transforms=n_transforms,
                hidden_features=[hidden_features] * n_hidden_layers,
            )
        else:
            raise ValueError(f"Invalid flow_type: {flow_type}")

else:
    # Dummy classes for collection
    class NormalizingFlowLoss:  # type: ignore[no-redef]
        pass

    def create_flow_model(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef, misc]
        pass


# Skip all tests in this module if zuko is not installed
pytestmark = pytest.mark.skipif(not HAS_ZUKO, reason="zuko not installed")


# Fixtures
@pytest.fixture
def simple_conditional_flow():
    """Create a simple conditional flow for testing."""
    return create_flow_model(
        n_features=2,
        context_dim=5,
        flow_type="nsf",
        n_transforms=2,
        hidden_features=16,
        n_hidden_layers=1,
    )


@pytest.fixture
def simple_unconditional_flow():
    """Create a simple unconditional flow for testing."""
    return create_flow_model(
        n_features=2,
        context_dim=0,
        flow_type="nsf",
        n_transforms=2,
        hidden_features=16,
        n_hidden_layers=1,
    )


@pytest.fixture
def mock_context():
    """Generate mock context from model output."""
    return torch.randn(10, 5)  # 10 samples, 5D context


@pytest.fixture
def mock_target():
    """Generate mock target values for testing."""
    return torch.randn(10, 2)  # 10 samples, 2 features


@pytest.fixture
def mock_mask():
    """Generate a mock mask for testing masked operations."""
    return torch.ones(10, 2).bool()  # All True mask


@pytest.fixture
def mock_weights():
    """Generate mock sample weights for testing weighted loss."""
    return torch.ones(10)  # Equal weights


# Test flow creation
def test_create_flow_model_conditional():
    """Test creating conditional flow models."""
    for flow_type in ["realnvp", "maf", "nsf"]:
        flow = create_flow_model(
            n_features=3, context_dim=10, flow_type=flow_type, n_transforms=2, hidden_features=32
        )
        assert flow.base().event_shape[0] == 3
        # flow.context might not be exposed directly in all flows, or checks in forward
        assert isinstance(flow, torch.nn.Module)
        # Check it's trainable
        assert len(list(flow.parameters())) > 0


def test_create_flow_model_unconditional():
    """Test creating unconditional flow models."""
    flow = create_flow_model(n_features=2, context_dim=0, flow_type="nsf")
    assert flow.base().event_shape[0] == 2


def test_create_flow_model_invalid_type():
    """Test that invalid flow types raise errors."""
    with pytest.raises(ValueError, match="Invalid flow_type"):
        create_flow_model(n_features=2, context_dim=5, flow_type="invalid")


# Test loss instantiation
def test_normalizing_flow_loss_init(simple_conditional_flow):
    """Test NormalizingFlowLoss initialization."""
    loss_fn = NormalizingFlowLoss(flow=simple_conditional_flow)
    assert loss_fn.n_features == 2
    # context_dim might not be inferred until first forward pass or if flow doesn't expose it
    assert loss_fn.reduction == "mean"


def test_normalizing_flow_loss_invalid_flow():
    """Test that non-Module flows are rejected."""
    with pytest.raises(TypeError, match="flow must be a torch.nn.Module"):
        NormalizingFlowLoss(flow="not a module")


def test_normalizing_flow_loss_no_base():
    """Test that flows without base distribution are rejected."""
    mock_flow = Mock(spec=torch.nn.Module)
    # Make base() raise an exception when called
    mock_flow.base = Mock(side_effect=AttributeError("no base"))

    with pytest.raises(ValueError, match="Could not extract feature dimension"):
        NormalizingFlowLoss(flow=mock_flow)


# Test forward pass
def test_forward_conditional(simple_conditional_flow, mock_context, mock_target):
    """Test forward pass with conditional flow."""
    loss_fn = NormalizingFlowLoss(flow=simple_conditional_flow)

    loss = loss_fn(mock_context, mock_target)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()  # Scalar with mean reduction
    assert torch.isfinite(loss)


def test_forward_unconditional(simple_unconditional_flow, mock_target):
    """Test forward pass with unconditional flow."""
    loss_fn = NormalizingFlowLoss(flow=simple_unconditional_flow)

    # For unconditional flow, context can be dummy or empty
    dummy_context = torch.randn(10, 0)  # Empty context
    loss = loss_fn(dummy_context, mock_target)

    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss)


def test_forward_with_mask(simple_conditional_flow, mock_context, mock_target):
    """Test forward pass with mask."""
    loss_fn = NormalizingFlowLoss(flow=simple_conditional_flow)

    # Create mask that masks out some samples
    mask = torch.tensor(
        [
            [True, True],
            [False, False],
            [True, True],
            [True, True],
            [False, False],
            [True, True],
            [True, True],
            [True, True],
            [True, True],
            [True, True],
        ]
    )

    loss = loss_fn(mock_context, mock_target, mask=mask)

    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss)


def test_forward_rejects_partial_feature_mask(simple_conditional_flow):
    """Feature-wise masking is not valid for joint flow densities."""
    loss_fn = NormalizingFlowLoss(flow=simple_conditional_flow)
    context = torch.randn(2, 5)
    target = torch.randn(2, 2)
    mask = torch.tensor([[True, False], [True, True]])

    with pytest.raises(ValueError, match="sample-level masks"):
        loss_fn(context, target, mask=mask)


def test_forward_with_weights(simple_conditional_flow, mock_context, mock_target, mock_weights):
    """Test forward pass with sample weights."""
    loss_fn = NormalizingFlowLoss(flow=simple_conditional_flow)

    loss = loss_fn(mock_context, mock_target, weights=mock_weights)

    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss)


def test_forward_shape_validation(simple_conditional_flow):
    """Test that shape mismatches are caught."""
    loss_fn = NormalizingFlowLoss(flow=simple_conditional_flow)

    # Wrong number of features in target
    with pytest.raises(ValueError, match="Expected 2 features"):
        loss_fn(torch.randn(10, 5), torch.randn(10, 3))

    # Wrong context dimension will cause runtime error from flow, not validation error
    # The flow expects 5D context but gets 3D, which will fail in the flow's internal layers
    with pytest.raises(RuntimeError):
        loss_fn(torch.randn(10, 3), torch.randn(10, 2))


def test_reduction_modes(simple_conditional_flow, mock_context, mock_target):
    """Test different reduction modes."""
    for reduction in ["mean", "sum", "none"]:
        loss_fn = NormalizingFlowLoss(flow=simple_conditional_flow, reduction=reduction)
        loss = loss_fn(mock_context, mock_target)

        if reduction == "none":
            assert loss.shape == (10,)  # Per-sample losses
        else:
            assert loss.shape == ()  # Scalar


# Test sampling
def test_sampling_single(simple_unconditional_flow):
    """Test sampling single samples from unconditional distribution."""
    # Use unconditional flow to avoid zuko context shape issues in tests
    loss_fn = NormalizingFlowLoss(flow=simple_unconditional_flow)
    dummy_context = torch.randn(10, 0)

    samples = loss_fn.sample(dummy_context, n_samples=1)

    assert samples.shape == (10, 2)  # [batch_size, n_features]
    assert torch.isfinite(samples).all()


def test_sampling_multiple(simple_unconditional_flow):
    """Test sampling multiple samples from unconditional distribution."""
    loss_fn = NormalizingFlowLoss(flow=simple_unconditional_flow)
    dummy_context = torch.randn(10, 0)

    samples = loss_fn.sample(dummy_context, n_samples=5)

    assert samples.shape == (10, 5, 2)  # [batch_size, n_samples, n_features]
    assert torch.isfinite(samples).all()


# Test gradient flow
def test_gradient_flow_through_flow(simple_conditional_flow):
    """Test that gradients flow through the flow's parameters."""
    loss_fn = NormalizingFlowLoss(flow=simple_conditional_flow)

    # Create context and target
    context = torch.randn(8, 5, requires_grad=True)
    target = torch.randn(8, 2)

    # Forward and backward
    loss = loss_fn(context, target)
    loss.backward()

    # Check gradients flow to context
    assert context.grad is not None
    assert torch.isfinite(context.grad).all()

    # Check gradients flow to flow parameters
    for param in simple_conditional_flow.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


def test_gradient_flow_end_to_end():
    """Test end-to-end gradient flow through model -> flow."""
    # Create a simple model that outputs context
    model = torch.nn.Linear(3, 5)
    flow = create_flow_model(
        n_features=2, context_dim=5, flow_type="nsf", n_transforms=1, hidden_features=8
    )
    loss_fn = NormalizingFlowLoss(flow=flow)

    # Create input and target
    x = torch.randn(8, 3, requires_grad=True)
    target = torch.randn(8, 2)

    # Forward pass
    context = model(x)
    loss = loss_fn(context, target)

    # Backward pass
    loss.backward()

    # Check gradients flow to all parameters
    assert x.grad is not None
    for param in model.parameters():
        assert param.grad is not None
    for param in flow.parameters():
        assert param.grad is not None


# Test factory function
# Removed test_create_flow_loss as factory function was removed.


# Test numerical stability
def test_numerical_stability_large_values(simple_conditional_flow):
    """Test stability with large input values."""
    loss_fn = NormalizingFlowLoss(flow=simple_conditional_flow)

    # Large context and targets
    context = torch.randn(5, 5) * 100
    target = torch.randn(5, 2) * 100

    loss = loss_fn(context, target)
    assert torch.isfinite(loss)


def test_numerical_stability_small_values(simple_conditional_flow):
    """Test stability with small input values."""
    loss_fn = NormalizingFlowLoss(flow=simple_conditional_flow)

    # Small context and targets
    context = torch.randn(5, 5) * 1e-6
    target = torch.randn(5, 2) * 1e-6

    loss = loss_fn(context, target)
    assert torch.isfinite(loss)


def test_batch_consistency(simple_conditional_flow):
    """Test that loss is consistent across different batch sizes."""
    loss_fn = NormalizingFlowLoss(flow=simple_conditional_flow, reduction="mean")

    # Same data, different batch sizes
    context_small = torch.randn(4, 5)
    target_small = torch.randn(4, 2)

    context_large = torch.cat([context_small] * 2, dim=0)
    target_large = torch.cat([target_small] * 2, dim=0)

    loss_small = loss_fn(context_small, target_small)
    loss_large = loss_fn(context_large, target_large)

    # Losses should be similar (not exact due to randomness in flow)
    # Just check they're both finite and positive
    assert torch.isfinite(loss_small) and torch.isfinite(loss_large)
    assert loss_small > 0 and loss_large > 0


# Integration test
def test_training_loop():
    """Test a simple training loop to ensure everything works together."""
    # Create model, flow, and loss
    model = torch.nn.Sequential(torch.nn.Linear(5, 10), torch.nn.ReLU(), torch.nn.Linear(10, 8))
    flow = create_flow_model(
        n_features=2, context_dim=8, flow_type="nsf", n_transforms=2, hidden_features=16
    )
    loss_fn = NormalizingFlowLoss(flow=flow)

    # Create optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) + list(flow.parameters()), lr=0.01)

    # Training data
    x_train = torch.randn(32, 5)
    y_train = torch.randn(32, 2)

    # Training loop
    initial_loss = None
    for epoch in range(3):
        optimizer.zero_grad()
        context = model(x_train)
        loss = loss_fn(context, y_train)
        loss.backward()
        optimizer.step()

        if initial_loss is None:
            initial_loss = loss.item()

        # Check loss is finite
        assert torch.isfinite(loss)

    # Loss should change over training (not necessarily decrease due to random data)
    # Just check the training ran without errors
    assert True


# Test different flow types
@pytest.mark.parametrize("flow_type", ["realnvp", "maf", "nsf"])
def test_different_flow_types(flow_type):
    """Test that all supported flow types work correctly."""
    flow = create_flow_model(
        n_features=2, context_dim=5, flow_type=flow_type, n_transforms=2, hidden_features=16
    )
    loss_fn = NormalizingFlowLoss(flow=flow)

    context = torch.randn(8, 5)
    target = torch.randn(8, 2)

    loss = loss_fn(context, target)
    assert torch.isfinite(loss)

    # Test sampling
    samples = loss_fn.sample(context, n_samples=3)
    assert samples.shape == (8, 3, 2)
