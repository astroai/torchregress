import torch
from torch import optim

from torchregress.losses.poisson_gaussian import (
    EnhancedPoissonGaussianConfig,
    EnhancedPoissonGaussianMixtureLoss,
    PoissonGaussianLikelihoodRatioConfig,
    PoissonGaussianLikelihoodRatioLoss,
    PoissonGaussianMixtureLoss,
)


class SimpleModel(torch.nn.Module):
    """Simple model for testing gradient flow through the loss functions"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class TestPoissonGaussianMixtureLoss:
    """Tests for PoissonGaussianMixtureLoss"""

    def test_init_default_parameters(self):
        """Test default parameter initialization"""
        loss_fn = PoissonGaussianMixtureLoss()
        assert loss_fn.eps == 1e-8
        assert not loss_fn.learn_variance
        assert loss_fn.initial_variance == 1.0
        assert loss_fn.min_variance == 1e-6
        assert not loss_fn.log_input
        assert loss_fn.mixture_weights is None
        assert not loss_fn.extra_variance_model

    def test_forward_basic(self):
        """Test basic forward pass with simple inputs"""
        loss_fn = PoissonGaussianMixtureLoss()
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        loss = loss_fn(y_pred, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar output
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_forward_with_log_input(self):
        """Test forward pass with log input option"""
        loss_fn = PoissonGaussianMixtureLoss(log_input=True)
        y_pred = torch.log(torch.tensor([[10.0, 20.0], [5.0, 15.0]]))
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        loss = loss_fn(y_pred, target)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_learnable_variance(self):
        """Test with learnable variance parameter"""
        loss_fn = PoissonGaussianMixtureLoss(learn_variance=True, initial_variance=0.5)
        assert hasattr(loss_fn, "log_variance")
        assert isinstance(loss_fn.log_variance, torch.nn.Parameter)

        # Check if variance is correctly initialized
        variance = torch.exp(loss_fn.log_variance)
        assert torch.allclose(variance, torch.tensor([0.5]))

        # Test forward with learnable variance
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        loss = loss_fn(y_pred, target)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_fixed_mixture_weights(self):
        """Test with fixed mixture weights"""
        poisson_weight = 0.7
        loss_fn = PoissonGaussianMixtureLoss(mixture_weights=poisson_weight)
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        loss = loss_fn(y_pred, target)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_learnable_mixture_weights(self):
        """Test with learnable mixture weights"""
        loss_fn = PoissonGaussianMixtureLoss(mixture_weights="learn")
        assert hasattr(loss_fn, "weight_logit")
        assert isinstance(loss_fn.weight_logit, torch.nn.Parameter)

        # Test forward with learnable mixture weights
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        loss = loss_fn(y_pred, target)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_mask(self):
        """Test forward pass with mask"""
        loss_fn = PoissonGaussianMixtureLoss()
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        mask = torch.tensor([[1, 1], [0, 1]], dtype=torch.bool)
        loss = loss_fn(y_pred, target, mask=mask)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_weights(self):
        """Test forward pass with sample weights"""
        loss_fn = PoissonGaussianMixtureLoss()
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        weights = torch.tensor([[0.5, 1.0], [0.8, 0.3]])
        loss = loss_fn(y_pred, target, weights=weights)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_extra_variance(self):
        """Test forward pass with extra variance parameter"""
        loss_fn = PoissonGaussianMixtureLoss(extra_variance_model=True)
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        extra_var = torch.tensor([[0.1, 0.2], [0.3, 0.1]])
        loss = loss_fn(y_pred, target, extra_var=extra_var)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_all_features(self):
        """Test with all features enabled"""
        loss_fn = PoissonGaussianMixtureLoss(
            learn_variance=True,
            initial_variance=0.2,
            log_input=True,
            mixture_weights="learn",
            extra_variance_model=True,
        )

        y_pred = torch.log(torch.tensor([[10.0, 20.0], [5.0, 15.0]]))
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        mask = torch.tensor([[1, 1], [0, 1]], dtype=torch.bool)
        weights = torch.tensor([[0.5, 1.0], [0.8, 0.3]])
        extra_var = torch.tensor([[0.1, 0.2], [0.3, 0.1]])

        loss = loss_fn(y_pred, target, mask=mask, weights=weights, extra_var=extra_var)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_factory_function(self):
        """Test direct construction of PoissonGaussianMixtureLoss"""
        loss_fn = PoissonGaussianMixtureLoss(
            learn_variance=True,
            initial_variance=0.5,
            log_input=True,
            mixture_weights="learn",
            eps=1e-6,
        )

        assert isinstance(loss_fn, PoissonGaussianMixtureLoss)
        assert loss_fn.learn_variance
        assert loss_fn.initial_variance == 0.5
        assert loss_fn.log_input
        assert loss_fn.mixture_weights == "learn"
        assert loss_fn.eps == 1e-6

    def test_gradients(self):
        """Test that gradients flow correctly through PoissonGaussianMixtureLoss"""
        torch.manual_seed(42)

        # Create a simple model
        model = SimpleModel(5, 2)

        # Create loss function with learnable parameters
        loss_fn = PoissonGaussianMixtureLoss(learn_variance=True, mixture_weights="learn")

        # Optimizer includes both model and loss function parameters
        optimizer = optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=0.01)

        # Create inputs and targets
        x = torch.randn(10, 5)
        targets = torch.rand(10, 2) * 20  # Random values between 0 and 20

        # Store initial parameter values
        initial_weight = model.linear.weight.clone().detach()
        initial_bias = model.linear.bias.clone().detach()
        initial_variance = loss_fn.log_variance.clone().detach()
        initial_weight_logit = loss_fn.weight_logit.clone().detach()

        # Training loop
        for _ in range(5):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, targets)
            loss.backward()
            optimizer.step()

        # Check that parameters have changed
        assert not torch.allclose(model.linear.weight, initial_weight)
        assert not torch.allclose(model.linear.bias, initial_bias)
        assert not torch.allclose(loss_fn.log_variance, initial_variance)
        assert not torch.allclose(loss_fn.weight_logit, initial_weight_logit)


class TestEnhancedPoissonGaussianMixtureLoss:
    """Tests for EnhancedPoissonGaussianMixtureLoss"""

    def test_init_default_parameters(self):
        """Test default parameter initialization"""
        loss_fn = EnhancedPoissonGaussianMixtureLoss()
        assert not loss_fn.log_input
        assert not loss_fn.calibration
        assert loss_fn.eps == 1e-8
        assert not loss_fn.learn_gain
        assert not loss_fn.learn_offset
        assert not loss_fn.learn_read_noise
        assert not loss_fn.learn_shot_noise

    def test_forward_basic(self):
        """Test basic forward pass with simple inputs"""
        loss_fn = EnhancedPoissonGaussianMixtureLoss()
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        loss = loss_fn(y_pred, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar output
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_forward_with_log_input(self):
        """Test forward pass with log input option"""
        loss_fn = EnhancedPoissonGaussianMixtureLoss(log_input=True)
        y_pred = torch.log(torch.tensor([[10.0, 20.0], [5.0, 15.0]]))
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        loss = loss_fn(y_pred, target)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_learnable_gain(self):
        """Test with learnable gain parameter"""
        loss_fn = EnhancedPoissonGaussianMixtureLoss(gain="learn")
        assert hasattr(loss_fn, "log_gain")
        assert isinstance(loss_fn.log_gain, torch.nn.Parameter)

        # Test forward with learnable gain
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        loss = loss_fn(y_pred, target)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_learnable_offset(self):
        """Test with learnable offset parameter"""
        loss_fn = EnhancedPoissonGaussianMixtureLoss(offset="learn")
        assert hasattr(loss_fn, "offset")
        assert isinstance(loss_fn.offset, torch.nn.Parameter)

        # Test forward with learnable offset
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        loss = loss_fn(y_pred, target)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_learnable_noise_params(self):
        """Test with learnable noise parameters"""
        loss_fn = EnhancedPoissonGaussianMixtureLoss(read_noise="learn", shot_noise="learn")
        assert hasattr(loss_fn, "log_read_noise")
        assert isinstance(loss_fn.log_read_noise, torch.nn.Parameter)
        assert hasattr(loss_fn, "log_shot_noise")
        assert isinstance(loss_fn.log_shot_noise, torch.nn.Parameter)

        # Test forward with learnable noise parameters
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        loss = loss_fn(y_pred, target)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_calibration(self):
        """Test with calibration parameters"""
        loss_fn = EnhancedPoissonGaussianMixtureLoss(calibration=True)
        assert hasattr(loss_fn, "calib_add")
        assert isinstance(loss_fn.calib_add, torch.nn.Parameter)
        assert hasattr(loss_fn, "calib_mult")
        assert isinstance(loss_fn.calib_mult, torch.nn.Parameter)

        # Test forward with calibration
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        loss = loss_fn(y_pred, target)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_mask(self):
        """Test forward pass with mask"""
        loss_fn = EnhancedPoissonGaussianMixtureLoss()
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        mask = torch.tensor([[1, 1], [0, 1]], dtype=torch.bool)
        loss = loss_fn(y_pred, target, mask=mask)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_weights(self):
        """Test forward pass with sample weights"""
        loss_fn = EnhancedPoissonGaussianMixtureLoss()
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        weights = torch.tensor([[0.5, 1.0], [0.8, 0.3]])
        loss = loss_fn(y_pred, target, weights=weights)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_all_features(self):
        """Test with all features enabled"""
        loss_fn = EnhancedPoissonGaussianMixtureLoss(
            gain="learn",
            offset="learn",
            read_noise="learn",
            shot_noise="learn",
            log_input=True,
            calibration=True,
        )

        y_pred = torch.log(torch.tensor([[10.0, 20.0], [5.0, 15.0]]))
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        mask = torch.tensor([[1, 1], [0, 1]], dtype=torch.bool)
        weights = torch.tensor([[0.5, 1.0], [0.8, 0.3]])

        loss = loss_fn(y_pred, target, mask=mask, weights=weights)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_factory_function(self):
        """Test the factory function for EnhancedPoissonGaussianMixtureLoss"""
        config = EnhancedPoissonGaussianConfig(
            gain="learn",
            offset=0.1,
            read_noise="learn",
            shot_noise=0.05,
            log_input=True,
            calibration=True,
        )
        loss_fn = EnhancedPoissonGaussianMixtureLoss(config=config)

        assert isinstance(loss_fn, EnhancedPoissonGaussianMixtureLoss)
        assert loss_fn.learn_gain
        assert not loss_fn.learn_offset
        assert loss_fn.learn_read_noise
        assert not loss_fn.learn_shot_noise
        assert loss_fn.log_input
        assert loss_fn.calibration

    def test_gradients(self):
        """Test that gradients flow correctly through EnhancedPoissonGaussianMixtureLoss"""
        torch.manual_seed(42)

        # Create a simple model
        model = SimpleModel(5, 2)

        # Create loss function with all learnable parameters
        loss_fn = EnhancedPoissonGaussianMixtureLoss(
            gain="learn", offset="learn", read_noise="learn", shot_noise="learn", calibration=True
        )

        # Combine all parameters
        params = list(model.parameters()) + list(loss_fn.parameters())
        optimizer = optim.Adam(params, lr=0.01)

        # Create inputs and targets
        x = torch.randn(10, 5)
        targets = torch.rand(10, 2) * 20  # Random values between 0 and 20

        # Store initial parameter values
        initial_weight = model.linear.weight.clone().detach()
        initial_bias = model.linear.bias.clone().detach()
        initial_log_gain = loss_fn.log_gain.clone().detach()
        initial_offset = loss_fn.offset.clone().detach()
        initial_log_read_noise = loss_fn.log_read_noise.clone().detach()
        initial_log_shot_noise = loss_fn.log_shot_noise.clone().detach()
        initial_calib_add = loss_fn.calib_add.clone().detach()
        initial_calib_mult = loss_fn.calib_mult.clone().detach()

        # Training loop
        for _ in range(5):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, targets)
            loss.backward()
            optimizer.step()

        # Check that parameters have changed
        assert not torch.allclose(model.linear.weight, initial_weight)
        assert not torch.allclose(model.linear.bias, initial_bias)
        assert not torch.allclose(loss_fn.log_gain, initial_log_gain)
        assert not torch.allclose(loss_fn.offset, initial_offset)
        assert not torch.allclose(loss_fn.log_read_noise, initial_log_read_noise)
        assert not torch.allclose(loss_fn.log_shot_noise, initial_log_shot_noise)
        assert not torch.allclose(loss_fn.calib_add, initial_calib_add)
        assert not torch.allclose(loss_fn.calib_mult, initial_calib_mult)


class TestPoissonGaussianLikelihoodRatioLoss:
    """Tests for PoissonGaussianLikelihoodRatioLoss"""

    def test_init_default_parameters(self):
        """Test default parameter initialization"""
        loss_fn = PoissonGaussianLikelihoodRatioLoss()
        assert loss_fn.eps == 1e-8
        assert loss_fn.log_input
        assert not loss_fn.learn_variance
        assert loss_fn.initial_variance == 1.0

    def test_forward_basic(self):
        """Test basic forward pass with simple inputs"""
        loss_fn = PoissonGaussianLikelihoodRatioLoss()
        y_pred = torch.log(torch.tensor([[10.0, 20.0], [5.0, 15.0]]))
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        loss = loss_fn(y_pred, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar output
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_forward_without_log_input(self):
        """Test forward pass without log input option"""
        loss_fn = PoissonGaussianLikelihoodRatioLoss(log_input=False)
        y_pred = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        loss = loss_fn(y_pred, target)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_learnable_variance(self):
        """Test with learnable variance parameter"""
        loss_fn = PoissonGaussianLikelihoodRatioLoss(learn_variance=True, initial_variance=0.5)
        assert hasattr(loss_fn, "log_variance")
        assert isinstance(loss_fn.log_variance, torch.nn.Parameter)

        # Check if variance is correctly initialized
        variance = torch.exp(loss_fn.log_variance)
        assert torch.allclose(variance, torch.tensor([0.5]))

        # Test forward with learnable variance
        y_pred = torch.log(torch.tensor([[10.0, 20.0], [5.0, 15.0]]))
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        loss = loss_fn(y_pred, target)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_mask(self):
        """Test forward pass with mask"""
        loss_fn = PoissonGaussianLikelihoodRatioLoss()
        y_pred = torch.log(torch.tensor([[10.0, 20.0], [5.0, 15.0]]))
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        mask = torch.tensor([[1, 1], [0, 1]], dtype=torch.bool)
        loss = loss_fn(y_pred, target, mask=mask)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_with_weights(self):
        """Test forward pass with sample weights"""
        loss_fn = PoissonGaussianLikelihoodRatioLoss()
        y_pred = torch.log(torch.tensor([[10.0, 20.0], [5.0, 15.0]]))
        target = torch.tensor([[11.0, 19.0], [6.0, 14.0]])
        weights = torch.tensor([[0.5, 1.0], [0.8, 0.3]])
        loss = loss_fn(y_pred, target, weights=weights)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_gradients(self):
        """Test that gradients flow correctly through PoissonGaussianLikelihoodRatioLoss"""
        torch.manual_seed(42)

        # Create a simple model
        model = SimpleModel(5, 2)

        # Create loss function with learnable variance
        loss_fn = PoissonGaussianLikelihoodRatioLoss(learn_variance=True)

        # Combine all parameters
        params = list(model.parameters()) + list(loss_fn.parameters())
        optimizer = optim.Adam(params, lr=0.01)

        # Create inputs and targets
        x = torch.randn(10, 5)
        targets = torch.rand(10, 2) * 20  # Random values between 0 and 20

        # Store initial parameter values
        initial_weight = model.linear.weight.clone().detach()
        initial_bias = model.linear.bias.clone().detach()
        initial_variance = loss_fn.log_variance.clone().detach()

        # Training loop
        for _ in range(5):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, targets)
            loss.backward()
            optimizer.step()

        # Check that parameters have changed
        assert not torch.allclose(model.linear.weight, initial_weight)
        assert not torch.allclose(model.linear.bias, initial_bias)
        assert not torch.allclose(loss_fn.log_variance, initial_variance)

    def test_factory_function(self):
        """Test direct construction"""
        config = PoissonGaussianLikelihoodRatioConfig(
            log_input=False, learn_variance=True, initial_variance=0.5
        )
        loss_fn = PoissonGaussianLikelihoodRatioLoss(config=config)

        assert isinstance(loss_fn, PoissonGaussianLikelihoodRatioLoss)
        assert not loss_fn.log_input
        assert loss_fn.learn_variance
        assert loss_fn.initial_variance == 0.5


def test_numerical_stability():
    """Test numerical stability with extreme values"""
    # Test PoissonGaussianMixtureLoss
    loss_fn1 = PoissonGaussianMixtureLoss()

    # Test with very small values
    y_pred1 = torch.tensor([[1e-10, 1e-8], [1e-9, 1e-7]])
    target1 = torch.tensor([[0.0, 1e-8], [1e-9, 1e-7]])
    loss1 = loss_fn1(y_pred1, target1)
    assert not torch.isnan(loss1)

    # Test EnhancedPoissonGaussianMixtureLoss
    loss_fn2 = EnhancedPoissonGaussianMixtureLoss()

    # Test with very small values
    y_pred2 = torch.tensor([[1e-10, 1e-8], [1e-9, 1e-7]])
    target2 = torch.tensor([[0.0, 1e-8], [1e-9, 1e-7]])
    loss2 = loss_fn2(y_pred2, target2)
    assert not torch.isnan(loss2)

    # Test PoissonGaussianLikelihoodRatioLoss
    loss_fn3 = PoissonGaussianLikelihoodRatioLoss(log_input=False)

    # Test with very small values
    y_pred3 = torch.tensor([[1e-10, 1e-8], [1e-9, 1e-7]])
    target3 = torch.tensor([[0.0, 1e-8], [1e-9, 1e-7]])
    loss3 = loss_fn3(y_pred3, target3)
    assert not torch.isnan(loss3)
