import torch

from torchregress.ensemble.bnn import (
    BayesianNeuralNetwork,
    HeteroscedasticBNN,
    VariationalLinear,
)


class TestVariationalLinear:
    def test_initialization(self):
        layer = VariationalLinear(in_features=10, out_features=5, prior_sigma=0.5, bias=True)
        assert layer.in_features == 10
        assert layer.out_features == 5
        assert layer.prior_sigma == 0.5
        assert layer.use_bias is True

        assert layer.weight_mu.shape == (5, 10)
        assert layer.weight_log_sigma.shape == (5, 10)
        assert layer.bias_mu.shape == (5,)
        assert layer.bias_log_sigma.shape == (5,)

        layer_no_bias = VariationalLinear(in_features=10, out_features=5, bias=False)
        assert layer_no_bias.use_bias is False
        assert getattr(layer_no_bias, "bias_mu", None) is None
        assert getattr(layer_no_bias, "bias_log_sigma", None) is None

    def test_forward_shape_and_variance(self):
        layer = VariationalLinear(in_features=10, out_features=5)
        x = torch.randn(8, 10)

        out1 = layer(x)
        out2 = layer(x)

        assert out1.shape == (8, 5)
        assert out2.shape == (8, 5)

        # Variational layer should have different outputs for the same input due to sampling
        assert not torch.allclose(out1, out2)

    def test_kl_divergence(self):
        layer = VariationalLinear(in_features=10, out_features=5)
        kl = layer.kl_divergence()

        assert kl.dim() == 0  # scalar
        assert torch.isfinite(kl)
        assert kl.item() >= 0  # KL divergence is non-negative

        layer_no_bias = VariationalLinear(in_features=10, out_features=5, bias=False)
        kl_no_bias = layer_no_bias.kl_divergence()
        assert kl_no_bias.dim() == 0
        assert torch.isfinite(kl_no_bias)
        assert kl_no_bias.item() >= 0


class TestBayesianNeuralNetwork:
    def test_initialization(self):
        model = BayesianNeuralNetwork(
            input_dim=10, hidden_dims=[16, 8], output_dim=1, prior_sigma=1.0, n_samples=20
        )
        assert model.n_samples == 20
        assert model.prior_sigma == 1.0

        # 3 layers total: [10 -> 16], [16 -> 8], [8 -> 1]
        assert len(model.variational_layers) == 3

    def test_forward_and_kl(self):
        model = BayesianNeuralNetwork(input_dim=10, hidden_dims=[16], output_dim=1)
        x = torch.randn(4, 10)

        out = model(x)
        assert out.shape == (4, 1)

        kl = model.kl_divergence()
        assert kl.dim() == 0
        assert torch.isfinite(kl)
        assert kl.item() >= 0

    def test_elbo_loss(self):
        model = BayesianNeuralNetwork(input_dim=10, hidden_dims=[16], output_dim=1)
        x = torch.randn(4, 10)
        y = torch.randn(4, 1)

        pred = model(x)
        loss = model.elbo_loss(pred, y, n_data=100, beta=1.0)

        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_inference_methods(self):
        model = BayesianNeuralNetwork(input_dim=10, hidden_dims=[16], output_dim=1, n_samples=5)
        x = torch.randn(4, 10)

        # mc_forward
        samples = model.mc_forward(x)
        assert samples.shape == (5, 4, 1)

        # predict_with_uncertainty
        mean, std = model.predict_with_uncertainty(x)
        assert mean.shape == (4, 1)
        assert std.shape == (4, 1)
        assert torch.all(std >= 0)

        # predict_interval
        lower, upper = model.predict_interval(x, confidence=0.95)
        assert lower.shape == (4, 1)
        assert upper.shape == (4, 1)
        assert torch.all(lower <= upper)


class TestHeteroscedasticBNN:
    def test_initialization(self):
        model = HeteroscedasticBNN(
            input_dim=10, hidden_dims=[16, 8], output_dim=2, prior_sigma=1.0, n_samples=20
        )
        assert model.n_samples == 20
        assert model.output_dim == 2
        assert len(model.variational_layers) == 3

        # Final layer should output 2 * output_dim (mean and log_var)
        assert model.variational_layers[-1].out_features == 4

    def test_forward_and_kl(self):
        model = HeteroscedasticBNN(input_dim=10, hidden_dims=[16], output_dim=2)
        x = torch.randn(4, 10)

        mean, log_var = model(x)
        assert mean.shape == (4, 2)
        assert log_var.shape == (4, 2)

        kl = model.kl_divergence()
        assert kl.dim() == 0
        assert torch.isfinite(kl)
        assert kl.item() >= 0

    def test_inference_methods(self):
        model = HeteroscedasticBNN(input_dim=10, hidden_dims=[16], output_dim=2, n_samples=5)
        x = torch.randn(4, 10)

        # mc_forward
        means, log_vars = model.mc_forward(x)
        assert means.shape == (5, 4, 2)
        assert log_vars.shape == (5, 4, 2)

        # predict_with_decomposition
        torch.manual_seed(42)
        mean, aleatoric, epistemic = model.predict_with_decomposition(x)
        assert mean.shape == (4, 2)
        assert aleatoric.shape == (4, 2)
        assert epistemic.shape == (4, 2)
        assert torch.all(aleatoric >= 0)
        assert torch.all(epistemic >= 0)

        # predict_with_uncertainty
        torch.manual_seed(42)
        mean2, total_std = model.predict_with_uncertainty(x)
        assert mean2.shape == (4, 2)
        assert total_std.shape == (4, 2)
        assert torch.all(total_std >= 0)
        assert torch.allclose(mean, mean2)

        # Check total_std calculation
        expected_total_std = torch.sqrt(aleatoric + epistemic)
        assert torch.allclose(total_std, expected_total_std)

        # predict_interval
        lower, upper = model.predict_interval(x, confidence=0.95)
        assert lower.shape == (4, 2)
        assert upper.shape == (4, 2)
        assert torch.all(lower <= upper)
