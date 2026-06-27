import torch

from torchregress.algorithms import ErrorAwareFeatureEncoder, NoiseAwareRegressor


def test_error_aware_feature_encoder_shapes_and_finiteness():
    torch.manual_seed(0)
    encoder = ErrorAwareFeatureEncoder(input_dim=4, hidden_dim=12)
    x = torch.randn(10, 4)
    sigma = torch.full_like(x, 0.2)
    encoded = encoder(x, sigma)
    assert encoded.shape == (10, 12)
    assert torch.isfinite(encoded).all()


def test_error_aware_encoder_downweights_higher_noise():
    torch.manual_seed(1)
    encoder = ErrorAwareFeatureEncoder(input_dim=2, hidden_dim=8)
    x = torch.ones(4, 2)
    sigma_low = torch.full_like(x, 0.1)
    sigma_high = torch.full_like(x, 10.0)
    low_gate = encoder.quality_gate(x, sigma_low)
    high_gate = encoder.quality_gate(x, sigma_high)
    assert float(low_gate.mean()) > float(high_gate.mean())


def test_noise_aware_regressor_forward():
    torch.manual_seed(2)
    model = NoiseAwareRegressor(3, 2, encoder_hidden_dim=8, backbone_hidden_dims=(8,))
    x = torch.randn(6, 3)
    sigma = torch.full_like(x, 0.3)
    out = model(x, sigma)
    assert out.shape == (6, 2)
    assert torch.isfinite(out).all()
