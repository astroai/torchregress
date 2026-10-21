"""Tests for the tabular Normalizing Flow Map."""

from __future__ import annotations

import pytest
import torch

from torchregress.models import NormalizingFlowMap


def test_nfm_roundtrip_and_likelihood() -> None:
    torch.manual_seed(0)
    for d in (1, 2, 3):
        model = NormalizingFlowMap(d, 8, n_layers=3, hidden=(16, 16))
        context = torch.randn(6, 8)
        y = torch.randn(6, d)
        z, logdet = model(y, 0.0, 1.0, context)
        rec, inv_logdet = model.inverse(z, 0.0, 1.0, context)
        assert torch.allclose(rec, y, atol=1e-5)
        assert torch.allclose(logdet, -inv_logdet, atol=1e-5)
        logp = model.log_prob(context, y)
        assert logp.shape == (6,) and torch.isfinite(logp).all()
        assert model.sample(context, n_samples=8).shape == (6, 8, d)


def test_nfm_training_loss_gradients() -> None:
    torch.manual_seed(0)
    model = NormalizingFlowMap(2, 8, n_layers=2, hidden=(16, 16))
    context = torch.randn(10, 8)
    y = torch.randn(10, 2)
    loss = model.training_loss(context, y)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_nfm_flow_map_consistency() -> None:
    torch.manual_seed(0)
    model = NormalizingFlowMap(2, 4, n_layers=2, hidden=(16, 16))
    context = torch.randn(5, 4)
    y = torch.randn(5, 2)
    z, _ = model(y, 0.0, 1.0, context)
    # Intermediate maps should compose consistently for a no-op identity-like
    # map; here just check shape/finiteness and invertibility at arbitrary time.
    mid, _ = model(y, 0.2, 0.8, context)
    back, _ = model.inverse(mid, 0.2, 0.8, context)
    assert torch.allclose(back, y, atol=1e-5)


def test_nfm_endpoint_density_matches_noise_to_target_map() -> None:
    """The interpolant and likelihood must use the same noise-to-data map."""
    import math

    model = NormalizingFlowMap(1, 1, n_layers=1, hidden=())
    scale, shift = 1.2, 2.5
    with torch.no_grad():
        model.layers[0].net[-1].bias.copy_(torch.tensor([math.atanh(2 * math.log(scale)), shift]))
    context = torch.zeros(3, 1)
    target = torch.tensor([[1.0], [2.5], [4.0]])
    expected = torch.distributions.Normal(shift, scale).log_prob(target).sum(-1)
    torch.testing.assert_close(model.log_prob(context, target), expected)


def test_nfm_samples_use_trained_endpoint_map() -> None:
    model = NormalizingFlowMap(1, 1, n_layers=1, hidden=())
    with torch.no_grad():
        model.layers[0].net[-1].bias[1] = 2.5
    context = torch.zeros(2, 1)
    torch.manual_seed(11)
    expected = torch.randn(2, 4, 1) + 2.5
    torch.manual_seed(11)
    torch.testing.assert_close(model.sample(context, n_samples=4), expected)


def test_nfm_scalar_time_broadcast_preserves_other_batched_time() -> None:
    model = NormalizingFlowMap(1, 1, n_layers=1, hidden=())
    with torch.no_grad():
        # First t sine feature controls shift; s is independently scalar.
        model.layers[0].net[-1].weight[1, 16] = 1.0
    x = torch.zeros(3, 1)
    context = torch.zeros(3, 1)
    times = torch.tensor([0.1, 0.4, 0.7])
    batched, _ = model(x, torch.tensor(0.0), times, context)
    expected = torch.cat(
        [model(x[i : i + 1], 0.0, times[i], context[i : i + 1])[0] for i in range(3)]
    )
    torch.testing.assert_close(batched, expected)


@pytest.mark.parametrize("arch", ["coupling", "tarflow"])
def test_nfm_nonidentity_roundtrip_and_exact_jacobian(arch: str) -> None:
    torch.manual_seed(12)
    model = NormalizingFlowMap(3, 2, n_layers=2, hidden=(8,), arch=arch).double()
    with torch.no_grad():
        for layer in model.layers:
            head = layer.head if arch == "tarflow" else layer.net[-1]
            head.weight.normal_(std=0.1)
            head.bias.normal_(std=0.1)
    x = torch.randn(1, 3, dtype=torch.double)
    context = torch.randn(1, 2, dtype=torch.double)
    y, logdet = model(x, 0.2, 0.8, context)
    recovered, inverse_logdet = model.inverse(y, 0.2, 0.8, context)
    torch.testing.assert_close(recovered, x)
    torch.testing.assert_close(inverse_logdet, -logdet)
    jacobian = torch.autograd.functional.jacobian(
        lambda value: model(value, 0.2, 0.8, context)[0], x
    )[0, :, 0, :]
    torch.testing.assert_close(torch.linalg.slogdet(jacobian).logabsdet, logdet[0])


@pytest.mark.parametrize("arch", ["coupling", "tarflow"])
@pytest.mark.parametrize("scalar_s", [False, True])
def test_nfm_mixed_scalar_and_batched_times(arch: str, scalar_s: bool) -> None:
    torch.manual_seed(13)
    model = NormalizingFlowMap(2, 1, n_layers=1, hidden=(8,), arch=arch)
    with torch.no_grad():
        layer = model.layers[0]
        head = layer.head if arch == "tarflow" else layer.net[-1]
        head.weight.normal_(std=0.1)
    x = torch.randn(3, 2)
    context = torch.randn(3, 1)
    times = torch.tensor([0.1, 0.4, 0.7])
    scalar = torch.tensor(0.25)
    s, t = (scalar, times) if scalar_s else (times, scalar)
    batched, _ = model(x, s, t, context)
    expected = torch.cat(
        [
            model(
                x[i : i + 1],
                scalar if scalar_s else times[i],
                times[i] if scalar_s else scalar,
                context[i : i + 1],
            )[0]
            for i in range(3)
        ]
    )
    torch.testing.assert_close(batched, expected)
