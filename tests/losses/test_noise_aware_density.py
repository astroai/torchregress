"""Tests for noise-aware and binned conditional density heads."""

from __future__ import annotations

import pytest
import torch

from torchregress.losses import (
    BinnedDensityLoss,
    MixtureDensityLoss,
    NoiseAwareDensityLoss,
    NormalizingFlowLoss,
    UncertaintyAwareConR,
    create_flow_model,
)


def _flow_loss(n_features: int = 1, context_dim: int = 4) -> NormalizingFlowLoss:
    pytest.importorskip("zuko")
    flow = create_flow_model(
        n_features=n_features,
        context_dim=context_dim,
        flow_type="nsf",
        n_transforms=2,
        hidden_features=[16, 16],
        bound=8.0,
    )
    return NormalizingFlowLoss(flow=flow)


def test_binned_density_soft_and_hard() -> None:
    torch.manual_seed(0)
    edges = torch.linspace(-2.0, 2.0, 5)
    loss_fn = BinnedDensityLoss(edges)
    logits = torch.randn(16, 4, requires_grad=True)
    target = torch.randn(16)
    hard = loss_fn(logits, target)
    soft = loss_fn(logits, target, sigma=torch.full((16,), 0.2))
    assert hard.ndim == 0
    assert soft.ndim == 0
    assert torch.isfinite(hard) and torch.isfinite(soft)
    hard.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

    probs = loss_fn.bin_probabilities(target, sigma=0.2)
    assert probs.shape == (16, 4)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(16), atol=1e-5)
    assert loss_fn.n_bins == 4


def test_noise_aware_flow_zero_sigma_matches_base() -> None:
    torch.manual_seed(1)
    base = _flow_loss()
    loss_fn = NoiseAwareDensityLoss(base, mode="quadrature")
    context = torch.randn(12, 4)
    target = torch.randn(12, 1)
    base_loss = base(context, target)
    no_noise = loss_fn(context, target, sigma=torch.zeros(12))
    assert torch.allclose(base_loss, no_noise, atol=1e-4, rtol=1e-4)


def test_noise_aware_flow_gradient_and_contamination() -> None:
    torch.manual_seed(2)
    base = _flow_loss()
    loss_fn = NoiseAwareDensityLoss(base, contamination_fraction=0.05, contamination_scale=2.0)
    context = torch.randn(10, 4, requires_grad=True)
    target = torch.randn(10, 1)
    loss = loss_fn(context, target, sigma=torch.full((10,), 0.1))
    assert torch.isfinite(loss)
    loss.backward()
    assert context.grad is not None
    assert torch.isfinite(context.grad).all()


def test_noise_aware_mdn_analytic_matches_mc() -> None:
    torch.manual_seed(3)
    mdn = MixtureDensityLoss(n_components=2, n_features=1)
    y_pred = torch.randn(8, mdn.expected_output_size)
    target = torch.randn(8, 1)
    sigma = torch.full((8,), 0.2)
    analytic = NoiseAwareDensityLoss(mdn, mode="analytic")(y_pred, target, sigma=sigma)
    mc = NoiseAwareDensityLoss(mdn, mode="mc", n_samples=512)(y_pred, target, sigma=sigma)
    assert torch.allclose(analytic, mc, atol=2.0, rtol=0.5)


def test_bpf_and_cinn_flow_types() -> None:
    pytest.importorskip("zuko")
    context = torch.randn(6, 3)
    target = torch.randn(6, 1)
    bpf = create_flow_model(n_features=1, context_dim=3, flow_type="bpf", degree=8)
    cinn = create_flow_model(n_features=1, context_dim=3, flow_type="cinn", n_transforms=2)
    for flow in (bpf, cinn):
        loss = NormalizingFlowLoss(flow=flow)(context, target)
        assert torch.isfinite(loss)


def test_uncertainty_aware_conr() -> None:
    torch.manual_seed(4)
    features = torch.randn(24, 8, requires_grad=True)
    labels = torch.randn(24)
    predictions = torch.randn(24)
    loss_fn = UncertaintyAwareConR(label_sigma=0.05)
    loss = loss_fn(features, labels, prediction=predictions)
    assert torch.isfinite(loss)
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
        ),
    ],
)
def test_binned_density_preserves_boundary_precision_and_input_device(device):
    loss = BinnedDensityLoss([0.0, 1.0, 2.0])
    target = torch.tensor([1.0 - 1e-8, 1.0 + 1e-8], dtype=torch.float64, device=device)
    probabilities = loss.bin_probabilities(target)
    assert probabilities.device == target.device
    assert probabilities.dtype == torch.float64
    torch.testing.assert_close(probabilities, torch.eye(2, dtype=target.dtype, device=device))
    logits = torch.tensor(
        [[2.0, -1.0], [-1.0, 2.0]], dtype=torch.float64, device=device, requires_grad=True
    )
    actual = loss(logits, target)
    expected = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1], device=device))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()
    samples = loss.sample(logits.detach(), 5)
    assert samples.dtype == logits.dtype and samples.device == logits.device


def test_binned_density_rejects_nonfinite_unmasked_targets_but_allows_missing_mask():
    loss = BinnedDensityLoss([0.0, 1.0, 2.0])
    target = torch.tensor([0.5, float("nan")])
    logits = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="finite"):
        loss(logits, target)
    result = loss(logits, target, mask=torch.tensor([True, False]))
    assert torch.isfinite(result)


def test_binned_density_preserves_close_float64_edges():
    edges = torch.tensor([0.0, 1.0, 1.0 + 1e-8, 2.0], dtype=torch.float64)
    loss = BinnedDensityLoss(edges)
    torch.testing.assert_close(loss.bin_edges, edges, rtol=0, atol=0)
    probabilities = loss.bin_probabilities(torch.tensor([1.0 + 5e-9], dtype=torch.float64))
    torch.testing.assert_close(probabilities, torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64))
