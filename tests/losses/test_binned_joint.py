"""Tests for the joint multi-target binned density head."""

from __future__ import annotations

import pytest
import torch

from torchregress.losses.noise_aware_density import JointBinnedDensityLoss


def _loss() -> JointBinnedDensityLoss:
    return JointBinnedDensityLoss(
        [torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]), torch.tensor([-1.0, 0.0, 1.0])]
    )


def test_joint_bins_and_probabilities_normalize() -> None:
    loss = _loss()
    assert loss.joint_bins == 8
    target = torch.tensor([[0.2, -0.2], [1.2, 0.2], [-0.2, 0.2]])
    probs = loss.bin_probabilities(target)
    assert probs.shape == (3, 8)
    assert torch.allclose(probs.sum(-1), torch.ones(3))
    assert torch.all((probs == 0) | (probs == 1))


def test_joint_forward_matches_hard_cross_entropy() -> None:
    loss = _loss()
    target = torch.tensor([[0.2, -0.2]])
    logits = torch.zeros(1, 8)
    value = loss(logits, target)
    expected = torch.log(torch.tensor(8.0))
    assert torch.allclose(value, expected, atol=1e-6)


def test_joint_noise_soft_target_and_log_prob() -> None:
    loss = _loss()
    target = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    probs = loss.bin_probabilities(target, sigma=torch.tensor([0.3, 0.4]))
    assert probs.shape == (2, 8)
    assert torch.allclose(probs.sum(-1), torch.ones(2), atol=1e-6)
    assert torch.allclose(loss.bin_probabilities(target), loss.bin_probabilities(target))
    logp = loss.log_prob(torch.zeros(2, 8), target)
    assert logp.shape == (2,) and torch.isfinite(logp).all()


def test_joint_sample_and_cdf() -> None:
    torch.manual_seed(0)
    loss = _loss()
    logits = torch.zeros(4, 8)
    samples = loss.sample(logits, n_samples=64)
    assert samples.shape == (4, 64, 2)
    assert samples[..., 0].min() >= -2.0 and samples[..., 0].max() <= 2.0
    assert samples[..., 1].min() >= -1.0 and samples[..., 1].max() <= 1.0
    values = torch.tensor([[-3.0, -3.0], [-2.0, -1.0], [2.0, 1.0], [3.0, 3.0]])
    cdf = loss.cdf(logits, values)
    assert cdf.shape == (4,)
    assert torch.allclose(cdf[0], torch.tensor(0.0))
    assert torch.allclose(cdf[-1], torch.tensor(1.0), atol=1e-6)
    assert torch.all(cdf[1:] >= cdf[:-1])
    m = loss.marginal_cdf(logits, 0, values[:, 0])
    assert m.shape == (4,) and torch.all(m[1:] >= m[:-1])


def test_joint_mask_weights_and_reduction() -> None:
    loss = _loss()
    logits = torch.zeros(3, 8)
    target = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, -0.5]])
    mask = torch.tensor([[True, True], [False, True], [True, True]])
    out = JointBinnedDensityLoss(loss.bin_edges, reduction="none")(logits, target, mask=mask)
    assert out.shape == (3,)
    assert out[1].item() == 0.0
    weights = torch.tensor([1.0, 2.0, 0.0])
    plain = JointBinnedDensityLoss(loss.bin_edges, reduction="none")(logits, target)
    expected = (plain * weights).sum() / weights.sum()
    mean = loss(logits, target, weights=weights)
    assert torch.allclose(mean, expected, atol=1e-6)


def test_joint_validation() -> None:
    with pytest.raises(ValueError, match="at least two targets"):
        JointBinnedDensityLoss([torch.tensor([0.0, 1.0])])
    with pytest.raises(ValueError, match="strictly increasing"):
        JointBinnedDensityLoss([torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0])])
    with pytest.raises(ValueError, match="max_joint_bins"):
        JointBinnedDensityLoss([torch.linspace(0, 1, 100)] * 3, max_joint_bins=4096)
    loss = _loss()
    with pytest.raises(ValueError, match="target must be"):
        loss(torch.zeros(1, 8), torch.zeros(1, 1))
    with pytest.raises(ValueError, match="y_pred must be"):
        loss(torch.zeros(1, 3), torch.zeros(1, 2))
