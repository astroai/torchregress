"""Shared density API contracts for generic heads."""

from __future__ import annotations

import torch

from torchregress.losses import BinnedDensityLoss, JointBinnedDensityLoss, MDNLoss
from torchregress.prediction import PredictiveBatch


def test_mdn_log_prob_and_samples() -> None:
    mdn = MDNLoss(n_components=3, n_features=2)
    target = torch.randn(4, 2)
    out = torch.randn(4, mdn.expected_output_size)
    lp = mdn.log_prob(out, target)
    assert lp.shape == (4,) and torch.isfinite(lp).all()
    assert mdn.sample(out, n_samples=3).shape == (3, 4, 2)


def test_binned_log_prob_and_samples() -> None:
    binned = BinnedDensityLoss(torch.linspace(-2, 2, 9))
    logits = torch.randn(4, 8)
    target = torch.rand(4) * 2 - 1
    lp = binned.log_prob(logits, target)
    assert lp.shape == (4,) and torch.isfinite(lp).all()
    assert binned.sample(logits, n_samples=3).shape == (4, 3)


def test_joint_binned_log_prob_and_samples() -> None:
    joint = JointBinnedDensityLoss([torch.linspace(-2, 2, 5), torch.linspace(-2, 2, 5)])
    logits = torch.randn(4, joint.joint_bins)
    target = torch.randn(4, 2)
    assert joint.log_prob(logits, target).shape == (4,)
    assert joint.sample(logits, n_samples=3).shape == (4, 3, 2)


def test_predictive_batch_constructors() -> None:
    support = torch.linspace(0, 1, 10)
    density = torch.ones(3, 10) / 10.0
    batch = PredictiveBatch.from_density(support, density)
    assert batch.support is support and batch.density is density
    samples = torch.randn(3, 20)
    assert PredictiveBatch.from_samples(samples).samples is samples


def test_binned_log_prob_respects_finite_support():
    from torchregress.losses import BinnedDensityLoss

    loss = BinnedDensityLoss(bin_edges=torch.tensor([0.0, 1.0, 3.0]))
    logits = torch.zeros((5, 2), dtype=torch.double)
    values = torch.tensor([-1.0, 0.0, 0.5, 2.0, 4.0], dtype=torch.double)
    actual = loss.log_prob(logits, values)
    assert torch.isneginf(actual[[0, 4]]).all()
    torch.testing.assert_close(
        actual[1:4], torch.tensor([0.5, 0.5, 0.25], dtype=torch.double).log()
    )


def test_joint_binned_log_prob_respects_finite_support():
    loss = JointBinnedDensityLoss([torch.tensor([0.0, 1.0]), torch.tensor([-1.0, 1.0])])
    logits = torch.zeros((3, 1), dtype=torch.double)
    target = torch.tensor([[0.5, 0.0], [-0.1, 0.0], [0.5, 1.1]], dtype=torch.double)
    values = loss.log_prob(logits, target)
    torch.testing.assert_close(values[:1], torch.tensor([0.5], dtype=torch.double).log())
    assert torch.isneginf(values[1:]).all()
