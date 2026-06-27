"""
Unit tests for torchregress.utils.augment.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchregress.utils.augment import (
    Adversarial,
    Augmentation,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Augmentation base class
# ═══════════════════════════════════════════════════════════════════════════════


class DummyAugmentation(Augmentation):
    """Concrete augmentation for testing the base class."""

    def augment(
        self, x: torch.Tensor, y: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return x * 2, y


class TestAugmentationBase:
    def test_init_defaults(self) -> None:
        """Default probability is 0.5."""
        aug = DummyAugmentation()
        assert aug.probability == 0.5

    def test_init_custom_probability(self) -> None:
        """Custom probability is stored."""
        aug = DummyAugmentation(probability=0.8)
        assert aug.probability == 0.8

    def test_init_invalid_probability_high(self) -> None:
        """Probability > 1 raises ValueError."""
        with pytest.raises(ValueError, match="Probability must be between 0 and 1"):
            DummyAugmentation(probability=1.5)

    def test_init_invalid_probability_low(self) -> None:
        """Probability < 0 raises ValueError."""
        with pytest.raises(ValueError, match="Probability must be between 0 and 1"):
            DummyAugmentation(probability=-0.1)

    def test_forward_always_applies_when_probability_1(self) -> None:
        """When probability=1.0, augment is always called."""
        aug = DummyAugmentation(probability=1.0)
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([10.0, 20.0, 30.0])
        x_out, y_out = aug(x, y)
        assert torch.equal(x_out, x * 2)
        assert torch.equal(y_out, y)  # type: ignore[arg-type]

    def test_forward_never_applies_when_probability_0(self) -> None:
        """When probability=0.0, original data is returned."""
        aug = DummyAugmentation(probability=0.0)
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([10.0, 20.0, 30.0])
        x_out, y_out = aug(x, y)
        assert torch.equal(x_out, x)
        assert torch.equal(y_out, y)  # type: ignore[arg-type]

    def test_forward_without_y(self) -> None:
        """Forward works when y is None."""
        aug = DummyAugmentation(probability=1.0)
        x = torch.tensor([1.0, 2.0, 3.0])
        x_out, y_out = aug(x)
        assert torch.equal(x_out, x * 2)
        assert y_out is None

    def test_augment_not_implemented(self) -> None:
        """Base class augment raises NotImplementedError."""
        base = Augmentation()
        with pytest.raises(NotImplementedError, match="Subclasses must implement"):
            base.augment(torch.tensor([1.0]), None)  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════════
# Adversarial
# ═══════════════════════════════════════════════════════════════════════════════


class TestAdversarial:
    def test_init_defaults(self) -> None:
        """Defaults: epsilon=0.05, steps=3, alpha=epsilon/steps."""
        model = nn.Linear(3, 1)
        loss_fn = nn.MSELoss()
        adv = Adversarial(model=model, loss_fn=loss_fn)
        assert adv.epsilon == 0.05
        assert adv.steps == 3
        assert adv.alpha == pytest.approx(0.05 / 3.0)
        assert adv.random_start is False

    def test_init_custom_alpha(self) -> None:
        """Custom alpha overrides epsilon/steps default."""
        model = nn.Linear(3, 1)
        loss_fn = nn.MSELoss()
        adv = Adversarial(model=model, loss_fn=loss_fn, epsilon=0.1, steps=2, alpha=0.01)
        assert adv.alpha == 0.01

    def test_init_random_start(self) -> None:
        """random_start=True is stored."""
        model = nn.Linear(3, 1)
        loss_fn = nn.MSELoss()
        adv = Adversarial(model=model, loss_fn=loss_fn, random_start=True)
        assert adv.random_start is True

    def test_init_invalid_epsilon(self) -> None:
        """Negative epsilon raises ValueError."""
        model = nn.Linear(3, 1)
        loss_fn = nn.MSELoss()
        with pytest.raises(ValueError, match="epsilon must be non-negative"):
            Adversarial(model=model, loss_fn=loss_fn, epsilon=-0.1)

    def test_init_invalid_steps(self) -> None:
        """Zero or negative steps raises ValueError."""
        model = nn.Linear(3, 1)
        loss_fn = nn.MSELoss()
        with pytest.raises(ValueError, match="steps must be positive"):
            Adversarial(model=model, loss_fn=loss_fn, steps=0)

    def test_augment_without_y_raises(self) -> None:
        """augment without y raises ValueError."""
        model = nn.Linear(3, 1)
        loss_fn = nn.MSELoss()
        adv = Adversarial(model=model, loss_fn=loss_fn)
        x = torch.randn(4, 3)
        with pytest.raises(ValueError, match="Target values must be provided"):
            adv.augment(x, None)  # type: ignore[arg-type]

    def test_augment_eval_mode_no_perturbation(self) -> None:
        """In eval mode, input is returned unchanged."""
        model = nn.Linear(3, 1)
        model.eval()
        loss_fn = nn.MSELoss()
        adv = Adversarial(model=model, loss_fn=loss_fn)
        x = torch.randn(4, 3)
        y = torch.randn(4, 1)
        x_aug, y_aug = adv.augment(x, y)
        assert torch.equal(x_aug, x)
        assert torch.equal(y_aug, y)

    def test_augment_train_mode_modifies_input(self) -> None:
        """In train mode, adversarial perturbation is applied."""
        model = nn.Linear(3, 1)
        model.train()
        loss_fn = nn.MSELoss()
        adv = Adversarial(model=model, loss_fn=loss_fn, epsilon=0.1, steps=2)
        x = torch.randn(4, 3)
        y = torch.randn(4, 1)
        x_aug, y_aug = adv.augment(x, y)
        assert x_aug.shape == x.shape
        assert not torch.equal(x_aug, x)
        # Perturbation should be bounded by epsilon
        delta_l_inf = (x_aug - x).abs().max().item()
        assert delta_l_inf <= 0.1 + 1e-5

    def test_augment_random_start(self) -> None:
        """random_start=True adds initial random noise."""
        model = nn.Linear(3, 1)
        model.train()
        loss_fn = nn.MSELoss()
        adv = Adversarial(model=model, loss_fn=loss_fn, random_start=True, epsilon=0.1)
        x = torch.randn(4, 3)
        y = torch.randn(4, 1)
        x_aug, _ = adv.augment(x, y)
        assert x_aug.shape == x.shape

    def test_augment_epsilon_zero(self) -> None:
        """epsilon=0 means no perturbation possible."""
        model = nn.Linear(3, 1)
        model.train()
        loss_fn = nn.MSELoss()
        adv = Adversarial(model=model, loss_fn=loss_fn, epsilon=0.0)
        x = torch.randn(4, 3)
        y = torch.randn(4, 1)
        x_aug, _ = adv.augment(x, y)
        assert torch.allclose(x_aug, x)
