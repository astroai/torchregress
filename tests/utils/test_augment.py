"""
Unit tests for torchregress.utils.augment.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from torchregress.utils.augment import (
    Adversarial,
    Augmentation,
    EnsemblePerturbationAugmenter,
    FeatureMask,
    GaussianNoise,
    MixUp,
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

    def test_apply_delegates_to_augment(self) -> None:
        """apply() shim calls augment()."""
        aug = DummyAugmentation(probability=1.0)
        x = torch.tensor([1.0, 2.0])
        x_out, _ = aug.apply(x, None)
        assert torch.equal(x_out, x * 2)


# ═══════════════════════════════════════════════════════════════════════════════
# GaussianNoise
# ═══════════════════════════════════════════════════════════════════════════════


class TestGaussianNoise:
    def test_init_defaults(self) -> None:
        """Default std=0.1, probability=0.5."""
        gn = GaussianNoise()
        assert gn.std == 0.1
        assert gn.probability == 0.5

    def test_augment_adds_noise(self) -> None:
        """augment adds random noise to x, leaves y unchanged."""
        gn = GaussianNoise(std=0.1)
        torch.manual_seed(42)
        x = torch.randn(8, 3)
        y = torch.randn(8, 1)
        x_aug, y_aug = gn.augment(x, y)
        assert x_aug.shape == x.shape
        assert not torch.equal(x_aug, x)
        assert torch.equal(y_aug, y)  # type: ignore[arg-type]

    def test_augment_without_y(self) -> None:
        """augment works when y is None."""
        gn = GaussianNoise(std=0.1)
        x = torch.randn(4, 2)
        x_aug, y_aug = gn.augment(x, None)
        assert x_aug.shape == x.shape
        assert y_aug is None

    def test_augment_tensor_std(self) -> None:
        """Per-feature std tensor produces per-dim noise."""
        std = torch.tensor([0.01, 0.1, 1.0])
        gn = GaussianNoise(std=std)
        x = torch.zeros(100, 3)  # zero input, noise = output
        x_aug, _ = gn.augment(x, None)
        # Per-feature std should differ
        stds = x_aug.std(dim=0)
        assert stds[2] > stds[1] > stds[0]


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


# ═══════════════════════════════════════════════════════════════════════════════
# MixUp
# ═══════════════════════════════════════════════════════════════════════════════


class TestMixUp:
    def test_init_defaults(self) -> None:
        """Default alpha=0.2, probability=0.5."""
        mu = MixUp()
        assert mu.alpha == 0.2
        assert mu.probability == 0.5

    def test_augment_without_y_raises(self) -> None:
        """augment without y raises ValueError."""
        mu = MixUp()
        x = torch.randn(4, 3)
        with pytest.raises(ValueError, match="Target values must be provided"):
            mu.augment(x, None)  # type: ignore[arg-type]

    def test_augment_mixes_x_and_y(self) -> None:
        """augment produces convex combinations of x and y."""
        np.random.seed(42)
        mu = MixUp(alpha=1.0)
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        y = torch.tensor([[100.0], [200.0]])
        x_mix, y_mix = mu.augment(x, y)
        assert x_mix.shape == x.shape
        assert y_mix.shape == y.shape  # type: ignore[union-attr]
        # y should be a convex combination
        assert float(y_mix[0].item()) != float(y[0].item())  # type: ignore[index,union-attr]

    def test_augment_batch_shape(self) -> None:
        """Output shapes match input shapes."""
        np.random.seed(42)
        mu = MixUp()
        x = torch.randn(16, 5)
        y = torch.randn(16, 3)
        x_mix, y_mix = mu.augment(x, y)
        assert x_mix.shape == (16, 5)
        assert y_mix.shape == (16, 3)  # type: ignore[union-attr]


# ═══════════════════════════════════════════════════════════════════════════════
# FeatureMask
# ═══════════════════════════════════════════════════════════════════════════════


class TestFeatureMask:
    def test_init_defaults(self) -> None:
        """Default mask_ratio=0.1, probability=0.5."""
        fm = FeatureMask()
        assert fm.mask_ratio == 0.1
        assert fm.probability == 0.5

    def test_init_invalid_mask_ratio_high(self) -> None:
        """mask_ratio >= 1 raises ValueError."""
        with pytest.raises(ValueError, match="Mask ratio must be between 0 and 1"):
            FeatureMask(mask_ratio=1.0)

    def test_init_invalid_mask_ratio_low(self) -> None:
        """mask_ratio < 0 raises ValueError."""
        with pytest.raises(ValueError, match="Mask ratio must be between 0 and 1"):
            FeatureMask(mask_ratio=-0.1)

    def test_augment_mask_ratio_zero_returns_original(self) -> None:
        """mask_ratio=0 returns original data unchanged."""
        fm = FeatureMask(mask_ratio=0.0)
        x = torch.randn(4, 10)
        y = torch.randn(4, 1)
        x_aug, y_aug = fm.augment(x, y)
        assert torch.equal(x_aug, x)
        assert torch.equal(y_aug, y)  # type: ignore[arg-type]

    def test_augment_masks_features(self) -> None:
        """Some features are set to zero."""
        fm = FeatureMask(mask_ratio=0.3)
        x = torch.ones(8, 10)
        x_aug, _ = fm.augment(x, None)
        # At least some features should be zero
        assert (x_aug == 0).any()

    def test_augment_without_y(self) -> None:
        """augment works when y is None."""
        fm = FeatureMask(mask_ratio=0.2)
        x = torch.randn(4, 5)
        x_aug, y_aug = fm.augment(x, None)
        assert x_aug.shape == x.shape
        assert y_aug is None


# ═══════════════════════════════════════════════════════════════════════════════
# EnsemblePerturbationAugmenter
# ═══════════════════════════════════════════════════════════════════════════════


class TestEnsemblePerturbationAugmenter:
    def test_init_defaults(self) -> None:
        """Default n_samples=20, perturb_method='gaussian', sigma=0.1."""
        epa = EnsemblePerturbationAugmenter()
        assert epa.n_samples == 20
        assert epa.perturb_method == "gaussian"
        assert epa.sigma == 0.1
        assert epa.feature_wise is True

    def test_init_feature_wise_false(self) -> None:
        """feature_wise=False uses same noise for all features."""
        epa = EnsemblePerturbationAugmenter(feature_wise=False)
        assert epa.feature_wise is False

    def test_init_invalid_n_samples(self) -> None:
        """n_samples <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="n_samples must be a positive integer"):
            EnsemblePerturbationAugmenter(n_samples=0)

    def test_init_invalid_perturb_method(self) -> None:
        """Invalid perturb_method raises ValueError."""
        with pytest.raises(ValueError, match="perturb_method must be"):
            EnsemblePerturbationAugmenter(perturb_method="laplace")

    def test_forward_gaussian_scalar_feature_wise(self) -> None:
        """Gaussian + scalar sigma + feature_wise=True produces per-feature noise."""
        epa = EnsemblePerturbationAugmenter(sigma=0.1, feature_wise=True, n_samples=5)
        x = torch.randn(4, 3)
        samples = epa(x)
        assert len(samples) == 5
        assert all(s.shape == (4, 3) for s in samples)

    def test_forward_gaussian_scalar_not_feature_wise(self) -> None:
        """Gaussian + scalar sigma + feature_wise=False uses a single scalar sigma."""
        epa = EnsemblePerturbationAugmenter(sigma=0.1, feature_wise=False, n_samples=5)
        x = torch.randn(4, 3)
        samples = epa(x)
        assert len(samples) == 5
        assert all(s.shape == (4, 3) for s in samples)

    def test_forward_gaussian_1d_sigma(self) -> None:
        """Gaussian with 1D sigma tensor."""
        sigma = torch.tensor([0.01, 0.1, 1.0])
        epa = EnsemblePerturbationAugmenter(sigma=sigma, n_samples=200)
        x = torch.randn(50, 3)
        samples = epa(x)
        assert len(samples) == 200
        # Per-feature noise differs (large n_samples makes this reliable)
        stacked = torch.stack(samples)
        stds = stacked.std(dim=(0, 1))
        assert stds[2] > stds[1] > stds[0]

    def test_forward_gaussian_1d_sigma_wrong_shape(self) -> None:
        """1D sigma with wrong size raises ValueError."""
        sigma = torch.tensor([0.1, 0.2])  # 2 features, but x has 3
        epa = EnsemblePerturbationAugmenter(sigma=sigma, n_samples=5)
        x = torch.randn(4, 3)
        with pytest.raises(ValueError, match="doesn't match feature dimension"):
            epa(x)

    def test_forward_gaussian_2d_sigma_full_cov(self) -> None:
        """Gaussian with full 2D covariance matrix."""
        x = torch.randn(4, 3)
        # Pin dtype/device to ``x`` so the fixture doesn't implicitly rely
        # on the augmenter module handling dtype/device of input fixtures internally.
        sigma = torch.eye(3, device=x.device, dtype=x.dtype) + 0.1 * torch.ones(
            3, 3, device=x.device, dtype=x.dtype
        )
        epa = EnsemblePerturbationAugmenter(sigma=sigma, n_samples=5)
        x = torch.randn(4, 3)
        samples = epa(x)
        assert len(samples) == 5
        assert all(s.shape == (4, 3) for s in samples)

    def test_forward_gaussian_2d_sigma_wrong_shape(self) -> None:
        """2D sigma with wrong shape raises ValueError."""
        # ``torch.eye`` is used here purely as a shape-stub for ``pytest.raises``
        # validation: the augmenter checks shape before consuming dtype/device,
        # so the fixture is intentionally unpinned (SKIP per
        # docs/loss_test_coverage.md rationale).
        sigma = torch.eye(2)  # noqa: TOR001
        epa = EnsemblePerturbationAugmenter(sigma=sigma, n_samples=5)
        x = torch.randn(4, 3)
        with pytest.raises(ValueError, match="doesn't match expected shape"):
            epa(x)

    def test_forward_gaussian_2d_fallback_to_diag(self) -> None:
        """Singular 2D cov falls back to diagonal approximation."""
        sigma = torch.zeros(3, 3)  # zero matrix is singular
        epa = EnsemblePerturbationAugmenter(sigma=sigma, n_samples=3)
        x = torch.randn(4, 3)
        samples = epa(x)
        assert len(samples) == 3
        assert all(s.shape == (4, 3) for s in samples)

    def test_forward_uniform_scalar(self) -> None:
        """Uniform perturbation with scalar sigma."""
        epa = EnsemblePerturbationAugmenter(perturb_method="uniform", sigma=0.1, n_samples=5)
        x = torch.randn(4, 3)
        samples = epa(x)
        assert len(samples) == 5
        assert all(s.shape == (4, 3) for s in samples)
        # Uniform noise should differ from input
        for s in samples:
            assert not torch.equal(s, x)

    def test_forward_uniform_2d_sigma(self) -> None:
        """Uniform perturbation with 2D sigma uses diagonal."""
        x = torch.randn(4, 3)
        # Pin dtype/device to ``x`` so the fixture doesn't implicitly rely
        # on the augmenter module handling dtype/device of input fixtures internally.
        sigma = torch.eye(3, device=x.device, dtype=x.dtype) * 0.5
        epa = EnsemblePerturbationAugmenter(perturb_method="uniform", sigma=sigma, n_samples=5)
        samples = epa(x)
        assert len(samples) == 5

    def test_forward_with_device(self) -> None:
        """Device parameter is respected."""
        epa = EnsemblePerturbationAugmenter(sigma=0.1, n_samples=3, device=torch.device("cpu"))
        x = torch.randn(4, 3)
        samples = epa(x)
        assert all(s.device.type == "cpu" for s in samples)

    def test_forward_sigma_tensor_0d_expands(self) -> None:
        """0D sigma tensor expands to n_features when feature_wise=True."""
        sigma = torch.tensor(0.5)
        epa = EnsemblePerturbationAugmenter(sigma=sigma, feature_wise=True, n_samples=3)
        x = torch.randn(4, 3)
        samples = epa(x)
        assert len(samples) == 3

    def test_generate_and_stack(self) -> None:
        """generate_and_stack returns a stacked tensor."""
        epa = EnsemblePerturbationAugmenter(sigma=0.1, n_samples=5)
        x = torch.randn(4, 3)
        stacked = epa.generate_and_stack(x)
        assert stacked.shape == (5, 4, 3)
