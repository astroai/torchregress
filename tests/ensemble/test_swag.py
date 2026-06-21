"""
Unit tests for torchregress.ensemble.swag — SWAG and MultiSWAG.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchregress.ensemble.swag import SWAG, MultiSWAG


def _make_model(in_dim: int = 3, hidden: int = 8, out_dim: int = 1) -> nn.Module:
    """Small MLP for SWAG testing."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SWAG — construction
# ═══════════════════════════════════════════════════════════════════════════════


class TestSWAGInit:
    def test_default_construction(self) -> None:
        """Default max_num_models=20, var_clamp=1e-30."""
        model = _make_model()
        swag = SWAG(model)
        assert swag.max_num_models == 20
        assert swag.var_clamp == 1e-30
        assert swag.n_models.item() == 0

    def test_custom_max_num_models(self) -> None:
        """Custom max_num_models is stored."""
        model = _make_model()
        swag = SWAG(model, max_num_models=10)
        assert swag.max_num_models == 10

    def test_custom_var_clamp(self) -> None:
        """Custom var_clamp is stored."""
        model = _make_model()
        swag = SWAG(model, var_clamp=1e-6)
        assert swag.var_clamp == 1e-6

    def test_registers_buffers_for_trainable_params(self) -> None:
        """Buffers _mean are created for each trainable param."""
        model = _make_model()
        swag = SWAG(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                buf_name = name.replace(".", "_") + "_mean"
                assert hasattr(swag, buf_name), f"Missing buffer {buf_name}"

    def test_deviations_created(self) -> None:
        """deviations dict has entries for each trainable param."""
        model = _make_model()
        swag = SWAG(model, max_num_models=5)
        trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
        assert len(swag.deviations) == trainable_count
        for name, dev in swag.deviations.items():
            assert dev.shape[0] == 5  # max_num_models


# ═══════════════════════════════════════════════════════════════════════════════
# SWAG — collect_model
# ═══════════════════════════════════════════════════════════════════════════════


class TestSWAGCollectModel:
    def test_n_models_increments(self) -> None:
        """collect_model increments n_models."""
        model = _make_model()
        swag = SWAG(model)
        assert swag.n_models.item() == 0
        swag.collect_model(model)
        assert swag.n_models.item() == 1
        swag.collect_model(model)
        assert swag.n_models.item() == 2

    def test_updates_mean_buffer(self) -> None:
        """After collection, mean buffer reflects the parameter values."""
        model = _make_model()
        swag = SWAG(model, max_num_models=5)
        swag.collect_model(model)
        # Mean buffer should equal the current parameter values (single snapshot)
        for (name, param), (_, swag_param) in zip(
            model.named_parameters(), swag.base_model.named_parameters()
        ):
            if param.requires_grad:
                name_cleaned = swag._name_map[name]
                mean_buf = getattr(swag, f"{name_cleaned}_mean")
                assert torch.allclose(mean_buf, param.data)

    def test_online_mean_with_two_snapshots(self) -> None:
        """After two collect_model calls with different weights, mean is correct."""
        model = _make_model()
        swag = SWAG(model, max_num_models=5)
        swag.collect_model(model)

        # Modify model weights
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)

        swag.collect_model(model)
        # Mean should be average: (orig + (orig+1)) / 2 = orig + 0.5
        swag.base_model.load_state_dict(model.state_dict())
        # Reset base model to original, then we can check mean
        for (name, param), (_, swag_param) in zip(
            model.named_parameters(), swag.base_model.named_parameters()
        ):
            if param.requires_grad:
                name_cleaned = swag._name_map[name]
                mean_buf = getattr(swag, f"{name_cleaned}_mean")
                assert torch.allclose(mean_buf, param.data - 0.5, atol=1e-6)

    def test_deviations_stored_at_correct_index(self) -> None:
        """Deviations are stored at n_models % max_num_models."""
        model = _make_model()
        swag = SWAG(model, max_num_models=2)
        # First collection at index 0
        swag.collect_model(model)
        for name, dev in swag.deviations.items():
            # Deviation = param - mean = param - param = 0 (first snapshot)
            assert torch.allclose(dev[0], torch.zeros_like(dev[0]), atol=1e-6)

        # Modify model
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)

        # Second collection at index 1
        swag.collect_model(model)
        # Third collection wraps around to index 0
        swag.collect_model(model)
        # Index 0 should now contain the third collection's deviation
        for name in swag.deviations:
            assert torch.any(swag.deviations[name][0] != 0)


# ═══════════════════════════════════════════════════════════════════════════════
# SWAG — sample
# ═══════════════════════════════════════════════════════════════════════════════


class TestSWAGSample:
    def test_raises_before_collection(self) -> None:
        """sample() raises ValueError if no models collected."""
        model = _make_model()
        swag = SWAG(model)
        with pytest.raises(ValueError, match="No models collected"):
            swag.sample()

    def test_sample_modifies_base_model_params(self) -> None:
        """After sampling, base model parameters differ from original."""
        model = _make_model()
        swag = SWAG(model, max_num_models=5)

        # Collect a few snapshots with different weights
        for i in range(3):
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(0.1)
            swag.collect_model(model)

        # Collect a few snapshots with different weights
        swag.sample(scale=1.0, diag_noise=True)

        # Params should differ from original (unless sampled near mean by chance)
        any_different = False
        for orig, new in zip([p.clone() for p in model.parameters()], swag.base_model.parameters()):
            if not torch.allclose(orig, new.data):
                any_different = True
                break
        assert any_different

    def test_diag_noise_false_gives_mean(self) -> None:
        """With diag_noise=False, sample() sets params to the mean."""
        model = _make_model()
        swag = SWAG(model, max_num_models=5)

        for _ in range(3):
            swag.collect_model(model)

        swag.sample(scale=1.0, diag_noise=False)

        # With diag_noise=False, params should be set to mean
        for name, param in swag.base_model.named_parameters():
            if param.requires_grad:
                name_cleaned = swag._name_map[name]
                mean_buf = getattr(swag, f"{name_cleaned}_mean")
                assert torch.allclose(param.data, mean_buf)

    def test_scale_changes_variance(self) -> None:
        """Higher scale produces more variance in sampled parameters."""
        model = _make_model()
        swag = SWAG(model, max_num_models=10)

        for _ in range(5):
            swag.collect_model(model)

        # Sample with scale=0.1
        swag.sample(scale=0.1, diag_noise=True)
        params_small = [p.clone() for p in swag.base_model.parameters()]

        # Reset to mean and sample with scale=5.0
        swag.sample(scale=1.0, diag_noise=False)  # reset to mean
        swag.sample(scale=5.0, diag_noise=True)
        params_large = [p.clone() for p in swag.base_model.parameters()]

        # Large scale should deviate more from mean
        mean_params = [
            getattr(swag, f"{swag._name_map[name]}_mean")
            for name, p in model.named_parameters()
            if p.requires_grad
        ]
        small_dist = sum((ps - mp).norm().item() for ps, mp in zip(params_small, mean_params))
        large_dist = sum((pl - mp).norm().item() for pl, mp in zip(params_large, mean_params))
        assert large_dist > small_dist


# ═══════════════════════════════════════════════════════════════════════════════
# SWAG — forward
# ═══════════════════════════════════════════════════════════════════════════════


class TestSWAGForward:
    def test_forward_delegates_to_base_model(self) -> None:
        """forward passes through base_model."""
        model = _make_model()
        swag = SWAG(model)
        x = torch.randn(4, 3)
        out = swag(x)
        # Should match base_model output
        expected = model(x)
        assert torch.allclose(out, expected)


# ═══════════════════════════════════════════════════════════════════════════════
# MultiSWAG
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultiSWAGInit:
    def test_default_construction(self) -> None:
        """Default n_models=5."""
        model = _make_model()
        mswag = MultiSWAG(model)
        assert mswag.n_models == 5
        assert len(mswag.swag_models) == 5

    def test_custom_n_models(self) -> None:
        """Custom n_models creates that many SWAG instances."""
        model = _make_model()
        mswag = MultiSWAG(model, n_models=3)
        assert mswag.n_models == 3
        assert len(mswag.swag_models) == 3

    def test_each_swag_independent(self) -> None:
        """Each SWAG gets its own deep copy of the base model."""
        model = _make_model()
        mswag = MultiSWAG(model, n_models=3)
        # Base model parameters should be independent copies
        swag0 = mswag.swag_models[0]
        swag1 = mswag.swag_models[1]
        for p0, p1 in zip(swag0.base_model.parameters(), swag1.base_model.parameters()):
            assert p0 is not p1  # different objects


# ═══════════════════════════════════════════════════════════════════════════════
# MultiSWAG — predict_with_uncertainty
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultiSWAGPredictUncertainty:
    def test_returns_mean_and_variances(self) -> None:
        """Returns (mean, epistemic_var, aleatoric_var)."""
        model = _make_model()
        mswag = MultiSWAG(model, n_models=2, max_num_models=3)

        # Collect snapshots for each SWAG
        for swag_model in mswag.swag_models:
            for _ in range(3):
                swag_model.collect_model(model)  # type: ignore[arg-type]

        x = torch.randn(4, 3)
        mean, epistemic, aleatoric = mswag.predict_with_uncertainty(x, n_samples=2, scale=1.0)
        assert mean.shape == (4, 1)
        assert epistemic.shape == (4, 1)
        assert aleatoric.shape == (4, 1)

    def test_epistemic_variance_non_negative(self) -> None:
        """Epistemic variance should be >= 0."""
        model = _make_model()
        mswag = MultiSWAG(model, n_models=2, max_num_models=3)

        for swag_model in mswag.swag_models:
            for _ in range(3):
                swag_model.collect_model(model)  # type: ignore[arg-type]

        x = torch.randn(4, 3)
        _, epistemic, _ = mswag.predict_with_uncertainty(x, n_samples=2)
        assert torch.all(epistemic >= 0)

    def test_aleatoric_is_zero(self) -> None:
        """Aleatoric variance is zero for point-estimate base models."""
        model = _make_model()
        mswag = MultiSWAG(model, n_models=2, max_num_models=3)

        for swag_model in mswag.swag_models:
            for _ in range(3):
                swag_model.collect_model(model)  # type: ignore[arg-type]

        x = torch.randn(4, 3)
        _, _, aleatoric = mswag.predict_with_uncertainty(x, n_samples=2)
        assert torch.all(aleatoric == 0)

    def test_single_swag_small_epistemic(self) -> None:
        """With n_models=2 collecting from same model, epistemic variance is near zero."""
        model = _make_model()
        mswag = MultiSWAG(model, n_models=2, max_num_models=3)

        for swag_model in mswag.swag_models:
            for _ in range(3):
                swag_model.collect_model(model)  # type: ignore[arg-type]

        x = torch.randn(4, 3)
        _, epistemic, _ = mswag.predict_with_uncertainty(x, n_samples=2)
        # Same model collected for both SWAGs → similar means → low epistemic
        assert torch.all(torch.isfinite(epistemic))
        assert float(epistemic.max().item()) < 0.1


# ═══════════════════════════════════════════════════════════════════════════════
# MultiSWAG — predict_with_samples
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultiSWAGPredictSamples:
    def test_shape(self) -> None:
        """Returns [n_models * n_samples, batch, out_dim]."""
        model = _make_model()
        mswag = MultiSWAG(model, n_models=2, max_num_models=3)

        for swag_model in mswag.swag_models:
            for _ in range(3):
                swag_model.collect_model(model)  # type: ignore[arg-type]

        x = torch.randn(4, 3)
        samples = mswag.predict_with_samples(x, n_samples=3)
        assert samples.shape == (2 * 3, 4, 1)

    def test_samples_have_variance(self) -> None:
        """Samples should vary from each other."""
        model = _make_model()
        mswag = MultiSWAG(model, n_models=2, max_num_models=3)

        for swag_model in mswag.swag_models:
            for _ in range(5):
                swag_model.collect_model(model)  # type: ignore[arg-type]

        x = torch.randn(4, 3)
        samples = mswag.predict_with_samples(x, n_samples=5, scale=1.0)
        # Check that not all samples are identical
        std_across_samples = samples.std(dim=0)
        assert torch.any(std_across_samples > 0)


# ═══════════════════════════════════════════════════════════════════════════════
# MultiSWAG — forward
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultiSWAGForward:
    def test_forward_returns_mean(self) -> None:
        """forward returns mean prediction across models."""
        model = _make_model()
        mswag = MultiSWAG(model, n_models=2, max_num_models=3)

        for swag_model in mswag.swag_models:
            for _ in range(3):
                swag_model.collect_model(model)  # type: ignore[arg-type]

        x = torch.randn(4, 3)
        out = mswag(x, n_samples=2)
        assert out.shape == (4, 1)

    def test_forward_default_n_samples(self) -> None:
        """Default n_samples=1."""
        model = _make_model()
        mswag = MultiSWAG(model, n_models=1, max_num_models=2)

        mswag.swag_models[0].collect_model(model)
        mswag.swag_models[0].collect_model(model)

        x = torch.randn(4, 3)
        out = mswag(x)  # uses default n_samples=1
        assert out.shape == (4, 1)
