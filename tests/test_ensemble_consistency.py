"""
Cross-class consistency tests for ensemble models.

Verifies that all ensemble models follow a consistent predict() contract
(returning ``mean`` and ``variance``), that heteroscedastic ensembles
correctly decompose epistemic/aleatoric variance, and that non-Gaussian
ensembles follow their own output contracts.

These tests complement (not replace) the per-class unit tests in
``test_ensemble.py``.
"""

import pytest
import torch
from torch import nn

from tests._test_models import (
    ConstantLogitModel,
    ConstantMDNModel,
    HeteroscedasticMLP,
    SimpleMLP,
)
from torchregress.ensemble.models import (
    BinnedPDFEnsembleModel,
    CumulativeLinkEnsembleModel,
    DeepEnsemble,
    HeteroscedasticBatchEnsembleModel,
    HeteroscedasticEnsembleModel,
    MDNEnsembleModel,
)

_GAUSSIAN_ENSEMBLE_TYPES = [
    "DeepEnsemble",
    "HeteroscedasticEnsemble",
    "HeteroscedasticBatchEnsemble",
]


# ── predict() API contract ─────────────────────────────────────────────


class TestPredictAPIContract:
    """Every Gaussian ensemble model's predict() must return a dict
    with at least ``mean`` and ``variance`` keys, and the shapes must
    be consistent."""

    def _make_deep_ensemble(self) -> DeepEnsemble:
        return DeepEnsemble(
            base_model=SimpleMLP(),
            ensemble_size=3,
            input_size=4,
            hidden_size=8,
            output_size=1,
        )

    def _make_heteroscedastic_ensemble(self) -> HeteroscedasticEnsembleModel:
        return HeteroscedasticEnsembleModel(
            base_model=HeteroscedasticMLP(),
            ensemble_size=3,
            input_size=4,
            hidden_size=8,
            output_size=1,
        )

    def _make_heteroscedastic_batch_ensemble(self) -> HeteroscedasticBatchEnsembleModel:
        backbone = nn.Sequential(
            nn.Linear(4, 6),
            nn.ReLU(),
        )
        return HeteroscedasticBatchEnsembleModel(
            backbone=backbone,
            input_size=6,
            output_size=1,
            ensemble_size=3,
        )

    @pytest.mark.parametrize("ensemble_type", _GAUSSIAN_ENSEMBLE_TYPES)
    def test_predict_returns_mean_and_variance(self, ensemble_type):
        if ensemble_type == "DeepEnsemble":
            model = self._make_deep_ensemble()
        elif ensemble_type == "HeteroscedasticEnsemble":
            model = self._make_heteroscedastic_ensemble()
        else:
            model = self._make_heteroscedastic_batch_ensemble()

        x = torch.randn(5, 4)
        result = model.predict(x)

        assert "mean" in result, f"{ensemble_type}: missing 'mean'"
        assert "variance" in result, f"{ensemble_type}: missing 'variance'"
        assert result["mean"].shape == (5, 1), f"{ensemble_type}: mean shape {result['mean'].shape}"
        assert result["variance"].shape == (5, 1), (
            f"{ensemble_type}: variance shape {result['variance'].shape}"
        )

    @pytest.mark.parametrize("ensemble_type", _GAUSSIAN_ENSEMBLE_TYPES)
    def test_variance_is_non_negative(self, ensemble_type):
        if ensemble_type == "DeepEnsemble":
            model = self._make_deep_ensemble()
        elif ensemble_type == "HeteroscedasticEnsemble":
            model = self._make_heteroscedastic_ensemble()
        else:
            model = self._make_heteroscedastic_batch_ensemble()

        x = torch.randn(5, 4)
        result = model.predict(x)
        assert (result["variance"] >= 0).all(), (
            f"{ensemble_type}: negative variance (min={result['variance'].min().item()})"
        )


# ── variance decomposition ─────────────────────────────────────────────


class TestVarianceDecomposition:
    """Heteroscedastic ensembles must decompose total variance into
    epistemic + aleatoric components."""

    def test_heteroscedastic_ensemble_decomposes_variance(self):
        model = HeteroscedasticEnsembleModel(
            base_model=HeteroscedasticMLP(),
            ensemble_size=3,
            input_size=4,
            hidden_size=8,
            output_size=2,
        )
        x = torch.randn(5, 4)
        result = model.predict(x)

        assert "epistemic_variance" in result
        assert "aleatoric_variance" in result

        total = result["variance"]
        epi = result["epistemic_variance"]
        ale = result["aleatoric_variance"]

        torch.testing.assert_close(total, epi + ale, msg="total variance ≠ epistemic + aleatoric")

    def test_batch_ensemble_decomposes_variance(self):
        backbone = nn.Sequential(nn.Linear(4, 6), nn.ReLU())
        model = HeteroscedasticBatchEnsembleModel(
            backbone=backbone,
            input_size=6,
            output_size=2,
            ensemble_size=3,
        )
        x = torch.randn(5, 4)
        result = model.predict(x)

        assert "epistemic_variance" in result
        assert "aleatoric_variance" in result

        total = result["variance"]
        epi = result["epistemic_variance"]
        ale = result["aleatoric_variance"]

        torch.testing.assert_close(total, epi + ale, msg="total variance ≠ epistemic + aleatoric")

    def test_single_member_ensemble_has_zero_epistemic(self):
        """With ensemble_size=1, epistemic variance should be 0
        (no disagreement among members)."""
        model = HeteroscedasticEnsembleModel(
            base_model=HeteroscedasticMLP(),
            ensemble_size=1,
            input_size=4,
            hidden_size=8,
            output_size=1,
        )
        x = torch.randn(5, 4)
        result = model.predict(x)
        torch.testing.assert_close(
            result["epistemic_variance"],
            torch.zeros_like(result["epistemic_variance"]),
        )

    def test_heteroscedastic_batch_single_member_zero_epistemic(self):
        backbone = nn.Sequential(nn.Linear(4, 6), nn.ReLU())
        model = HeteroscedasticBatchEnsembleModel(
            backbone=backbone,
            input_size=6,
            output_size=1,
            ensemble_size=1,
        )
        x = torch.randn(5, 4)
        result = model.predict(x)
        torch.testing.assert_close(
            result["epistemic_variance"],
            torch.zeros_like(result["epistemic_variance"]),
        )


# ── determinism ────────────────────────────────────────────────────────


class TestEnsembleDeterminism:
    """Ensemble predict() should be deterministic: same input → same output."""

    def test_deep_ensemble_predict_is_deterministic(self):
        model = DeepEnsemble(
            base_model=SimpleMLP(),
            ensemble_size=3,
            input_size=4,
            hidden_size=8,
            output_size=1,
        )
        x = torch.randn(5, 4)
        r1 = model.predict(x)
        r2 = model.predict(x)
        torch.testing.assert_close(r1["mean"], r2["mean"])
        torch.testing.assert_close(r1["variance"], r2["variance"])

    def test_heteroscedastic_ensemble_predict_is_deterministic(self):
        model = HeteroscedasticEnsembleModel(
            base_model=HeteroscedasticMLP(),
            ensemble_size=3,
        )
        x = torch.randn(5, 4)
        r1 = model.predict(x)
        r2 = model.predict(x)
        for key in ["mean", "variance", "epistemic_variance", "aleatoric_variance"]:
            torch.testing.assert_close(r1[key], r2[key], msg=f"{key} differs between calls")

    def test_batch_ensemble_predict_is_deterministic(self):
        backbone = nn.Sequential(nn.Linear(4, 6), nn.ReLU())
        model = HeteroscedasticBatchEnsembleModel(
            backbone=backbone,
            input_size=6,
            output_size=1,
            ensemble_size=3,
        )
        x = torch.randn(5, 4)
        r1 = model.predict(x)
        r2 = model.predict(x)
        for key in ["mean", "variance", "epistemic_variance", "aleatoric_variance"]:
            torch.testing.assert_close(r1[key], r2[key], msg=f"{key} differs between calls")


# ── ensemble size behavior ─────────────────────────────────────────────


class TestEnsembleSizeBehavior:
    """Larger ensembles should give more stable estimates."""

    def test_base_ensemble_variance_matches_sample_variance(self):
        """DeepEnsemble.predict()['variance'] should equal the sample
        variance of member predictions."""
        model = DeepEnsemble(
            base_model=SimpleMLP(),
            ensemble_size=4,
            input_size=4,
            hidden_size=8,
            output_size=1,
        )
        x = torch.randn(3, 4)
        result = model.predict(x)

        with torch.no_grad():
            stacked = model.forward(x)
        manual_var = torch.var(stacked, dim=0, unbiased=True)

        torch.testing.assert_close(
            result["variance"], manual_var, msg="DeepEnsemble variance ≠ sample variance of members"
        )

    def test_ensemble_variance_converges_with_larger_ensemble(self):
        """With a fixed seed, the epistemic variance should not
        increase materially when moving from 5 to 10 members."""
        torch.manual_seed(42)
        x = torch.randn(10, 4)

        model5 = DeepEnsemble(
            base_model=SimpleMLP(),
            ensemble_size=5,
            input_size=4,
            hidden_size=8,
            output_size=1,
        )
        model10 = DeepEnsemble(
            base_model=SimpleMLP(),
            ensemble_size=10,
            input_size=4,
            hidden_size=8,
            output_size=1,
        )

        var5 = model5.predict(x)["variance"].mean()
        var10 = model10.predict(x)["variance"].mean()

        # Variance should not grow with ensemble size — it should
        # be of a similar order of magnitude (or shrink slightly).
        assert var10 <= var5 * 2.0, (
            f"variance grew unexpectedly: size-5={var5:.4f}, size-10={var10:.4f}"
        )


# ── non-Gaussian ensemble contracts ────────────────────────────────────


class TestNonGaussianEnsembleContracts:
    """BinnedPDF, CumulativeLink, and MDN ensembles each follow their
    own output contracts consistently."""

    def test_binned_pdf_returns_probabilities_and_mean_variance(self):
        ensemble = BinnedPDFEnsembleModel(
            base_model=ConstantLogitModel(torch.tensor([0.0, 0.0, 0.0])),
            ensemble_size=2,
        )
        ensemble.models[0] = ConstantLogitModel(torch.tensor([2.0, 0.0, -1.0]))
        ensemble.models[1] = ConstantLogitModel(torch.tensor([-1.0, 1.0, 0.5]))
        x = torch.randn(3, 4)
        result = ensemble.predict(x)

        assert "probabilities" in result
        assert "mean" in result
        assert "variance" in result
        assert result["probabilities"].shape == (3, 3)
        assert result["mean"].shape == (3,)
        assert result["variance"].shape == (3,)

        # Probabilities sum to 1
        torch.testing.assert_close(
            result["probabilities"].sum(dim=-1),
            torch.ones(3),
        )

    def test_cumulative_link_returns_probabilities_and_mean_variance(self):
        ensemble = CumulativeLinkEnsembleModel(
            base_model=ConstantLogitModel(torch.tensor([0.0, 0.0])),
            ensemble_size=2,
        )
        ensemble.models[0] = ConstantLogitModel(torch.tensor([2.0, -1.0]))
        ensemble.models[1] = ConstantLogitModel(torch.tensor([0.5, 1.5]))
        x = torch.randn(3, 4)
        result = ensemble.predict(x)

        assert "probabilities" in result
        assert "mean" in result
        assert "variance" in result
        assert result["probabilities"].shape == (3, 3)
        assert result["mean"].shape == (3,)
        assert result["variance"].shape == (3,)

        torch.testing.assert_close(
            result["probabilities"].sum(dim=-1),
            torch.ones(3),
        )

    def test_mdn_ensemble_returns_mixture_params(self):
        packed_a = torch.tensor([4.0, -2.0, 0.0, 1.0, -4.0, -4.0])
        packed_b = torch.tensor([-3.0, 3.0, 2.0, 3.0, -4.0, -4.0])
        ensemble = MDNEnsembleModel(
            base_model=ConstantMDNModel(packed_a),
            ensemble_size=2,
            n_components=2,
            n_features=1,
        )
        ensemble.models[0] = ConstantMDNModel(packed_a)
        ensemble.models[1] = ConstantMDNModel(packed_b)
        x = torch.randn(3, 4)
        result = ensemble.predict(x)

        assert "mixture_weights" in result
        assert "component_means" in result
        assert "component_stds" in result
        assert "mean" in result
        assert "variance" in result

        assert result["mixture_weights"].shape[0] == 3
        assert result["component_means"].shape[1] == 4  # 2 members × 2 components
        assert result["component_stds"].shape[1] == 4

        torch.testing.assert_close(
            result["mixture_weights"].sum(dim=-1),
            torch.ones(3),
        )

    def test_binned_pdf_sample_returns_correct_shape(self):
        ensemble = BinnedPDFEnsembleModel(
            base_model=ConstantLogitModel(torch.tensor([0.0, 0.0, 0.0])),
            ensemble_size=2,
        )
        ensemble.models[0] = ConstantLogitModel(torch.tensor([2.0, 0.0, -1.0]))
        ensemble.models[1] = ConstantLogitModel(torch.tensor([-1.0, 1.0, 0.5]))
        x = torch.randn(4, 4)
        samples = ensemble.sample(x, n_samples=7)
        assert samples.shape == (7, 4), f"expected (7, 4), got {samples.shape}"

    def test_cumulative_link_sample_returns_correct_shape(self):
        ensemble = CumulativeLinkEnsembleModel(
            base_model=ConstantLogitModel(torch.tensor([0.0, 0.0])),
            ensemble_size=2,
        )
        ensemble.models[0] = ConstantLogitModel(torch.tensor([2.0, -1.0]))
        ensemble.models[1] = ConstantLogitModel(torch.tensor([0.5, 1.5]))
        x = torch.randn(4, 4)
        samples = ensemble.sample(x, n_samples=7)
        assert samples.shape == (7, 4), f"expected (7, 4), got {samples.shape}"

    def test_mdn_ensemble_sample_returns_correct_shape(self):
        packed_a = torch.tensor([4.0, -2.0, 0.0, 1.0, -4.0, -4.0])
        packed_b = torch.tensor([-3.0, 3.0, 2.0, 3.0, -4.0, -4.0])
        ensemble = MDNEnsembleModel(
            base_model=ConstantMDNModel(packed_a),
            ensemble_size=2,
            n_components=2,
            n_features=1,
        )
        ensemble.models[0] = ConstantMDNModel(packed_a)
        ensemble.models[1] = ConstantMDNModel(packed_b)
        x = torch.randn(4, 4)
        samples = ensemble.sample(x, n_samples=7)
        assert samples.shape == (7, 4, 1)
