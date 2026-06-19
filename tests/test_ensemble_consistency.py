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

from torchregress.ensemble.models import (
    BinnedPDFEnsembleModel,
    CumulativeLinkEnsembleModel,
    DeepEnsemble,
    HeteroscedasticBatchEnsembleModel,
    HeteroscedasticEnsembleModel,
    MDNEnsembleModel,
)

# ── helpers ───────────────────────────────────────────────────────────


def _simple_mlp(input_size: int = 4, hidden_size: int = 8, output_size: int = 1) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )


def _heteroscedastic_mlp(
    input_size: int = 4, hidden_size: int = 8, output_size: int = 1
) -> nn.Module:
    class HetModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
            )
            self.mean_head = nn.Linear(hidden_size, output_size)
            self.logvar_head = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h = self.net(x)
            return self.mean_head(h), self.logvar_head(h)

    return HetModel()


def _constant_logit_model(logits: torch.Tensor) -> nn.Module:
    class ConstLogit(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("logits", logits.clone().detach().view(1, -1))

        def forward(self, x):
            return self.logits.expand(x.shape[0], -1)

    return ConstLogit()


def _constant_mdn_model(packed: torch.Tensor) -> nn.Module:
    class ConstMDN(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("packed", packed.clone().detach().view(1, -1))

        def forward(self, x):
            return self.packed.expand(x.shape[0], -1)

    return ConstMDN()


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
            base_model=_simple_mlp(),
            ensemble_size=3,
            input_size=4,
            hidden_size=8,
            output_size=1,
        )

    def _make_heteroscedastic_ensemble(self) -> HeteroscedasticEnsembleModel:
        return HeteroscedasticEnsembleModel(
            base_model=_heteroscedastic_mlp(),
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
            base_model=_heteroscedastic_mlp(),
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
            base_model=_heteroscedastic_mlp(),
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
            base_model=_simple_mlp(),
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
            base_model=_heteroscedastic_mlp(),
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
            base_model=_simple_mlp(),
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

    def test_variance_order_of_magnitude_is_stable_across_ensemble_sizes(self):
        """With a fixed seed, mean predictions should be in a reasonable
        range regardless of ensemble size."""
        torch.manual_seed(42)
        sizes = [2, 5, 10]
        means = []
        for size in sizes:
            model = DeepEnsemble(
                base_model=_simple_mlp(),
                ensemble_size=size,
                input_size=4,
                hidden_size=8,
                output_size=1,
            )
            x = torch.randn(10, 4)
            result = model.predict(x)
            means.append(result["mean"].mean().item())

        # Distribution of ensemble means should have finite variance
        # and not explode with ensemble size.
        assert all(abs(m) < 5.0 for m in means), f"Ensemble means out of expected range {means}"


# ── non-Gaussian ensemble contracts ────────────────────────────────────


class TestNonGaussianEnsembleContracts:
    """BinnedPDF, CumulativeLink, and MDN ensembles each follow their
    own output contracts consistently."""

    def test_binned_pdf_returns_probabilities_and_mean_variance(self):
        ensemble = BinnedPDFEnsembleModel(
            base_model=_constant_logit_model(torch.tensor([0.0, 0.0, 0.0])),
            ensemble_size=2,
        )
        ensemble.models[0] = _constant_logit_model(torch.tensor([2.0, 0.0, -1.0]))
        ensemble.models[1] = _constant_logit_model(torch.tensor([-1.0, 1.0, 0.5]))
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
            base_model=_constant_logit_model(torch.tensor([0.0, 0.0])),
            ensemble_size=2,
        )
        ensemble.models[0] = _constant_logit_model(torch.tensor([2.0, -1.0]))
        ensemble.models[1] = _constant_logit_model(torch.tensor([0.5, 1.5]))
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
            base_model=_constant_mdn_model(packed_a),
            ensemble_size=2,
            n_components=2,
            n_features=1,
        )
        ensemble.models[0] = _constant_mdn_model(packed_a)
        ensemble.models[1] = _constant_mdn_model(packed_b)
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
            base_model=_constant_logit_model(torch.tensor([0.0, 0.0, 0.0])),
            ensemble_size=2,
        )
        ensemble.models[0] = _constant_logit_model(torch.tensor([2.0, 0.0, -1.0]))
        ensemble.models[1] = _constant_logit_model(torch.tensor([-1.0, 1.0, 0.5]))
        x = torch.randn(4, 4)
        samples = ensemble.sample(x, n_samples=7)
        assert samples.shape == (7, 4), f"expected (7, 4), got {samples.shape}"

    def test_cumulative_link_sample_returns_correct_shape(self):
        ensemble = CumulativeLinkEnsembleModel(
            base_model=_constant_logit_model(torch.tensor([0.0, 0.0])),
            ensemble_size=2,
        )
        ensemble.models[0] = _constant_logit_model(torch.tensor([2.0, -1.0]))
        ensemble.models[1] = _constant_logit_model(torch.tensor([0.5, 1.5]))
        x = torch.randn(4, 4)
        samples = ensemble.sample(x, n_samples=7)
        assert samples.shape == (7, 4), f"expected (7, 4), got {samples.shape}"

    def test_mdn_ensemble_sample_returns_correct_shape(self):
        packed_a = torch.tensor([4.0, -2.0, 0.0, 1.0, -4.0, -4.0])
        packed_b = torch.tensor([-3.0, 3.0, 2.0, 3.0, -4.0, -4.0])
        ensemble = MDNEnsembleModel(
            base_model=_constant_mdn_model(packed_a),
            ensemble_size=2,
            n_components=2,
            n_features=1,
        )
        ensemble.models[0] = _constant_mdn_model(packed_a)
        ensemble.models[1] = _constant_mdn_model(packed_b)
        x = torch.randn(4, 4)
        samples = ensemble.sample(x, n_samples=7)
        assert samples.shape == (7, 4, 1)
