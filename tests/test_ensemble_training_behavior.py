from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchregress.ensemble import BaseEnsembleModel, HeteroscedasticEnsembleModel
from torchregress.losses import GaussianNLLLoss


class _TupleHeteroscedasticRegressor(nn.Module):
    def __init__(self, input_dim: int = 3, output_dim: int = 2) -> None:
        super().__init__()
        self.mean = nn.Linear(input_dim, output_dim)
        self.log_var = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mean(x), self.log_var(x).clamp(min=-5.0, max=5.0)


class _ConcatHeteroscedasticRegressor(nn.Module):
    def __init__(self, input_dim: int = 3, output_dim: int = 2) -> None:
        super().__init__()
        self.mean = nn.Linear(input_dim, output_dim)
        self.log_var = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.mean(x), self.log_var(x).clamp(min=-5.0, max=5.0)],
            dim=-1,
        )


class _DictOutputModel(nn.Module):
    def __init__(self, input_dim: int = 3, output_dim: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        y = self.linear(x)
        return {"mean": y}


def _tiny_regressor(input_dim: int = 3, output_dim: int = 1) -> nn.Module:
    return nn.Sequential(nn.Linear(input_dim, 8), nn.ReLU(), nn.Linear(8, output_dim))


def test_deep_ensemble_fit_returns_member_histories_and_finite_losses() -> None:
    torch.manual_seed(0)
    x = torch.randn(16, 3)
    y = (x.sum(dim=1, keepdim=True) + 0.1 * torch.randn(16, 1)).detach()
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    ensemble = BaseEnsembleModel(_tiny_regressor(), ensemble_size=2)
    history = ensemble.fit(loader, nn.MSELoss(), epochs=1, lr=1e-2, verbose=False)

    assert "member_histories" in history
    assert len(history["member_histories"]) == 2
    for member_history in history["member_histories"]:
        assert len(member_history) == 1
        assert torch.isfinite(torch.tensor(member_history[0]))


def test_deep_ensemble_fit_supports_optimizer_factory_tuple() -> None:
    """Per-member ``optimizer_factory`` may return multiple optimizers (e.g. AdamW + Muon)."""
    torch.manual_seed(0)
    x = torch.randn(16, 3)
    y = (x.sum(dim=1, keepdim=True) + 0.1 * torch.randn(16, 1)).detach()
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
    lr = 1e-2

    def optimizer_factory(model: nn.Module) -> tuple[torch.optim.Adam, torch.optim.Adam]:
        plist = list(model.parameters())
        return (
            torch.optim.Adam(plist[:2], lr=lr),
            torch.optim.Adam(plist[2:], lr=lr),
        )

    ensemble = BaseEnsembleModel(_tiny_regressor(), ensemble_size=2)
    history = ensemble.fit(
        loader,
        nn.MSELoss(),
        epochs=1,
        lr=lr,
        optimizer_factory=optimizer_factory,
        verbose=False,
    )

    assert "member_histories" in history
    assert len(history["member_histories"]) == 2
    assert hasattr(ensemble, "_optimizers")
    assert all(isinstance(o, tuple) and len(o) == 2 for o in ensemble._optimizers)


def test_deep_ensemble_fit_supports_optimizer_kwargs() -> None:
    torch.manual_seed(0)
    x = torch.randn(16, 3)
    y = (x.sum(dim=1, keepdim=True) + 0.1 * torch.randn(16, 1)).detach()
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    ensemble = BaseEnsembleModel(_tiny_regressor(), ensemble_size=2)
    history = ensemble.fit(
        loader,
        nn.MSELoss(),
        epochs=1,
        lr=1e-2,
        optimizer_kwargs={"weight_decay": 1.0e-3},
        verbose=False,
    )

    assert "member_histories" in history
    assert len(history["member_histories"]) == 2
    assert hasattr(ensemble, "_optimizers")
    for optimizer in ensemble._optimizers:
        assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1.0e-3)


def test_base_ensemble_fit_rejects_invalid_batch_format() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 3)
    loader = DataLoader(TensorDataset(x), batch_size=4, shuffle=False)
    ensemble = BaseEnsembleModel(_tiny_regressor(), ensemble_size=2)

    with pytest.raises(ValueError, match="train_loader must yield"):
        ensemble.fit(loader, nn.MSELoss(), epochs=1, verbose=False)


def test_base_ensemble_predict_rejects_non_tensor_member_outputs() -> None:
    torch.manual_seed(0)
    ensemble = BaseEnsembleModel(_DictOutputModel(), ensemble_size=2)
    x = torch.randn(4, 3)

    with pytest.raises(ValueError, match="expects tensor outputs"):
        ensemble.predict(x)


def test_heteroscedastic_ensemble_predict_handles_tuple_outputs() -> None:
    torch.manual_seed(0)
    ensemble = HeteroscedasticEnsembleModel(_TupleHeteroscedasticRegressor(), ensemble_size=3)
    x = torch.randn(5, 3)

    result = ensemble.predict(x)
    assert set(result) == {
        "mean",
        "variance",
        "epistemic_variance",
        "aleatoric_variance",
    }
    for key, value in result.items():
        assert value.shape == (5, 2)
        assert torch.all(torch.isfinite(value))
    assert torch.all(result["variance"] >= 0)
    assert torch.allclose(
        result["variance"],
        result["epistemic_variance"] + result["aleatoric_variance"],
        atol=1e-6,
        rtol=1e-5,
    )


def test_heteroscedastic_ensemble_full_covariance_handles_concatenated_outputs() -> None:
    torch.manual_seed(0)
    ensemble = HeteroscedasticEnsembleModel(_ConcatHeteroscedasticRegressor(), ensemble_size=3)
    x = torch.randn(4, 3)

    result = ensemble.predict_full_covariance(x)
    assert set(result) == {
        "mean",
        "epistemic_covariance",
        "aleatoric_covariance",
        "total_covariance",
    }
    assert result["mean"].shape == (4, 2)
    assert result["epistemic_covariance"].shape == (4, 2, 2)
    assert result["aleatoric_covariance"].shape == (4, 2, 2)
    assert result["total_covariance"].shape == (4, 2, 2)
    assert torch.all(torch.diagonal(result["aleatoric_covariance"], dim1=-2, dim2=-1) >= 0)



