from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import torchregress.algorithms.irls as irls_mod
from torchregress.losses import WeightedL1Loss


def _tiny_loader(n: int = 8, batch_size: int = 4) -> DataLoader:
    torch.manual_seed(0)
    x = torch.randn(n, 3)
    y = (x.sum(dim=1, keepdim=True) + 0.05 * torch.randn(n, 1)).detach()
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _fake_irls_factory(call_log: list[dict[str, Any]]):
    def _fake_iteratively_reweighted_least_squares(**kwargs: Any):
        y_true = kwargs["y_true"]
        call_log.append(
            {
                "update_context_batch": kwargs["x"].shape[0],
                "return_all_predictions": kwargs.get("return_all_predictions", False),
                "batch_size_arg": kwargs.get("batch_size"),
            }
        )
        y_pred = torch.zeros_like(y_true)
        precision = torch.ones_like(y_true)
        history = [0.5, 0.25]
        if kwargs.get("return_all_predictions", False):
            return y_pred, history, precision, [y_pred.clone(), y_pred.clone()]
        return y_pred, history, precision

    return _fake_iteratively_reweighted_least_squares


def test_irls_epoch_reweighting_validation_and_callbacks(monkeypatch) -> None:
    loader = _tiny_loader(n=8, batch_size=4)
    val_loader = _tiny_loader(n=4, batch_size=2)
    model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1))
    loss_fn = WeightedL1Loss()

    irls_calls: list[dict[str, Any]] = []
    callback_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        irls_mod,
        "iteratively_reweighted_least_squares",
        _fake_irls_factory(irls_calls),
    )
    monkeypatch.setattr(irls_mod, "validate_model", lambda *args, **kwargs: 0.123)

    def _callback(**kwargs: Any) -> None:
        callback_calls.append(kwargs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = irls_mod.IRLS(
            model=model,
            train_data=loader,
            loss_fn=loss_fn,
            num_epochs=2,
            batch_size=4,
            device="cpu",
            verbose=False,
            progress_bar=False,
            update_weights="epoch",
            val_data=val_loader,
            val_freq=1,
            callbacks=[_callback],
            return_all_iterations=True,
            base_loss="l1",
        )

    # One epoch-level reweighting per epoch (2 epochs)
    assert len(irls_calls) == 2
    assert all(call["return_all_predictions"] is True for call in irls_calls)
    assert "val_loss_history" in result and len(result["val_loss_history"]) == 2
    assert "all_iterations" in result and len(result["all_iterations"]) == 2
    assert len(callback_calls) == 2
    assert callback_calls[-1]["update_type"] == "epoch"
    assert callback_calls[-1]["val_loss"] == 0.123
    assert callback_calls[-1]["train_loss"] is not None


def test_irls_batch_reweighting_calls_irls_per_batch(monkeypatch) -> None:
    loader = _tiny_loader(n=8, batch_size=4)
    model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1))
    loss_fn = WeightedL1Loss()

    irls_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        irls_mod,
        "iteratively_reweighted_least_squares",
        _fake_irls_factory(irls_calls),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = irls_mod.IRLS(
            model=model,
            train_data=loader,
            loss_fn=loss_fn,
            num_epochs=1,
            batch_size=4,
            device="cpu",
            verbose=False,
            progress_bar=False,
            update_weights="batch",
            base_loss="l1",
        )

    # Two training batches -> two batch-level IRLS calls
    assert len(irls_calls) == 2
    assert all(call["update_context_batch"] == 4 for call in irls_calls)
    assert "train_loss_history" in result and len(result["train_loss_history"]) == 1
