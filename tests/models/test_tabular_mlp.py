"""Tests for torchregress.models: TabularMLP, TabularPreprocessor, fit_tabular."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.datasets import make_friedman1
from torch import nn

import torchregress as tr
from torchregress.losses import GaussianNLLLoss, MultiQuantileLoss, WeightedMSELoss
from torchregress.models import (
    PeriodicEmbedding,
    TabularMLP,
    TabularPreprocessor,
    fit_tabular,
    fit_tabular_ensemble,
)


@pytest.fixture(autouse=True)
def _single_thread():
    """Small MLPs are fastest on one thread and avoid oversubscription under xdist."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _data(n: int = 400, d: int = 5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2] + 0.1 * rng.normal(size=n)
    return X, y


def _rmse(a: torch.Tensor | np.ndarray, b: np.ndarray) -> float:
    a = a.numpy() if isinstance(a, torch.Tensor) else a
    return float(np.sqrt(np.mean((np.asarray(a).ravel() - b) ** 2)))


class TestModel:
    @pytest.mark.parametrize("embedding", ["periodic", "none"])
    @pytest.mark.parametrize("activation", ["silu", "selu", "relu"])
    def test_shapes(self, embedding, activation):
        m = TabularMLP(7, 3, hidden=(16, 8), embedding=embedding, activation=activation)
        assert m(torch.randn(11, 7)).shape == (11, 3)

    def test_default_hidden_and_embedding_width(self):
        m = TabularMLP(4, 1)
        assert m.hidden == (256, 256, 256)
        assert isinstance(m.embedding, PeriodicEmbedding)
        assert m.embedding.out_features == 4 * (8 + 1)

    def test_callable_activation(self):
        m = TabularMLP(3, 1, hidden=(4,), activation=nn.Tanh)
        assert any(isinstance(layer, nn.Tanh) for layer in m.blocks)

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            TabularMLP(3, 1, embedding="piecewise")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            TabularMLP(3, 1, activation="nope")
        with pytest.raises(ValueError):
            TabularMLP(3, 1, dropout=1.0)

    def test_float64(self):
        X, y = _data(200, 3)
        model = TabularMLP(3, 1, hidden=(16,)).double()
        fit = fit_tabular(model, WeightedMSELoss(), X, y, epochs=3)
        out = fit.predict(X)
        assert out.dtype == torch.float64
        assert next(fit.model.parameters()).dtype == torch.float64


class TestPreprocessor:
    def test_robust_scaling_statistics(self):
        rng = np.random.default_rng(0)
        X = rng.normal(5.0, 3.0, size=(2000, 2))
        Z = TabularPreprocessor(clip=None).fit_transform(X)
        np.testing.assert_allclose(np.median(Z, 0), 0.0, atol=1e-9)
        iqr = np.subtract(*np.percentile(Z, [75, 25], axis=0))
        np.testing.assert_allclose(iqr, 1.0, rtol=1e-9)

    def test_smooth_clip_bounds_outliers(self):
        X = np.r_[np.random.default_rng(0).normal(size=(500, 1)), [[1e6]]]
        Z = TabularPreprocessor(clip_value=3.0).fit_transform(X)
        assert np.abs(Z).max() < 3.0
        Zh = TabularPreprocessor(clip="hard", clip_value=3.0).fit_transform(X)
        assert Zh.max() == 3.0

    @pytest.mark.parametrize("clip", ["smooth", None])
    def test_round_trip(self, clip):
        X = np.random.default_rng(1).standard_t(3, size=(300, 4)) * [1, 10, 100, 0.1]
        prep = TabularPreprocessor(clip=clip).fit(X)
        np.testing.assert_allclose(prep.inverse_transform(prep.transform(X)), X, rtol=1e-4)

    def test_missing_values_impute_and_indicator(self):
        X = np.array([[1.0, 10.0], [2.0, np.nan], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])
        prep = TabularPreprocessor(clip=None).fit(X)
        assert prep.n_features_in_ == 2 and prep.n_features_out_ == 3
        Z = prep.transform(X)
        assert np.isfinite(Z).all()
        assert Z[1, 1] == 0.0 and Z[1, 2] == 1.0 and Z[:, 2].sum() == 1.0
        back = prep.inverse_transform(Z)
        assert np.isnan(back[1, 1])
        np.testing.assert_allclose(np.delete(back, 1, 0), np.delete(X, 1, 0))

    def test_missing_indicator_modes(self):
        X = np.array([[1.0, np.nan], [2.0, 3.0], [3.0, 4.0]])
        assert TabularPreprocessor(missing_indicator="never").fit(X).n_features_out_ == 2
        assert TabularPreprocessor(missing_indicator="always").fit(X).n_features_out_ == 4

    def test_constant_and_all_nan_columns(self):
        X = np.c_[np.ones(10), np.full(10, np.nan), np.arange(10.0)]
        Z = TabularPreprocessor().fit_transform(X)
        assert np.isfinite(Z).all()

    def test_tensor_in_tensor_out_and_dtype(self):
        X = torch.randn(50, 3, dtype=torch.float64)
        prep = TabularPreprocessor().fit(X)
        Z = prep.transform(X)
        assert isinstance(Z, torch.Tensor) and Z.dtype == torch.float64
        assert prep.transform(X.numpy().astype(np.float32)).dtype == np.float32

    def test_errors(self):
        with pytest.raises(RuntimeError):
            TabularPreprocessor().transform(np.zeros((2, 2)))
        prep = TabularPreprocessor().fit(np.random.default_rng(0).normal(size=(10, 2)))
        with pytest.raises(ValueError):
            prep.transform(np.zeros((2, 3)))
        with pytest.raises(ValueError):
            TabularPreprocessor(clip="x")  # type: ignore[arg-type]


class TestFit:
    def test_determinism_with_seed(self):
        X, y = _data(300, 4)

        def run(seed: int) -> torch.Tensor:
            torch.manual_seed(7)
            m = TabularMLP(4, 1, hidden=(16, 16))
            return fit_tabular(m, WeightedMSELoss(), X, y, epochs=4, seed=seed).predict(X)

        a, b, c = run(0), run(0), run(1)
        assert torch.equal(a, b)
        assert not torch.equal(a, c)

    def test_global_rng_untouched(self):
        X, y = _data(100, 3)
        m = TabularMLP(3, 1, hidden=(8,))
        torch.manual_seed(5)
        expected = torch.rand(1)
        torch.manual_seed(5)
        fit_tabular(m, WeightedMSELoss(), X, y, epochs=2)
        assert torch.equal(torch.rand(1), expected)

    def test_early_stopping_restores_best_weights(self):
        X, y = _data(300, 3)
        torch.manual_seed(0)
        m = TabularMLP(3, 1, hidden=(64, 64), dropout=0.0)
        fit = fit_tabular(
            m, WeightedMSELoss(), X[:60], y[:60], X_val=X[60:], y_val=y[60:], epochs=300,
            lr=1e-2, weight_decay=0.0, patience=3, scheduler="none",
        )  # fmt: skip
        h = fit.history["val_loss"]
        assert fit.stopped_early and len(h) < 300
        assert fit.best_epoch == int(np.argmin(h)) and fit.best_val_loss == min(h)
        assert fit.best_epoch < len(h) - 1
        # restored weights reproduce the best validation loss, not the last one
        Xv = fit.preprocessor.transform(X[60:])  # type: ignore[union-attr]
        yv = (
            torch.as_tensor(y[60:], dtype=torch.float32)[:, None] - fit.target_mean
        ) / fit.target_std
        with torch.no_grad():
            val = float(WeightedMSELoss()(fit.model(torch.as_tensor(Xv, dtype=torch.float32)), yv))
        assert val == pytest.approx(fit.best_val_loss, rel=1e-4)
        assert val < h[-1]

    def test_no_validation_trains_all_epochs(self):
        X, y = _data(100, 3)
        fit = fit_tabular(
            TabularMLP(3, 1, hidden=(8,)), WeightedMSELoss(), X, y, val_fraction=0, epochs=3
        )
        assert len(fit.history["train_loss"]) == 3 and not fit.stopped_early

    def test_gaussian_nll_head_maps_back_to_original_scale(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(-2, 2, size=(1200, 3))
        sd = 0.2 + 0.8 * (X[:, 0] > 0)
        y = 1000.0 + 50.0 * X[:, 1] + 100.0 * sd * rng.normal(size=1200)
        torch.manual_seed(0)
        fit = fit_tabular(
            TabularMLP(3, 2, hidden=(32, 32)), GaussianNLLLoss(), X, y, epochs=40, batch_size=128
        )
        assert fit.output_layout == "gaussian"
        out = fit.predict(X)
        assert out.shape == (1200, 2)
        mean, logvar = out[:, 0].numpy(), out[:, 1].numpy()
        assert abs(mean.mean() - 1000.0) < 20
        std = np.exp(0.5 * logvar)
        assert std[X[:, 0] > 0].mean() > 1.5 * std[X[:, 0] < 0].mean()
        # outputs plug straight into the same loss in the original units
        assert torch.isfinite(
            GaussianNLLLoss()(out, torch.as_tensor(y, dtype=torch.float32)[:, None])
        )

    def test_multi_quantile_head(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(1500, 2))
        y = X[:, 0] + rng.normal(size=1500)
        qs = [0.1, 0.5, 0.9]
        torch.manual_seed(0)
        fit = fit_tabular(
            TabularMLP(2, 3, hidden=(32, 32)),
            MultiQuantileLoss(qs),
            X,
            y,
            epochs=30,
            batch_size=128,
        )
        q = fit.predict(X).numpy()
        assert q.shape == (1500, 3)
        cover = np.mean((y >= q[:, 0]) & (y <= q[:, 2]))
        assert 0.7 < cover < 0.9  # nominal 0.8
        assert (q[:, 0] < q[:, 2]).mean() > 0.95

    def test_nan_features_are_imputed(self):
        X, y = _data(200, 3)
        X[::7, 1] = np.nan
        prep = TabularPreprocessor().fit(X)
        assert prep.n_features_out_ == 4
        fit = fit_tabular(
            TabularMLP(prep.n_features_out_, 1, hidden=(8,)), WeightedMSELoss(), X, y, epochs=2
        )
        assert torch.isfinite(fit.predict(X)).all()

    def test_validation_errors(self):
        X, y = _data(50, 3)
        m = TabularMLP(3, 1, hidden=(4,))
        with pytest.raises(ValueError, match="both X_val and y_val"):
            fit_tabular(m, WeightedMSELoss(), X, y, X_val=X)
        with pytest.raises(ValueError, match="expects 5 input features"):
            fit_tabular(TabularMLP(5, 1), WeightedMSELoss(), X, y)
        with pytest.raises(ValueError, match="NaN"):
            fit_tabular(m, WeightedMSELoss(), X, np.where(np.arange(50) == 3, np.nan, y))

    def test_ensemble_averages_members(self):
        X, y = _data(200, 3)
        ens = fit_tabular_ensemble(
            lambda d: TabularMLP(d, 1, hidden=(8,)), WeightedMSELoss(), X, y, n_members=3, epochs=2
        )
        members = ens.predict_members(X)
        assert members.shape == (3, 200, 1)
        assert not torch.equal(members[0], members[1])
        torch.testing.assert_close(ens.predict(X), members.mean(0))


def _plain_mlp_rmse(Xtr, ytr, Xte, yte, epochs: int, seed: int) -> float:
    """The baseline recipe: standardised data, 2x64 ReLU, Adam 1e-3, batch 64, no stopping."""
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    ym, ys = ytr.mean(), ytr.std()
    Xt = torch.as_tensor((Xtr - mu) / sd, dtype=torch.float32)
    yt = torch.as_tensor((ytr - ym) / ys, dtype=torch.float32)[:, None]
    net = nn.Sequential(
        nn.Linear(Xt.shape[1], 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)
    )
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(epochs):
        order = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 64):
            idx = order[i : i + 64]
            loss = ((net(Xt[idx]) - yt[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = net(torch.as_tensor((Xte - mu) / sd, dtype=torch.float32))[:, 0].numpy() * ys + ym
    return _rmse(pred, yte)


def test_beats_plain_mlp_on_friedman1_with_same_epoch_budget():
    X, y = make_friedman1(2200, n_features=10, noise=1.0, random_state=0)
    Xtr, ytr, Xte, yte = X[:1500], y[:1500], X[1500:], y[1500:]
    epochs = 30
    torch.manual_seed(0)
    fit = fit_tabular(
        TabularMLP(10, 1, hidden=(128, 128)), WeightedMSELoss(), Xtr, ytr,
        epochs=epochs, batch_size=64, patience=epochs, seed=0,
    )  # fmt: skip
    ours = _rmse(fit.predict(Xte), yte)
    plain = _plain_mlp_rmse(Xtr, ytr, Xte, yte, epochs, seed=0)
    assert ours < 0.95 * plain, (ours, plain)


def test_models_is_lazy_top_level_submodule():
    assert "models" in tr.__all__
    assert tr.models.TabularMLP is TabularMLP


@pytest.mark.parametrize("epochs", [1, 2, 5, 10])
def test_onecycle_short_schedules(epochs: int) -> None:
    """epochs * steps_per_epoch around 10 used to divide by zero in OneCycleLR."""
    from torchregress.losses import WeightedMSELoss
    from torchregress.models import TabularMLP, fit_tabular

    gen = torch.Generator().manual_seed(0)
    X = torch.randn(40, 3, generator=gen)
    y = X[:, :1] + 0.1 * torch.randn(40, 1, generator=gen)
    model = TabularMLP(3, 1, hidden=(16,))
    fit = fit_tabular(model, WeightedMSELoss(), X, y, epochs=epochs, batch_size=32, patience=50)
    assert torch.isfinite(fit.predict(X)).all()
