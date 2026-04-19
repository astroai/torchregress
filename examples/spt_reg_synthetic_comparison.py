"""Synthetic competing-method benchmark for Shift-Factored Predictive Transport."""

import argparse
from dataclasses import dataclass
from math import erf
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from comparison_utils import (
    compute_point_metrics,
    print_comparison_summary,
    print_fairness_notes,
    timed_call,
    write_comparison_summary_json,
)
from torch.utils.data import DataLoader, TensorDataset

from torchregress.inference import PPIConfig, ppi_mean_ci, ppi_quantile_ci
from torchregress.losses import MDNLoss
from torchregress.metrics import (
    crps_from_samples,
    prediction_interval_coverage_probability,
    risk_coverage_curve,
)
from torchregress.prediction import PredictiveBatch
from torchregress.test_time import (
    FeatureStatNormalizer,
    RepresentationShiftCalibrator,
    ShiftFactoredPredictiveTransport,
    ShiftFactoredTransportConfig,
    SignificantSubspaceAligner,
)


@dataclass(frozen=True)
class SPTRegSyntheticConfig:
    seed: int = 260409
    n_source: int = 320
    n_target_unlabeled: int = 192
    n_target_cal: int = 64
    n_target_test: int = 128
    alpha: float = 0.1
    ppi_quantile: float = 0.9
    n_support: int = 128
    n_bins: int = 20
    n_samples_eval: int = 64
    target_label_budget: int = 64
    mdn_hidden: int = 48
    mdn_components: int = 5
    mdn_epochs: int = 32
    mdn_batch_size: int = 64
    mdn_lr: float = 3.0e-3
    mdn_predict_samples: int = 192


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _source_x(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.normal(size=(n, 4))


def _target_x(rng: np.random.Generator, n: int) -> np.ndarray:
    z = rng.normal(size=(n, 4))
    x = np.empty_like(z)
    x[:, 0] = 1.25 * z[:, 0] + 0.75
    x[:, 1] = 0.85 * z[:, 1] - 0.45
    x[:, 2] = 0.55 * z[:, 0] + 0.75 * z[:, 2]
    x[:, 3] = z[:, 3] + 0.6
    return x


def _truth_mean(x: np.ndarray) -> np.ndarray:
    return 0.9 * x[:, 0] - 0.7 * x[:, 1] + 0.45 * x[:, 0] * x[:, 2] + 0.25 * np.sin(2.0 * x[:, 3])


def _truth_std(x: np.ndarray, *, scale: float) -> np.ndarray:
    return scale * (0.18 + 0.07 * np.abs(x[:, 0]) + 0.05 * np.clip(x[:, 2], 0.0, None))


def _sample_targets(
    rng: np.random.Generator,
    x: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    mean = _truth_mean(x)
    std = _truth_std(x, scale=scale)
    return mean + rng.normal(scale=std, size=mean.shape[0])


def _add_intercept(x: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ones((x.shape[0], 1), dtype=float), x], axis=1)


def _fit_linear_gaussian(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    design = _add_intercept(x)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ beta
    sigma = float(np.sqrt(np.mean((y - pred) ** 2)))
    return beta, max(sigma, 1.0e-4)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf, otypes=[np.float64])(z / np.sqrt(2.0)))


class SyntheticPredictor:
    def __init__(self, beta: np.ndarray, sigma: float, bin_edges: np.ndarray) -> None:
        self.beta = np.asarray(beta, dtype=float)
        self.sigma = float(sigma)
        self.bin_edges = np.asarray(bin_edges, dtype=float)
        self.quantile_levels = [0.1, 0.5, 0.9]

    def _mean_std(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        design = _add_intercept(np.asarray(x, dtype=float))
        mean = design @ self.beta
        std = self.sigma * (1.0 + 0.12 * np.abs(x[:, 0]))
        return mean.astype(np.float32), std.astype(np.float32)

    def predict_distribution(self, x: np.ndarray, family: str = "gaussian") -> PredictiveBatch:
        mean, std = self._mean_std(x)
        if family == "gaussian":
            return PredictiveBatch(mean=mean, std=std)
        if family == "quantile":
            z = np.array([-1.2815515655446004, 0.0, 1.2815515655446004], dtype=float)
            quantiles = mean[:, None] + std[:, None] * z[None, :]
            return PredictiveBatch(
                quantiles=quantiles.astype(np.float32),
                quantile_levels=list(self.quantile_levels),
            )
        if family in {"bar", "binnedpdf"}:
            z = (self.bin_edges[None, :] - mean[:, None]) / np.clip(std[:, None], 1.0e-8, None)
            cdf = _normal_cdf(z)
            probs = np.diff(cdf, axis=1)
            probs = np.clip(probs, 1.0e-8, None)
            probs = probs / probs.sum(axis=1, keepdims=True)
            return PredictiveBatch(
                bar_logits=np.log(probs).astype(np.float32),
                bin_edges=self.bin_edges.astype(np.float32),
                extra={"family_label": "BinnedPDF" if family == "binnedpdf" else "Bar"},
            )
        raise ValueError(f"Unsupported family: {family}")


class _MDNBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_components: int) -> None:
        super().__init__()
        out_dim = n_components + 2 * n_components
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MDNSyntheticPredictor:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: MDNLoss,
        *,
        n_predict_samples: int,
    ) -> None:
        self.model = model.eval()
        self.loss_fn = loss_fn
        self.n_predict_samples = int(n_predict_samples)

    def predict_distribution(self, x: np.ndarray, family: str = "mdn") -> PredictiveBatch:
        del family
        x_tensor = torch.tensor(np.asarray(x, dtype=np.float32))
        with torch.no_grad():
            raw = self.model(x_tensor)
            mean, std = self.loss_fn.predict_mean_std(raw)
            samples = self.loss_fn.sample(raw, n_samples=self.n_predict_samples)
        mean_np = mean.squeeze(-1).cpu().numpy().astype(np.float32)
        std_np = std.squeeze(-1).cpu().numpy().astype(np.float32)
        samples_np = samples.squeeze(-1).transpose(0, 1).cpu().numpy().astype(np.float32)
        return PredictiveBatch(
            point=mean_np,
            mean=mean_np,
            std=std_np,
            samples=samples_np,
            extra={"family": "mdn"},
        )


def _fit_mdn_predictor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    cfg: SPTRegSyntheticConfig,
) -> MDNSyntheticPredictor:
    torch.manual_seed(cfg.seed)
    model = _MDNBackbone(
        in_dim=x_train.shape[1],
        hidden=cfg.mdn_hidden,
        n_components=cfg.mdn_components,
    )
    loss_fn = MDNLoss(
        n_components=cfg.mdn_components,
        n_features=1,
        covariance_type="diagonal",
    )
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train[:, None], dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=min(cfg.mdn_batch_size, x_train.shape[0]),
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.mdn_lr)
    for _ in range(cfg.mdn_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    return MDNSyntheticPredictor(
        model,
        loss_fn,
        n_predict_samples=cfg.mdn_predict_samples,
    )


def _dense_batch(batch: PredictiveBatch) -> PredictiveBatch:
    if batch.support is not None and batch.density is not None:
        return batch
    if batch.bar_logits is not None and batch.bin_edges is not None:
        return batch.with_density(n_support=128)
    if batch.quantiles is not None and batch.quantile_levels is not None:
        return batch.with_density(n_support=128)
    if batch.samples is not None:
        return batch.with_density(n_support=128)
    return batch


def _batch_len(batch: PredictiveBatch) -> int:
    for value in (
        batch.mean,
        batch.point,
        batch.std,
        batch.quantiles,
        batch.bar_logits,
        batch.density,
        batch.samples,
    ):
        if value is not None:
            arr = np.asarray(value)
            return int(arr.shape[0])
    raise ValueError("could not infer batch size")


def _slice_batch(batch: PredictiveBatch, start: int, stop: int) -> PredictiveBatch:
    n = _batch_len(batch)

    def _slice(value: Any) -> Any:
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == n:
            return arr[start:stop]
        return value

    return PredictiveBatch(
        point=_slice(batch.point),
        mean=_slice(batch.mean),
        std=_slice(batch.std),
        quantiles=_slice(batch.quantiles),
        quantile_levels=batch.quantile_levels,
        bar_logits=_slice(batch.bar_logits),
        bin_edges=_slice(batch.bin_edges),
        samples=_slice(batch.samples),
        support=_slice(batch.support),
        density=_slice(batch.density),
        extra=dict(batch.extra or {}),
    )


def _batch_mean_std(batch: PredictiveBatch) -> tuple[np.ndarray, np.ndarray]:
    batch = _dense_batch(batch)
    if batch.mean is not None and batch.std is not None:
        return np.asarray(batch.mean, dtype=float).reshape(-1), np.asarray(
            batch.std, dtype=float
        ).reshape(-1)
    if batch.support is None or batch.density is None:
        raise ValueError("batch must expose either mean/std or support/density")
    support = np.asarray(batch.support, dtype=float)
    density = np.asarray(batch.density, dtype=float)
    if support.ndim == 1:
        dx = max(float(np.mean(np.diff(support))), 1.0e-8)
        probs = density * dx
        probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1.0e-8, None)
        mean = probs @ support
        second = probs @ (support**2)
        std = np.sqrt(np.clip(second - mean**2, 1.0e-8, None))
        return mean, std
    dx = np.clip(np.mean(np.diff(support, axis=1), axis=1, keepdims=True), 1.0e-8, None)
    probs = density * dx
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1.0e-8, None)
    mean = np.sum(probs * support, axis=1)
    second = np.sum(probs * (support**2), axis=1)
    std = np.sqrt(np.clip(second - mean**2, 1.0e-8, None))
    return mean, std


def _batch_quantiles(
    batch: PredictiveBatch, levels: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    batch = _dense_batch(batch)
    if batch.quantiles is not None and batch.quantile_levels is not None:
        known = np.asarray(batch.quantile_levels, dtype=float)
        values = np.asarray(batch.quantiles, dtype=float)
        lower = np.stack(
            [np.interp(levels[0], known, values[i]) for i in range(values.shape[0])], axis=0
        )
        upper = np.stack(
            [np.interp(levels[1], known, values[i]) for i in range(values.shape[0])], axis=0
        )
        return lower, upper
    if batch.support is None or batch.density is None:
        mean, std = _batch_mean_std(batch)
        z = 1.6448536269514722
        return mean - z * std, mean + z * std
    support = np.asarray(batch.support, dtype=float)
    density = np.asarray(batch.density, dtype=float)
    if support.ndim == 1:
        dx = max(float(np.mean(np.diff(support))), 1.0e-8)
        probs = density * dx
        probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1.0e-8, None)
        cdf = np.cumsum(probs, axis=1)
        cdf[:, -1] = 1.0
        lower = np.stack(
            [np.interp(levels[0], cdf[i], support) for i in range(cdf.shape[0])], axis=0
        )
        upper = np.stack(
            [np.interp(levels[1], cdf[i], support) for i in range(cdf.shape[0])], axis=0
        )
        return lower, upper
    dx = np.clip(np.mean(np.diff(support, axis=1), axis=1, keepdims=True), 1.0e-8, None)
    probs = density * dx
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1.0e-8, None)
    cdf = np.cumsum(probs, axis=1)
    cdf[:, -1] = 1.0
    lower = np.stack(
        [np.interp(levels[0], cdf[i], support[i]) for i in range(cdf.shape[0])], axis=0
    )
    upper = np.stack(
        [np.interp(levels[1], cdf[i], support[i]) for i in range(cdf.shape[0])], axis=0
    )
    return lower, upper


def _batch_samples(batch: PredictiveBatch, n_samples: int, seed: int) -> np.ndarray:
    batch = _dense_batch(batch)
    rng = _rng(seed)
    if batch.support is not None and batch.density is not None:
        support = np.asarray(batch.support, dtype=float)
        density = np.asarray(batch.density, dtype=float)
        if support.ndim == 1:
            dx = max(float(np.mean(np.diff(support))), 1.0e-8)
            probs = density * dx
            probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1.0e-8, None)
            out = np.empty((n_samples, probs.shape[0]), dtype=float)
            idx = np.arange(support.shape[0], dtype=int)
            for row in range(probs.shape[0]):
                out[:, row] = support[rng.choice(idx, size=n_samples, replace=True, p=probs[row])]
            return out
        dx = np.clip(np.mean(np.diff(support, axis=1), axis=1, keepdims=True), 1.0e-8, None)
        probs = density * dx
        probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1.0e-8, None)
        out = np.empty((n_samples, probs.shape[0]), dtype=float)
        for row in range(probs.shape[0]):
            idx = np.arange(support.shape[1], dtype=int)
            out[:, row] = support[row, rng.choice(idx, size=n_samples, replace=True, p=probs[row])]
        return out
    mean, std = _batch_mean_std(batch)
    return rng.normal(loc=mean[None, :], scale=std[None, :], size=(n_samples, mean.shape[0]))


def _batch_log_density(batch: PredictiveBatch, targets: np.ndarray) -> np.ndarray:
    batch = _dense_batch(batch)
    y = np.asarray(targets, dtype=float).reshape(-1)
    if batch.support is not None and batch.density is not None:
        support = np.asarray(batch.support, dtype=float)
        density = np.asarray(batch.density, dtype=float)
        if support.ndim == 1:
            vals = np.array(
                [
                    np.interp(y[i], support, density[i], left=1.0e-8, right=1.0e-8)
                    for i in range(y.shape[0])
                ],
                dtype=float,
            )
        else:
            vals = np.array(
                [
                    np.interp(y[i], support[i], density[i], left=1.0e-8, right=1.0e-8)
                    for i in range(y.shape[0])
                ],
                dtype=float,
            )
        return np.log(np.clip(vals, 1.0e-8, None))
    mean, std = _batch_mean_std(batch)
    var = np.clip(std**2, 1.0e-8, None)
    return -0.5 * (np.log(2.0 * np.pi * var) + (y - mean) ** 2 / var)


def _interval_from_batch(batch: PredictiveBatch, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    extra = batch.extra or {}
    if "interval_lower" in extra and "interval_upper" in extra:
        return np.asarray(extra["interval_lower"], dtype=float), np.asarray(
            extra["interval_upper"], dtype=float
        )
    return _batch_quantiles(batch, (alpha / 2.0, 1.0 - alpha / 2.0))


def _tail_rmse(y_pred: np.ndarray, y_true: np.ndarray, q: float = 0.9) -> float:
    threshold = float(np.quantile(y_true, q))
    mask = y_true >= threshold
    return float(np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2)))


def _ppi_summary(
    y_cal: np.ndarray,
    batch_cal: PredictiveBatch,
    y_test: np.ndarray,
    batch_test: PredictiveBatch,
    *,
    q: float,
    alpha: float,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    pred_cal, _ = _batch_mean_std(batch_cal)
    pred_test, _ = _batch_mean_std(batch_test)
    config = PPIConfig(alpha=alpha, n_boot=n_boot, seed=seed)
    mean_ci = ppi_mean_ci(
        torch.tensor(y_cal, dtype=torch.float32),
        torch.tensor(pred_cal, dtype=torch.float32),
        torch.tensor(pred_test, dtype=torch.float32),
        config=config,
    )
    q_cal = _batch_quantiles(batch_cal, (q, q))[0]
    q_test = _batch_quantiles(batch_test, (q, q))[0]
    quantile_ci = ppi_quantile_ci(
        torch.tensor(y_cal, dtype=torch.float32),
        torch.tensor(q_cal, dtype=torch.float32),
        torch.tensor(q_test, dtype=torch.float32),
        q=q,
        config=config,
    )
    true_mean = float(np.mean(y_test))
    true_q = float(np.quantile(y_test, q))
    return {
        "PPIMeanCIWidth": float(mean_ci["ci_upper"]) - float(mean_ci["ci_lower"]),
        "PPIMeanCICovers": float(
            float(mean_ci["ci_lower"]) <= true_mean <= float(mean_ci["ci_upper"])
        ),
        "PPIQuantileCIWidth": float(quantile_ci["ci_upper"]) - float(quantile_ci["ci_lower"]),
        "PPIQuantileCICovers": float(
            float(quantile_ci["ci_lower"]) <= true_q <= float(quantile_ci["ci_upper"])
        ),
    }


def _evaluate_row(
    *,
    method: str,
    family: str,
    batch_cal: PredictiveBatch,
    batch_test: PredictiveBatch,
    y_cal: np.ndarray,
    y_test: np.ndarray,
    cfg: SPTRegSyntheticConfig,
    train_s: float,
    eval_s: float,
    notes: str,
) -> dict[str, object]:
    mean, std = _batch_mean_std(batch_test)
    lower, upper = _interval_from_batch(batch_test, cfg.alpha)
    samples = _batch_samples(batch_test, cfg.n_samples_eval, cfg.seed + len(method))
    point = compute_point_metrics(
        torch.tensor(mean[:, None], dtype=torch.float32),
        torch.tensor(y_test[:, None], dtype=torch.float32),
    )
    rc = risk_coverage_curve(
        torch.tensor(mean, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
        torch.tensor(std, dtype=torch.float32),
        n_points=32,
    )
    coverage = prediction_interval_coverage_probability(
        torch.tensor(lower[:, None], dtype=torch.float32),
        torch.tensor(upper[:, None], dtype=torch.float32),
        torch.tensor(y_test[:, None], dtype=torch.float32),
        alpha=cfg.alpha,
        return_diagnostics=True,
    )
    ppi = _ppi_summary(
        y_cal,
        batch_cal,
        y_test,
        batch_test,
        q=cfg.ppi_quantile,
        alpha=cfg.alpha,
        n_boot=200,
        seed=cfg.seed,
    )
    return {
        "Method": method,
        "Family": family,
        **point,
        "TailRMSE90": _tail_rmse(mean, y_test),
        "NLL": float(-np.mean(_batch_log_density(batch_test, y_test))),
        "CRPS": float(
            crps_from_samples(
                torch.tensor(samples[:, :, None], dtype=torch.float32),
                torch.tensor(y_test[:, None], dtype=torch.float32),
            )
        ),
        "Cov90": float(coverage["picp"]),
        "Width90": float(coverage["mpiw"]),
        "AURC": float(rc["aurc"]),
        **ppi,
        "train_s": float(train_s),
        "eval_s": float(eval_s),
        "Notes": notes,
    }


def _manual_split_conformal(
    batch_cal: PredictiveBatch,
    y_cal: np.ndarray,
    batch_test: PredictiveBatch,
    alpha: float,
) -> PredictiveBatch:
    mean_cal, std_cal = _batch_mean_std(batch_cal)
    scores = np.abs(y_cal - mean_cal) / np.clip(std_cal, 1.0e-8, None)
    q_hat = float(
        np.quantile(
            scores,
            min(np.ceil((scores.size + 1) * (1.0 - alpha)) / scores.size, 1.0),
            method="higher",
        )
    )
    mean_test, std_test = _batch_mean_std(batch_test)
    extra = dict(batch_test.extra or {})
    extra["interval_lower"] = (mean_test - q_hat * std_test).tolist()
    extra["interval_upper"] = (mean_test + q_hat * std_test).tolist()
    extra["conformal_method"] = "split"
    return PredictiveBatch(
        point=batch_test.point,
        mean=batch_test.mean,
        std=batch_test.std,
        quantiles=batch_test.quantiles,
        quantile_levels=batch_test.quantile_levels,
        bar_logits=batch_test.bar_logits,
        bin_edges=batch_test.bin_edges,
        samples=batch_test.samples,
        support=batch_test.support,
        density=batch_test.density,
        extra=extra,
    )


def _covariate_density_ratio_weights(
    x_source: np.ndarray,
    x_target: np.ndarray,
    x_query: np.ndarray,
    *,
    seed: int,
    max_fit_rows: int = 8000,
) -> np.ndarray:
    """Estimate \\hat w(x) \\propto p_{\\mathrm{tgt}}(x) / p_{\\mathrm{src}}(x) from a logistic discriminator.

    Uses a scaled logistic regression (Tibshirani-style density-ratio heuristic for weighted CP).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    xs = np.asarray(x_source, dtype=np.float64)
    xt = np.asarray(x_target, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    half = max(256, max_fit_rows // 2)
    if xs.shape[0] > half:
        xs = xs[rng.choice(xs.shape[0], half, replace=False)]
    if xt.shape[0] > half:
        xt = xt[rng.choice(xt.shape[0], half, replace=False)]
    x_fit = np.vstack([xs, xt])
    y_fit = np.concatenate(
        [np.zeros(xs.shape[0], dtype=np.int64), np.ones(xt.shape[0], dtype=np.int64)]
    )
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=int(seed) % (2**31),
            solver="lbfgs",
        ),
    )
    clf.fit(x_fit, y_fit)
    xq = np.asarray(x_query, dtype=np.float64)
    p_t = clf.predict_proba(xq)[:, 1]
    p_s = np.clip(1.0 - p_t, 1e-6, 1.0 - 1e-6)
    p_t = np.clip(p_t, 1e-6, 1.0 - 1e-6)
    pi_t = float(xt.shape[0] / (xs.shape[0] + xt.shape[0]))
    pi_s = 1.0 - pi_t
    odds = p_t / p_s
    w = odds * (pi_s / max(pi_t, 1e-12))
    return np.clip(w, 1e-3, 1e3).astype(np.float64)


def _weighted_split_conformal(
    batch_cal: PredictiveBatch,
    y_cal: np.ndarray,
    batch_test: PredictiveBatch,
    alpha: float,
    weights_cal: np.ndarray,
) -> PredictiveBatch:
    """Split conformal with nonnegative calibration weights (weighted residual quantile)."""
    mean_cal, std_cal = _batch_mean_std(batch_cal)
    scores = np.abs(np.asarray(y_cal, dtype=np.float64).ravel() - mean_cal) / np.clip(
        std_cal, 1.0e-8, None
    )
    w = np.asarray(weights_cal, dtype=np.float64).ravel()
    if w.shape[0] != scores.shape[0]:
        raise ValueError("weights_cal must match number of calibration scores")
    w = np.clip(w, 1e-12, None)
    w = w / float(np.sum(w))
    order = np.argsort(scores)
    s_sorted = scores[order]
    w_sorted = w[order]
    cum = np.cumsum(w_sorted)
    level = min(float(np.ceil((scores.size + 1) * (1.0 - alpha)) / max(scores.size, 1)), 1.0)
    idx = int(np.searchsorted(cum, level, side="left"))
    idx = min(max(idx, 0), s_sorted.size - 1)
    q_hat = float(s_sorted[idx])
    mean_test, std_test = _batch_mean_std(batch_test)
    extra = dict(batch_test.extra or {})
    extra["interval_lower"] = (mean_test - q_hat * std_test).tolist()
    extra["interval_upper"] = (mean_test + q_hat * std_test).tolist()
    extra["conformal_method"] = "weighted_split"
    return PredictiveBatch(
        point=batch_test.point,
        mean=batch_test.mean,
        std=batch_test.std,
        quantiles=batch_test.quantiles,
        quantile_levels=batch_test.quantile_levels,
        bar_logits=batch_test.bar_logits,
        bin_edges=batch_test.bin_edges,
        samples=batch_test.samples,
        support=batch_test.support,
        density=batch_test.density,
        extra=extra,
    )


def _refit_batch(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
) -> PredictiveBatch:
    """Ridge when n_samples is small vs dimension (stabilizes Year high-d + small cal)."""
    design = _add_intercept(np.asarray(x_train, dtype=float))
    y = np.asarray(y_train, dtype=float).reshape(-1)
    n, d = design.shape
    if n < d + 8:
        lam = 1e-2 * float(n)
        reg = lam * np.eye(d, dtype=float)
        reg[0, 0] = 0.0
        beta = np.linalg.solve(design.T @ design + reg, design.T @ y)
    else:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ beta
    sigma = float(np.sqrt(np.mean((y - pred) ** 2)))
    sigma = max(sigma, 1.0e-2)
    predictor = SyntheticPredictor(
        beta, sigma, np.linspace(float(y.min()) - 1.0, float(y.max()) + 1.0, 10)
    )
    return predictor.predict_distribution(np.asarray(x_eval, dtype=float), family="gaussian")


def run_comparison(cfg: SPTRegSyntheticConfig) -> tuple[list[dict[str, object]], list[str]]:
    rng = _rng(cfg.seed)
    source_x = _source_x(rng, cfg.n_source)
    source_y = _sample_targets(rng, source_x, scale=1.0)

    n_pool = cfg.n_target_unlabeled + cfg.n_target_cal + cfg.n_target_test
    target_pool_x = _target_x(rng, n_pool)
    target_pool_y = _sample_targets(rng, target_pool_x, scale=1.35)

    unlabeled_stop = cfg.n_target_unlabeled
    cal_stop = unlabeled_stop + cfg.n_target_cal
    y_cal = target_pool_y[unlabeled_stop:cal_stop]
    y_test = target_pool_y[cal_stop:]
    x_cal = target_pool_x[unlabeled_stop:cal_stop]
    x_test = target_pool_x[cal_stop:]

    fit_out, fit_s = timed_call(_fit_linear_gaussian, source_x, source_y)
    beta, sigma = fit_out
    bin_edges = np.quantile(source_y, np.linspace(0.0, 1.0, cfg.n_bins + 1))
    bin_edges = np.unique(bin_edges)
    if bin_edges.size < 3:
        bin_edges = np.linspace(source_y.min() - 1.0, source_y.max() + 1.0, cfg.n_bins + 1)
    predictor = SyntheticPredictor(beta, sigma, bin_edges)
    mdn_predictor, mdn_fit_s = timed_call(_fit_mdn_predictor, source_x, source_y, cfg)

    source_gaussian_pool = predictor.predict_distribution(target_pool_x, family="gaussian")
    source_binned_pool = predictor.predict_distribution(target_pool_x, family="binnedpdf")
    source_mdn_pool = mdn_predictor.predict_distribution(target_pool_x)
    source_gaussian_cal = _slice_batch(source_gaussian_pool, unlabeled_stop, cal_stop)
    source_gaussian_test = _slice_batch(source_gaussian_pool, cal_stop, n_pool)
    source_binned_cal = _slice_batch(source_binned_pool, unlabeled_stop, cal_stop)
    source_binned_test = _slice_batch(source_binned_pool, cal_stop, n_pool)
    source_mdn_cal = _slice_batch(source_mdn_pool, unlabeled_stop, cal_stop)
    source_mdn_test = _slice_batch(source_mdn_pool, cal_stop, n_pool)

    rows: list[dict[str, object]] = []

    rows.append(
        _evaluate_row(
            method="SourceGaussian",
            family="Gaussian",
            batch_cal=source_gaussian_cal,
            batch_test=source_gaussian_test,
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            train_s=fit_s,
            eval_s=0.0,
            notes="source linear-Gaussian predictor with no target adaptation",
        )
    )

    normalizer = FeatureStatNormalizer().fit(source_x)
    norm_pool = predictor.predict_distribution(
        normalizer.transform(target_pool_x), family="gaussian"
    )
    rows.append(
        _evaluate_row(
            method="FeatureStatNormGaussian",
            family="Gaussian",
            batch_cal=_slice_batch(norm_pool, unlabeled_stop, cal_stop),
            batch_test=_slice_batch(norm_pool, cal_stop, n_pool),
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            train_s=fit_s,
            eval_s=0.0,
            notes="feature-stat normalization before rerunning the source predictor",
        )
    )

    aligner = SignificantSubspaceAligner(rank=2, random_state=cfg.seed).fit(source_x, source_y)
    aligned_pool = predictor.predict_distribution(
        aligner.transform(target_pool_x), family="gaussian"
    )
    rows.append(
        _evaluate_row(
            method="SignificantSubspaceGaussian",
            family="Gaussian",
            batch_cal=_slice_batch(aligned_pool, unlabeled_stop, cal_stop),
            batch_test=_slice_batch(aligned_pool, cal_stop, n_pool),
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            train_s=fit_s,
            eval_s=0.0,
            notes="significance-weighted subspace alignment only",
        )
    )

    prior_transport = ShiftFactoredPredictiveTransport(
        ShiftFactoredTransportConfig(
            n_support=cfg.n_support,
            alpha=cfg.alpha,
            enable_alignment=False,
            enable_uncertainty_inflation=False,
            random_state=cfg.seed,
        )
    ).fit_source(
        predictor.predict_distribution(source_x, family="gaussian"),
        source_y,
        source_inputs=source_x,
    )
    prior_pool, prior_eval_s = timed_call(
        prior_transport.adapt_unlabeled_target,
        target_predictions=source_gaussian_pool,
        target_inputs=target_pool_x,
    )
    rows.append(
        _evaluate_row(
            method="PriorTransportGaussian",
            family="Gaussian",
            batch_cal=_slice_batch(prior_pool, unlabeled_stop, cal_stop),
            batch_test=_slice_batch(prior_pool, cal_stop, n_pool),
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            train_s=fit_s,
            eval_s=prior_eval_s,
            notes="output-space prior transport only",
        )
    )

    shift_calibrator = RepresentationShiftCalibrator(random_state=cfg.seed).fit(source_x)
    unc_mean, unc_std = _batch_mean_std(source_gaussian_pool)
    unc_pool = PredictiveBatch(
        mean=unc_mean.astype(np.float32),
        std=shift_calibrator.calibrate_std(unc_std, target_pool_x).astype(np.float32),
    )
    rows.append(
        _evaluate_row(
            method="UncertaintyOnlyGaussian",
            family="Gaussian",
            batch_cal=_slice_batch(unc_pool, unlabeled_stop, cal_stop),
            batch_test=_slice_batch(unc_pool, cal_stop, n_pool),
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            train_s=fit_s,
            eval_s=0.0,
            notes="representation-shift variance inflation only",
        )
    )

    raw_split = _manual_split_conformal(source_gaussian_cal, y_cal, source_gaussian_test, cfg.alpha)
    rows.append(
        _evaluate_row(
            method="RawSplitConformalGaussian",
            family="Gaussian",
            batch_cal=source_gaussian_cal,
            batch_test=raw_split,
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            train_s=fit_s,
            eval_s=0.0,
            notes="raw source predictions with split conformal intervals on target labels",
        )
    )

    w_cal = _covariate_density_ratio_weights(
        source_x,
        target_pool_x[:unlabeled_stop],
        x_cal,
        seed=cfg.seed,
    )
    weighted_split = _weighted_split_conformal(
        source_gaussian_cal, y_cal, source_gaussian_test, cfg.alpha, w_cal
    )
    rows.append(
        _evaluate_row(
            method="WeightedSplitConformalGaussian",
            family="Gaussian",
            batch_cal=source_gaussian_cal,
            batch_test=weighted_split,
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            train_s=fit_s,
            eval_s=0.0,
            notes=(
                "source Gaussian + covariate-weighted split conformal "
                "(logistic density-ratio on target-unlabeled vs source)"
            ),
        )
    )

    spt = ShiftFactoredPredictiveTransport(
        ShiftFactoredTransportConfig(
            n_support=cfg.n_support,
            alpha=cfg.alpha,
            random_state=cfg.seed,
        )
    ).fit_source(
        predictor.predict_distribution(source_x, family="gaussian"),
        source_y,
        source_inputs=source_x,
    )
    spt_pool, spt_eval_s = timed_call(
        spt.adapt_unlabeled_target,
        target_predictions=source_gaussian_pool,
        target_inputs=target_pool_x,
        predictor=predictor,
    )
    spt_cal = _slice_batch(spt_pool, unlabeled_stop, cal_stop)
    spt_test = _slice_batch(spt_pool, cal_stop, n_pool)
    rows.append(
        _evaluate_row(
            method="SPTTransportGaussian",
            family="Gaussian",
            batch_cal=spt_cal,
            batch_test=spt_test,
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            train_s=fit_s,
            eval_s=spt_eval_s,
            notes="SPT adaptation without conformal wrapping (isolates transport + inflation path)",
        )
    )
    _, spt_cal_s = timed_call(spt.calibrate_target, spt_cal, y_cal)
    spt_test_conf = spt.apply_conformal(spt_test)
    rows.append(
        _evaluate_row(
            method="SPTRegGaussian",
            family="Gaussian",
            batch_cal=spt_cal,
            batch_test=spt_test_conf,
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            train_s=fit_s + spt_cal_s,
            eval_s=spt_eval_s,
            notes="full shift-factored predictive transport with conformal wrapping",
        )
    )

    small_batch, small_s = timed_call(
        _refit_batch, x_cal[: cfg.target_label_budget], y_cal[: cfg.target_label_budget], x_test
    )
    rows.append(
        _evaluate_row(
            method="TargetRefitSmallGaussian",
            family="Gaussian",
            batch_cal=_refit_batch(
                x_cal[: cfg.target_label_budget], y_cal[: cfg.target_label_budget], x_cal
            ),
            batch_test=small_batch,
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            train_s=small_s,
            eval_s=0.0,
            notes="small target-label refit baseline using only calibration-budget labels",
        )
    )

    oracle_x = target_pool_x[:cal_stop]
    oracle_y = target_pool_y[:cal_stop]
    oracle_batch, oracle_s = timed_call(_refit_batch, oracle_x, oracle_y, x_test)
    rows.append(
        _evaluate_row(
            method="TargetRefitOracleGaussian",
            family="Gaussian",
            batch_cal=_refit_batch(oracle_x, oracle_y, x_cal),
            batch_test=oracle_batch,
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            train_s=oracle_s,
            eval_s=0.0,
            notes="oracle target retraining baseline using all non-test target labels",
        )
    )

    rows.append(
        _evaluate_row(
            method="SourceBinnedPDF",
            family="BinnedPDF",
            batch_cal=source_binned_cal,
            batch_test=source_binned_test,
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            train_s=fit_s,
            eval_s=0.0,
            notes="ordered-bin predictive law without target adaptation",
        )
    )

    rows.append(
        _evaluate_row(
            method="SourceMDN",
            family="MDN",
            batch_cal=source_mdn_cal,
            batch_test=source_mdn_test,
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            train_s=mdn_fit_s,
            eval_s=0.0,
            notes="source MDN predictive law evaluated via Monte Carlo samples",
        )
    )

    raw_mdn_transport = ShiftFactoredPredictiveTransport(
        ShiftFactoredTransportConfig(
            n_support=cfg.n_support,
            alpha=cfg.alpha,
            random_state=cfg.seed,
        )
    ).fit_source(
        mdn_predictor.predict_distribution(source_x),
        source_y,
        source_inputs=source_x,
    )
    raw_mdn_cal_dense = source_mdn_cal.with_density(n_support=cfg.n_support)
    raw_mdn_test_dense = source_mdn_test.with_density(n_support=cfg.n_support)
    _, raw_mdn_cal_s = timed_call(raw_mdn_transport.calibrate_target, raw_mdn_cal_dense, y_cal)
    raw_mdn_test_conf = raw_mdn_transport.apply_conformal(raw_mdn_test_dense)
    rows.append(
        _evaluate_row(
            method="RawConformalMDN",
            family="MDN",
            batch_cal=source_mdn_cal,
            batch_test=raw_mdn_test_conf,
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            train_s=mdn_fit_s + raw_mdn_cal_s,
            eval_s=0.0,
            notes="source MDN with conformal only (no prior transport; family-matched score routing)",
        )
    )

    spt_binned = ShiftFactoredPredictiveTransport(
        ShiftFactoredTransportConfig(
            n_support=cfg.n_support,
            alpha=cfg.alpha,
            random_state=cfg.seed,
        )
    ).fit_source(
        predictor.predict_distribution(source_x, family="binnedpdf"),
        source_y,
        source_inputs=source_x,
    )
    spt_binned_pool, spt_binned_eval_s = timed_call(
        spt_binned.adapt_unlabeled_target,
        target_predictions=source_binned_pool,
        target_inputs=target_pool_x,
        predictor=None,
    )
    spt_binned_cal = _slice_batch(spt_binned_pool, unlabeled_stop, cal_stop)
    spt_binned_test = _slice_batch(spt_binned_pool, cal_stop, n_pool)
    _, spt_binned_cal_s = timed_call(spt_binned.calibrate_target, spt_binned_cal, y_cal)
    rows.append(
        _evaluate_row(
            method="SPTRegBinnedPDF",
            family="BinnedPDF",
            batch_cal=spt_binned_cal,
            batch_test=spt_binned.apply_conformal(spt_binned_test),
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            train_s=fit_s + spt_binned_cal_s,
            eval_s=spt_binned_eval_s,
            notes="SPT-Reg applied to an ordered-bin predictive law",
        )
    )

    spt_mdn = ShiftFactoredPredictiveTransport(
        ShiftFactoredTransportConfig(
            n_support=cfg.n_support,
            alpha=cfg.alpha,
            random_state=cfg.seed,
        )
    ).fit_source(
        mdn_predictor.predict_distribution(source_x),
        source_y,
        source_inputs=source_x,
    )
    spt_mdn_pool, spt_mdn_eval_s = timed_call(
        spt_mdn.adapt_unlabeled_target,
        target_predictions=source_mdn_pool,
        target_inputs=target_pool_x,
        predictor=None,
    )
    spt_mdn_cal = _slice_batch(spt_mdn_pool, unlabeled_stop, cal_stop)
    spt_mdn_test = _slice_batch(spt_mdn_pool, cal_stop, n_pool)
    rows.append(
        _evaluate_row(
            method="SPTTransportMDN",
            family="MDN",
            batch_cal=spt_mdn_cal,
            batch_test=spt_mdn_test,
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            train_s=mdn_fit_s,
            eval_s=spt_mdn_eval_s,
            notes="SPT prior transport on MDN predictive law without conformal wrapping",
        )
    )
    _, spt_mdn_cal_s = timed_call(spt_mdn.calibrate_target, spt_mdn_cal, y_cal)
    rows.append(
        _evaluate_row(
            method="SPTRegMDN",
            family="MDN",
            batch_cal=spt_mdn_cal,
            batch_test=spt_mdn.apply_conformal(spt_mdn_test),
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            train_s=mdn_fit_s + spt_mdn_cal_s,
            eval_s=spt_mdn_eval_s,
            notes="SPT-Reg applied to a sampled MDN predictive law",
        )
    )

    notes = [
        "Synthetic target shift combines predictive-subspace covariate drift and residual variance inflation.",
        "Competing-method rows use a shared source linear-Gaussian backbone and the same target calibration budget.",
        "BinnedPDF rows reuse the same source backbone but expose the predictive law as ordered-bin probabilities.",
        "MDN rows use source-trained mixture density predictions transported through the same support-grid SPT operator.",
        "RawConformalMDN applies conformal scores to the source MDN without running prior transport.",
        "SPTTransportGaussian / SPTTransportMDN report post-adaptation laws before conformal widening.",
    ]
    return rows, notes


def main(
    cfg: SPTRegSyntheticConfig | None = None,
    summary_json_path: str | None = None,
) -> None:
    cfg = cfg or SPTRegSyntheticConfig()
    rows, notes = run_comparison(cfg)

    print_fairness_notes(
        title="SPT-Reg synthetic competing-method comparison",
        seed_policy=f"fixed seed = {cfg.seed}",
        train_budget="shared closed-form source backbone and matched target-label budgets",
        metric_policy="point, probabilistic, interval, selective, and PPI summaries",
    )
    print_comparison_summary(
        "SPT-Reg synthetic summary",
        rows,
        metric_order=[
            "MSE",
            "MAE",
            "TailRMSE90",
            "NLL",
            "CRPS",
            "Cov90",
            "Width90",
            "AURC",
            "PPIMeanCIWidth",
            "PPIQuantileCIWidth",
            "train_s",
            "eval_s",
        ],
    )

    if summary_json_path is not None:
        write_comparison_summary_json(
            summary_json_path,
            example="examples/spt_reg_synthetic_comparison.py",
            task="SPT-Reg synthetic competing-method benchmark with BinnedPDF",
            config=cfg,
            rows=rows,
            notes=notes,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the synthetic competing-method benchmark for SPT-Reg."
    )
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()
    main(summary_json_path=args.summary_json_path)
