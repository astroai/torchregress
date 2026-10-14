"""
Comparison: post-hoc calibration and conformal wrappers on a Gaussian TabularMLP.

Canonical task: a heteroscedastic synthetic regression problem with a fixed seed
and one shared train / calibration / test split. Every method starts from the
same Gaussian ``TabularMLP`` recipe (``fit_tabular`` with ``GaussianNLLLoss``)
and reaches a 90% prediction interval in a different way:

* **Gaussian MLP (uncalibrated)** - ``mean +- z_0.95 * std`` straight from the head.
* **CalibratedRegressor(vts)** - ``torchregress.estimators.CalibratedRegressor``
  fits one variance temperature ``T`` on the calibration rows:
  ``N(mu, T sigma^2)``.
* **CalibratedRegressor(isotonic+vts)** - isotonic mean map on half of the
  calibration rows, variance temperature on the other half.
* **ConformalRegressor(split)** - ``torchregress.estimators.ConformalRegressor``
  with absolute-residual split conformal on the Gaussian head's mean (no
  predictive density, so no NLL).
* **calibrated_deep_ensemble** - ``torchregress.estimators.calibrated_deep_ensemble``:
  ``n_members`` Gaussian TabularMLPs, mixture moments, a variance temperature and
  normalized split-conformal widths (the recipe's default ``conformal=True``).

Run::

    python examples/calibrated_regressor_comparison.py \\
        --summary-json-path reports/calibrated_regressor_comparison_latest.json

Notes / caveats
---------------
* The single-model rows share one trained ``TabularFit``; their ``train_s`` is the
  calibration cost only and the base training time is printed separately.
* ``calibrated_deep_ensemble`` does its own seeded calibration carve: it gets the
  union of the train and calibration rows and ``val_fraction = n_cal / (n_train +
  n_cal)``, so it sees the same number of training and calibration rows but not
  the same rows. It trains ``n_members`` models, so its training cost is about
  ``n_members`` times the single model's.
* The temperature and the conformal threshold of the ensemble recipe are fitted
  on the same rows, so its coverage guarantee is approximate. Split conformal on
  a fixed calibration set guarantees marginal coverage only; Gaussian rows have
  no finite-sample guarantee at all.
* Coverage is measured on ``n_test`` rows (binomial standard error about 0.009 at
  1000 rows); differences of one or two points between rows are within noise.
* ``NLL`` is the Gaussian negative log-likelihood in target units (lower is
  better); ``IntervalScore`` is the proper scoring rule for central intervals.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
import torch

from torchregress.comparison import (
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
    timed_call,
    write_comparison_summary_json,
)
from torchregress.estimators import (
    CalibratedRegressor,
    ConformalRegressor,
    calibrated_deep_ensemble,
)
from torchregress.losses import GaussianNLLLoss
from torchregress.metrics import interval_score
from torchregress.models import TabularMLP, fit_tabular


@dataclass(frozen=True)
class CalibratedComparisonConfig:
    seed: int = 20261014
    n_train: int = 1500
    n_cal: int = 500
    n_test: int = 1000
    n_features: int = 6
    hidden: tuple[int, ...] = (64, 64)
    epochs: int = 60
    batch_size: int = 128
    lr: float = 2e-3
    patience: int = 15
    n_members: int = 3
    alpha: float = 0.1


def _simulate(cfg: CalibratedComparisonConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_train + cfg.n_cal + cfg.n_test
    X = rng.uniform(-2.0, 2.0, size=(n, cfg.n_features)).astype(np.float32)
    mean = np.sin(1.5 * X[:, 0]) + 0.5 * X[:, 1] * X[:, 2] + 0.3 * X[:, 3] ** 2
    noise_std = 0.1 + 0.4 * np.abs(X[:, 0]) + 0.2 * (X[:, 4] > 1.0)
    y = (mean + noise_std * rng.standard_normal(n)).astype(np.float32)
    s, c = cfg.n_train, cfg.n_train + cfg.n_cal
    return {
        "X_train": X[:s],
        "y_train": y[:s],
        "X_cal": X[s:c],
        "y_cal": y[s:c],
        "X_test": X[c:],
        "y_test": y[c:],
    }


def _z(alpha: float) -> float:
    return float(torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(1.0 - alpha / 2.0)))


def _metrics(
    y: np.ndarray,
    mean: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    std: np.ndarray | None,
    *,
    alpha: float,
) -> dict[str, float | None]:
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    nll = None
    if std is not None:
        var = np.asarray(std, dtype=np.float64) ** 2
        nll = float(np.mean(0.5 * (math.log(2 * math.pi) + np.log(var) + (y - mean) ** 2 / var)))
    return {
        "RMSE": float(np.sqrt(np.mean((np.asarray(mean) - y) ** 2))),
        "Coverage": float(np.mean((y >= lo) & (y <= hi))),
        "Width": float(np.mean(hi - lo)),
        "IntervalScore": float(interval_score(lo, hi, y, alpha=alpha).mean().item()),
        "NLL": nll,
    }


def _row(
    name: str,
    metrics: dict[str, float | None],
    train_s: float,
    eval_s: float,
    notes: str,
    *,
    alpha: float,
) -> dict[str, object]:
    return {
        "Method": name,
        "TargetCoverage": 1.0 - alpha,
        **metrics,
        "train_s": train_s,
        "eval_s": eval_s,
        "Notes": notes,
    }


def main(
    cfg: CalibratedComparisonConfig | None = None,
    summary_json_path: str | None = None,
) -> list[dict[str, object]]:
    cfg = cfg or CalibratedComparisonConfig()
    set_comparison_seed(cfg.seed)
    d = _simulate(cfg)
    y_test = d["y_test"].astype(np.float64)
    alpha = cfg.alpha
    fit_kwargs = {
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "patience": cfg.patience,
    }
    rows: list[dict[str, object]] = []

    # Shared base model: Gaussian TabularMLP on the training rows only.
    torch.manual_seed(cfg.seed)
    model = TabularMLP(cfg.n_features, 2, hidden=cfg.hidden)
    base, base_train_s = timed_call(
        fit_tabular,
        model,
        GaussianNLLLoss(),
        d["X_train"],
        d["y_train"],
        seed=cfg.seed,
        **fit_kwargs,
    )
    print(f"Base Gaussian TabularMLP: {base_train_s:.1f} s, best epoch {base.best_epoch}")

    # 1. Uncalibrated Gaussian head.
    def _raw_predict() -> tuple[np.ndarray, np.ndarray]:
        out = base.predict(d["X_test"]).numpy().astype(np.float64)
        return out[:, 0], np.exp(0.5 * out[:, 1])

    (mean, std), eval_s = timed_call(_raw_predict)
    z = _z(alpha)
    rows.append(
        _row(
            "Gaussian MLP (uncalibrated)",
            _metrics(y_test, mean, mean - z * std, mean + z * std, std, alpha=alpha),
            0.0,
            eval_s,
            "fit_tabular + GaussianNLLLoss; mean +- z * std",
            alpha=alpha,
        )
    )

    # 2-3. CalibratedRegressor on the shared calibration split.
    for method in ("vts", "isotonic+vts"):
        cal, cal_s = timed_call(
            CalibratedRegressor(base, method, alpha=alpha).fit,
            d["X_cal"],
            d["y_cal"],
            seed=cfg.seed,
        )

        def _cal_predict(cal: CalibratedRegressor = cal) -> tuple[np.ndarray, ...]:
            m, s = cal.predict_dist(d["X_test"])
            lo, hi = cal.predict_interval(d["X_test"])
            return m, s, lo, hi

        (m, s, lo, hi), eval_s = timed_call(_cal_predict)
        rows.append(
            _row(
                f"CalibratedRegressor({method})",
                _metrics(y_test, m, lo, hi, s, alpha=alpha),
                cal_s,
                eval_s,
                f"temperature T = {cal.temperature_:.3f} on {cal.n_calibration_} rows",
                alpha=alpha,
            )
        )

    # 4. Split conformal on the Gaussian head's mean (TabularFit implies prefit).
    conf, conf_s = timed_call(
        ConformalRegressor(base, method="split", alpha=alpha).fit, d["X_cal"], d["y_cal"]
    )

    def _conf_predict() -> tuple[np.ndarray, ...]:
        lo, hi = conf.predict_interval(d["X_test"])
        return conf.predict(d["X_test"]), lo, hi

    (m, lo, hi), eval_s = timed_call(_conf_predict)
    rows.append(
        _row(
            "ConformalRegressor(split)",
            _metrics(y_test, m, lo, hi, None, alpha=alpha),
            conf_s,
            eval_s,
            "absolute-residual split conformal; constant width, no density",
            alpha=alpha,
        )
    )

    # 5. Calibrated deep ensemble on the union of the train and calibration rows.
    X_pool = np.concatenate([d["X_train"], d["X_cal"]])
    y_pool = np.concatenate([d["y_train"], d["y_cal"]])
    torch.manual_seed(cfg.seed)
    ens, ens_s = timed_call(
        calibrated_deep_ensemble,
        X_pool,
        y_pool,
        n_members=cfg.n_members,
        val_fraction=cfg.n_cal / len(y_pool),
        alpha=alpha,
        seed=cfg.seed,
        model_kwargs={"hidden": cfg.hidden},
        **fit_kwargs,
    )

    def _ens_predict() -> tuple[np.ndarray, ...]:
        m, s = ens.predict_dist(d["X_test"])
        lo, hi = ens.predict_interval(d["X_test"])
        return m, s, lo, hi

    (m, s, lo, hi), eval_s = timed_call(_ens_predict)
    rows.append(
        _row(
            f"calibrated_deep_ensemble(M={cfg.n_members})",
            _metrics(y_test, m, lo, hi, s, alpha=alpha),
            ens_s,
            eval_s,
            f"mixture moments, T = {ens.temperature_:.3f}, normalized conformal widths",
            alpha=alpha,
        )
    )

    print_fairness_notes(
        title="Calibrated regressor comparison",
        seed_policy=f"fixed seed {cfg.seed}; shared train/calibration/test split",
        train_budget=(
            f"TabularMLP hidden={cfg.hidden}, up to {cfg.epochs} epochs, batch {cfg.batch_size}, "
            f"lr {cfg.lr}, patience {cfg.patience}; single-model rows share one fit "
            f"({base_train_s:.1f} s); the ensemble trains {cfg.n_members} members"
        ),
        metric_policy=(
            "RMSE, coverage vs target 0.9, mean width, interval score, Gaussian NLL "
            "(where a density exists), calibration/eval runtime"
        ),
    )
    print_comparison_summary(
        "Calibration / conformal wrappers on a Gaussian TabularMLP",
        rows,
        metric_order=[
            "TargetCoverage",
            "RMSE",
            "Coverage",
            "Width",
            "IntervalScore",
            "NLL",
            "train_s",
            "eval_s",
        ],
    )

    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/calibrated_regressor_comparison.py",
            task="Post-hoc calibration and conformal wrappers on a Gaussian TabularMLP",
            config=cfg,
            rows=rows,
            notes=[
                f"alpha = {alpha}",
                f"Base TabularMLP training time (shared by the single-model rows): "
                f"{base_train_s:.2f} s; their train_s is the calibration cost only.",
                "calibrated_deep_ensemble gets the union of the train and calibration rows "
                "and carves its own calibration set of the same size (different rows).",
                "ConformalRegressor(split) has no predictive density, so NLL is null.",
                "torchregress.estimators is experimental and may change in 0.4.",
            ],
        )
        print(f"\nWrote summary JSON: {out}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CalibratedRegressor / ConformalRegressor / calibrated_deep_ensemble comparison"
    )
    parser.add_argument("--epochs", type=int, default=CalibratedComparisonConfig.epochs)
    parser.add_argument("--seed", type=int, default=CalibratedComparisonConfig.seed)
    parser.add_argument("--n-members", type=int, default=CalibratedComparisonConfig.n_members)
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()
    main(
        CalibratedComparisonConfig(seed=args.seed, epochs=args.epochs, n_members=args.n_members),
        summary_json_path=args.summary_json_path,
    )
