r"""
TabularMLP tutorial: one backbone, three heads, missing values.

Shows the 0.3.0 ``torchregress.models`` API on scikit-learn's bundled Diabetes
data (442 rows, 10 features, no download):

1. ``TabularMLP`` + ``fit_tabular`` with a point loss (``WeightedMSELoss``,
   ``out_features = D = 1``).
2. The same backbone with ``GaussianNLLLoss`` (``out_features = 2 * D``:
   ``[mean, log_variance]``). ``fit_tabular`` standardises the target and
   ``TabularFit.predict`` maps the head back to target units
   (mean ``mu + sigma * m``, log-variance ``+ 2 log sigma``), so a 90% interval is

   .. math::  \hat\mu(x) \pm z_{0.95}\,\exp(\hat\ell(x) / 2).

3. ``TabularPreprocessor`` on features with injected NaNs: NaNs are imputed with
   the training median and every column that had a NaN at fit time gets a 0/1
   indicator column, so the model is built with
   ``in_features = preprocessor.n_features_out_``.

Run::

    python examples/tabular_mlp_tutorial.py --epochs 150 --seed 0

Caveats
-------
* Diabetes is small (309 training rows here); test RMSE moves by a few units
  between seeds and the 90% coverage is estimated from 133 test rows
  (binomial standard error about 0.026).
* The Gaussian head is not post-hoc calibrated. See
  ``examples/calibrated_regressor_comparison.py`` for ``CalibratedRegressor`` and
  conformal wrappers on top of a ``TabularFit``.
* The preprocessor is fitted on the training rows before ``fit_tabular`` and passed
  in, so the indicator columns (and therefore the input width) are fixed by the
  training data even if the internal validation carve happens to contain the only
  NaNs of a column.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
from sklearn.datasets import load_diabetes

from torchregress.losses import GaussianNLLLoss, WeightedMSELoss
from torchregress.models import TabularMLP, TabularPreprocessor, fit_tabular

Z90 = 1.6448536269514722  # standard normal 0.95 quantile


def load_split(seed: int, test_fraction: float = 0.3) -> tuple[np.ndarray, ...]:
    """Diabetes features/targets with a seeded train/test split."""
    X, y = load_diabetes(return_X_y=True)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    perm = np.random.default_rng(seed).permutation(len(y))
    n_test = int(round(test_fraction * len(y)))
    te, tr = perm[:n_test], perm[n_test:]
    return X[tr], y[tr], X[te], y[te]


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def build_model(in_features: int, out_features: int, seed: int) -> TabularMLP:
    # fit_tabular seeds the split, shuffling and dropout; the initial weights come
    # from the global RNG at construction time.
    torch.manual_seed(seed)
    return TabularMLP(in_features, out_features, hidden=(64, 64), dropout=0.1)


def fit_kwargs(epochs: int, seed: int) -> dict[str, object]:
    return {"epochs": epochs, "batch_size": 64, "lr": 2e-3, "patience": 30, "seed": seed}


def main(epochs: int = 150, seed: int = 0, nan_fraction: float = 0.1) -> dict[str, float]:
    X_tr, y_tr, X_te, y_te = load_split(seed)
    n_features = X_tr.shape[1]
    print(f"Diabetes: {len(y_tr)} train / {len(y_te)} test rows, {n_features} features")
    print(f"Budget: up to {epochs} epochs, batch 64, AdamW lr 2e-3, one-cycle, patience 30\n")
    results: dict[str, float] = {}

    # 1. Point head ---------------------------------------------------------
    t0 = time.perf_counter()
    mse_fit = fit_tabular(
        build_model(n_features, 1, seed), WeightedMSELoss(), X_tr, y_tr, **fit_kwargs(epochs, seed)
    )
    mse_pred = mse_fit.predict(X_te)[:, 0].numpy()
    results["mse_rmse"] = rmse(mse_pred, y_te)
    print("[1] TabularMLP + WeightedMSELoss (out_features=1)")
    print(f"    best epoch {mse_fit.best_epoch}, stopped early: {mse_fit.stopped_early}")
    print(f"    test RMSE {results['mse_rmse']:.2f}   ({time.perf_counter() - t0:.1f} s)")

    # 2. Gaussian head ------------------------------------------------------
    t0 = time.perf_counter()
    gauss_fit = fit_tabular(
        build_model(n_features, 2, seed), GaussianNLLLoss(), X_tr, y_tr, **fit_kwargs(epochs, seed)
    )
    out = gauss_fit.predict(X_te)  # [mean, log_variance] in target units
    mean, log_var = out[:, 0].numpy(), out[:, 1].numpy()
    std = np.exp(0.5 * log_var)
    lo, hi = mean - Z90 * std, mean + Z90 * std
    results["gauss_rmse"] = rmse(mean, y_te)
    results["gauss_coverage90"] = float(np.mean((y_te >= lo) & (y_te <= hi)))
    results["gauss_width90"] = float(np.mean(hi - lo))
    nll = 0.5 * (math.log(2 * math.pi) + log_var + (y_te - mean) ** 2 / std**2)
    results["gauss_nll"] = float(np.mean(nll))
    print("\n[2] TabularMLP + GaussianNLLLoss (out_features=2: mean, log_variance)")
    print(f"    output layout: {gauss_fit.output_layout!r}, best epoch {gauss_fit.best_epoch}")
    print(f"    test RMSE {results['gauss_rmse']:.2f}, NLL {results['gauss_nll']:.3f}")
    print(
        f"    90% interval coverage {results['gauss_coverage90']:.3f} "
        f"(target 0.900), mean width {results['gauss_width90']:.1f}"
        f"   ({time.perf_counter() - t0:.1f} s)"
    )

    # 3. Missing values -----------------------------------------------------
    rng = np.random.default_rng(seed + 1)
    nan_cols = [0, 2, 8]  # age, bmi, s5
    X_tr_nan, X_te_nan = X_tr.copy(), X_te.copy()
    for X in (X_tr_nan, X_te_nan):
        for j in nan_cols:
            X[rng.random(len(X)) < nan_fraction, j] = np.nan
    prep = TabularPreprocessor().fit(X_tr_nan)
    print(f"\n[3] TabularPreprocessor with {nan_fraction:.0%} NaNs in columns {nan_cols}")
    print(
        f"    n_features_in_ = {prep.n_features_in_}, n_features_out_ = {prep.n_features_out_}"
        f", indicator columns = {prep.indicator_columns_.tolist()}"
    )
    Z = prep.transform(X_te_nan[:3])
    print(f"    transformed test rows (last {len(nan_cols)} columns are missing indicators):")
    with np.printoptions(precision=2, suppress=True, linewidth=110):
        print("    " + str(Z).replace("\n", "\n    "))
    t0 = time.perf_counter()
    nan_fit = fit_tabular(
        build_model(prep.n_features_out_, 1, seed),
        WeightedMSELoss(),
        X_tr_nan,
        y_tr,
        preprocessor=prep,
        **fit_kwargs(epochs, seed),
    )
    results["nan_rmse"] = rmse(nan_fit.predict(X_te_nan)[:, 0].numpy(), y_te)
    print(
        f"    test RMSE with NaNs {results['nan_rmse']:.2f} "
        f"(clean features: {results['mse_rmse']:.2f})   ({time.perf_counter() - t0:.1f} s)"
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TabularMLP / fit_tabular tutorial on Diabetes")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nan-fraction", type=float, default=0.1)
    args = parser.parse_args()
    main(epochs=args.epochs, seed=args.seed, nan_fraction=args.nan_fraction)
