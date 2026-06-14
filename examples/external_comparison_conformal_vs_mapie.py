"""
External-comparison benchmark: conformal prediction intervals
(torchregress vs MAPIE / crepes / torchcp).

Canonical task: split-conformal and CQR intervals on a heteroscedastic regression
dataset with a fixed seed and shared train/calibration/test split.

Run::

    uv pip install "torchregress[external]"
    uv run python examples/external_comparison_conformal_vs_mapie.py \\
        --summary-json-path reports/external_comparison_conformal_vs_mapie_latest.json

Notes
-----
* Four library wrappers are compared on the same split:
  - **torchregress**: small MLP backbones + ``ConformalLoss`` (split / CQR).
  - **MAPIE**: sklearn estimators + ``MapieRegressor`` / ``MapieQuantileRegressor``.
  - **crepes**: sklearn estimators + ``crepes.ConformalRegressor`` calibration.
  - **torchcp**: sklearn estimators + ``torchcp.regression.SplitCP`` / CQR.
* Capacity is intentionally not matched between libraries. torchregress uses an
  MLP backbone; the others wrap sklearn estimators by design. To isolate the
  effect of the wrapper itself, ``torchregress/Split+Linear`` is included as a
  torchregress wrapper around a single linear layer — directly comparable to
  ``MAPIE/Split+Linear``, ``crepes/Split+Linear``, and ``torchcp/Split+Linear``.
* All three external libraries are optional dependencies. If ``crepes`` or
  ``torchcp`` is not installed, the corresponding rows are emitted with
  ``Coverage``/``Width``/``IntervalScore`` set to ``null`` and a note explaining
  the skip — so the JSON artifact stays schema-stable across environments.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchregress.comparison import (
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
    timed_call,
    write_comparison_summary_json,
)
from torchregress.losses.conformal import ConformalLoss
from torchregress.losses.quantile import MultiQuantileLoss
from torchregress.metrics import interval_score

try:
    from crepes import ConformalRegressor

    _CREPES_AVAILABLE = True
    _CREPES_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - soft dependency
    _CREPES_AVAILABLE = False
    _CREPES_ERROR = str(exc)

try:
    from mapie.regression import MapieQuantileRegressor, MapieRegressor

    _MAPIE_AVAILABLE = True
    _MAPIE_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - soft dependency
    _MAPIE_AVAILABLE = False
    _MAPIE_ERROR = str(exc)

try:
    from torchcp.regression import SplitCP

    _TORCHCP_AVAILABLE = True
    _TORCHCP_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - soft dependency
    _TORCHCP_AVAILABLE = False
    _TORCHCP_ERROR = str(exc)


@dataclass(frozen=True)
class ConformalExternalConfig:
    seed: int = 260612
    n_train: int = 800
    n_cal: int = 200
    n_test: int = 400
    n_features: int = 4
    hidden: int = 32
    epochs: int = 60
    batch_size: int = 64
    lr: float = 1e-3
    alpha: float = 0.1


def _simulate(cfg: ConformalExternalConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_train + cfg.n_cal + cfg.n_test
    X = rng.standard_normal((n, cfg.n_features)).astype(np.float32)
    y_mean = 0.7 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * np.sin(1.6 * X[:, 2]) + 0.2 * X[:, 3] ** 2
    noise_std = 0.15 + 0.25 * np.abs(X[:, 0])
    y = (y_mean + noise_std * rng.standard_normal(n)).astype(np.float32)
    s = cfg.n_train
    c = s + cfg.n_cal
    return {
        "X_train": X[:s],
        "y_train": y[:s],
        "X_cal": X[s:c],
        "y_cal": y[s:c],
        "X_test": X[c:],
        "y_test": y[c:],
    }


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Linear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _train_torch(
    model: nn.Module,
    loss_fn: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
) -> None:
    opt = optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad(set_to_none=True)
            pred = model(X[idx])
            loss = loss_fn(pred, y[idx])
            loss.backward()
            opt.step()


def _intervals_metrics(
    lo: np.ndarray, hi: np.ndarray, y: np.ndarray
) -> tuple[float | None, float | None, float | None]:
    """Return (coverage, width, interval_score) or Nones if inputs are None."""
    if lo is None or hi is None:
        return None, None, None
    coverage = float(np.mean((y >= lo) & (y <= hi)))
    width = float(np.mean(hi - lo))
    iscore = float(interval_score(lo, hi, y, alpha=0.1).mean().item())
    return coverage, width, iscore


def _torch_split_intervals(
    model: nn.Module, splits: dict[str, np.ndarray], *, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        pred_cal = model(torch.from_numpy(splits["X_cal"]).float()).squeeze(-1)
        pred_test = model(torch.from_numpy(splits["X_test"]).float()).squeeze(-1)
    residuals = (torch.from_numpy(splits["y_cal"]).float() - pred_cal).abs()
    q = torch.quantile(residuals, 1.0 - alpha).item()
    return (pred_test - q).numpy(), (pred_test + q).numpy()


def _torch_cqr_intervals(
    model: nn.Module, splits: dict[str, np.ndarray], *, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        pred_cal = model(torch.from_numpy(splits["X_cal"]).float())
        pred_test = model(torch.from_numpy(splits["X_test"]).float())
    y_cal = torch.from_numpy(splits["y_cal"]).float()
    q_lo_cal, q_hi_cal = pred_cal[:, 0], pred_cal[:, 1]
    q_lo_test, q_hi_test = pred_test[:, 0], pred_test[:, 1]
    score = torch.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
    q = torch.quantile(score, 1.0 - alpha).item()
    return (q_lo_test - q).numpy(), (q_hi_test + q).numpy()


def _mapie_split_intervals(
    splits: dict[str, np.ndarray], *, alpha: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.linear_model import LinearRegression

    mapie = MapieRegressor(
        estimator=LinearRegression(),
        method="base",
        cv="split",
        random_state=seed,
    )
    mapie.fit(splits["X_train"], splits["y_train"])
    _, y_pis = mapie.predict(splits["X_test"], alpha=alpha)
    return np.asarray(y_pis[:, 0, 0]), np.asarray(y_pis[:, 1, 0])


def _mapie_cqr_intervals(
    splits: dict[str, np.ndarray], *, alpha: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.ensemble import GradientBoostingRegressor

    lo = GradientBoostingRegressor(loss="quantile", alpha=alpha / 2, random_state=seed)
    hi = GradientBoostingRegressor(loss="quantile", alpha=1 - alpha / 2, random_state=seed)
    mapie = MapieQuantileRegressor(
        [lo, hi],
        method="quantile",
        cv="split",
        alpha=alpha,
        random_state=seed,
    )
    mapie.fit(splits["X_train"], splits["y_train"])
    mapie.calibrate(splits["X_cal"], splits["y_cal"])
    _, y_pis = mapie.predict(splits["X_test"])
    return np.asarray(y_pis[:, 0, 0]), np.asarray(y_pis[:, 1, 0])


def _crepes_split_intervals(
    splits: dict[str, np.ndarray], *, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.linear_model import LinearRegression

    point = LinearRegression().fit(splits["X_train"], splits["y_train"])
    pred_cal = point.predict(splits["X_cal"])
    pred_test = point.predict(splits["X_test"])
    cr = ConformalRegressor()
    cr.fit(np.asarray(splits["y_cal"]).reshape(-1) - pred_cal.reshape(-1))
    intervals = cr.predict(pred_test.reshape(-1), significance=alpha)
    lo = np.asarray(intervals[:, 0])
    hi = np.asarray(intervals[:, 1])
    return lo, hi


def _crepes_cqr_intervals(
    splits: dict[str, np.ndarray], *, alpha: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """CQR via crepes' ``ConformalRegressor``.

    crepes does not ship a turnkey CQR wrapper, but ``ConformalRegressor`` is
    the natural primitive for "calibrate on a 1-D conformity score and read
    back a (1-alpha) symmetric interval": ``cr.fit(score)`` then
    ``cr.predict(zeros, significance=alpha)`` returns a symmetric interval
    around 0 whose half-width is the (1-alpha) conformal correction. Apply
    that correction to the test quantile predictions to form the CQR
    prediction interval. This exercises crepes' intended API for
    residual-based calibration rather than re-implementing it on top of
    ``np.quantile``.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    lo = GradientBoostingRegressor(loss="quantile", alpha=alpha / 2, random_state=seed)
    hi = GradientBoostingRegressor(loss="quantile", alpha=1 - alpha / 2, random_state=seed)
    lo.fit(splits["X_train"], splits["y_train"])
    hi.fit(splits["X_train"], splits["y_train"])
    q_lo_cal = lo.predict(splits["X_cal"])
    q_hi_cal = hi.predict(splits["X_cal"])
    q_lo_test = lo.predict(splits["X_test"])
    q_hi_test = hi.predict(splits["X_test"])
    y_cal = np.asarray(splits["y_cal"]).reshape(-1)
    # CQR conformity score: max(q_lo - y, y - q_hi) — non-negative by
    # construction. crepes will treat the absolute value of these as
    # residuals.
    score = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
    cr = ConformalRegressor()
    cr.fit(score)
    # Query a dummy test point with conformity 0; crepes returns a symmetric
    # interval [-q, +q] whose upper bound is the (1-alpha) bound on |score|.
    _, intervals = cr.predict(np.zeros(1), significance=alpha)
    q = float(intervals[0, 1])
    return q_lo_test - q, q_hi_test + q


def _torchcp_split_intervals(
    splits: dict[str, np.ndarray], *, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Wrap a sklearn LinearRegression via torchcp's SplitCP.

    The torchcp regression API has changed across releases; this helper tries
    the most common entry points and falls back to the plain ``predict`` API.
    """
    from sklearn.linear_model import LinearRegression

    point = LinearRegression().fit(splits["X_train"], splits["y_train"])
    pred_cal = point.predict(splits["X_cal"])

    cp = SplitCP(point)
    # Try the modern ``calibrate`` API first; fall back to ``fit`` if absent.
    if hasattr(cp, "calibrate"):
        cp.calibrate(splits["X_cal"], splits["y_cal"])
    elif hasattr(cp, "fit"):
        cp.fit(residuals=np.asarray(splits["y_cal"]).reshape(-1) - pred_cal.reshape(-1))
    else:  # pragma: no cover - defensive
        raise RuntimeError("torchcp SplitCP has neither `calibrate` nor `fit`")

    if hasattr(cp, "predict"):
        out = cp.predict(splits["X_test"], alpha=alpha)
    elif hasattr(cp, "predict_intervals"):
        out = cp.predict_intervals(splits["X_test"], alpha=alpha)
    else:  # pragma: no cover - defensive
        raise RuntimeError("torchcp SplitCP has neither `predict` nor `predict_intervals`")

    # torchcp's return contract varies across releases. Four known shapes:
    # 1. ``tuple`` / ``list`` of length 2 — (lower, upper) each of shape (n,).
    # 2. ``np.ndarray`` of shape (n, 2) — stack of [lower, upper].
    # 3. ``np.ndarray`` of shape (2, n) — older releases stack as [lower; upper].
    # 4. Anything else — defensive error.
    if isinstance(out, (tuple, list)) and len(out) == 2:
        lo, hi = np.asarray(out[0]).reshape(-1), np.asarray(out[1]).reshape(-1)
    elif hasattr(out, "ndim") and out.ndim == 2:
        arr = np.asarray(out)
        if arr.shape[-1] == 2 and arr.shape[0] != 2:
            lo, hi = arr[:, 0].reshape(-1), arr[:, 1].reshape(-1)
        elif arr.shape[0] == 2 and arr.shape[-1] != 2:
            lo, hi = arr[0].reshape(-1), arr[1].reshape(-1)
        else:  # pragma: no cover - defensive
            raise RuntimeError(f"torchcp SplitCP returned an unrecognized 2-D shape: {arr.shape}")
    else:  # pragma: no cover - defensive
        raise RuntimeError(
            f"torchcp SplitCP returned an unrecognized shape/type: "
            f"type={type(out).__name__}, ndim={getattr(out, 'ndim', None)}, "
            f"shape={getattr(out, 'shape', None)}"
        )
    return lo, hi


def _row(
    name: str,
    library: str,
    coverage: float | None,
    width: float | None,
    interval_score_value: float | None,
    train_s: float | None,
    eval_s: float | None,
    notes: str,
) -> dict[str, object]:
    return {
        "Method": name,
        "Library": library,
        "TargetCoverage": 1 - 0.1,
        "Coverage": coverage,
        "Width": width,
        "IntervalScore": interval_score_value,
        "train_s": train_s,
        "eval_s": eval_s,
        "Notes": notes,
    }


def main(
    cfg: ConformalExternalConfig | None = None,
    summary_json_path: str | None = None,
) -> None:
    cfg = cfg or ConformalExternalConfig()
    set_comparison_seed(cfg.seed)
    splits = _simulate(cfg)

    X_train = torch.from_numpy(splits["X_train"]).float()
    y_train = torch.from_numpy(splits["y_train"]).float().unsqueeze(1)
    rows: list[dict[str, object]] = []

    # torchregress: Split + MLP point head
    set_comparison_seed(cfg.seed + 1)
    point_model = _MLP(cfg.n_features, 1, cfg.hidden)
    point_loss = ConformalLoss(method="split", alpha=cfg.alpha)
    _, pt_train_s = timed_call(
        _train_torch,
        point_model,
        point_loss,
        X_train,
        y_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
    )
    (lo, hi), pt_eval_s = timed_call(_torch_split_intervals, point_model, splits, alpha=cfg.alpha)
    cov, w, iscore = _intervals_metrics(lo, hi, splits["y_test"])
    rows.append(
        _row(
            "torchregress/Split+MLP",
            "torchregress",
            cov,
            w,
            iscore,
            pt_train_s,
            pt_eval_s,
            "point MLP + torchregress.ConformalLoss(split)",
        )
    )

    # torchregress: CQR + MLP quantile head
    set_comparison_seed(cfg.seed + 2)
    qr_model = _MLP(cfg.n_features, 2, cfg.hidden)
    qr_loss = MultiQuantileLoss(quantiles=[cfg.alpha / 2, 1 - cfg.alpha / 2])
    _, qr_train_s = timed_call(
        _train_torch,
        qr_model,
        qr_loss,
        X_train,
        y_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
    )
    (lo, hi), qr_eval_s = timed_call(_torch_cqr_intervals, qr_model, splits, alpha=cfg.alpha)
    cov, w, iscore = _intervals_metrics(lo, hi, splits["y_test"])
    rows.append(
        _row(
            "torchregress/CQR+MLP",
            "torchregress",
            cov,
            w,
            iscore,
            qr_train_s,
            qr_eval_s,
            "quantile MLP + torchregress.ConformalLoss(cqr)",
        )
    )

    # torchregress: Split + single Linear layer (fair-capacity baseline for
    # apples-to-apples comparison with MAPIE/crepes/torchcp on a linear backbone)
    set_comparison_seed(cfg.seed + 3)
    lin_model = _Linear(cfg.n_features, 1)
    _, lin_train_s = timed_call(
        _train_torch,
        lin_model,
        nn.MSELoss(),
        X_train,
        y_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
    )
    (lo, hi), lin_eval_s = timed_call(_torch_split_intervals, lin_model, splits, alpha=cfg.alpha)
    cov, w, iscore = _intervals_metrics(lo, hi, splits["y_test"])
    rows.append(
        _row(
            "torchregress/Split+Linear",
            "torchregress",
            cov,
            w,
            iscore,
            lin_train_s,
            lin_eval_s,
            "single linear layer + torchregress.ConformalLoss(split); matches sklearn backbone",
        )
    )

    # MAPIE: split + sklearn LinearRegression
    if _MAPIE_AVAILABLE:
        (lo, hi), mape_split_s = timed_call(
            _mapie_split_intervals, splits, alpha=cfg.alpha, seed=cfg.seed
        )
        cov, w, iscore = _intervals_metrics(lo, hi, splits["y_test"])
        rows.append(
            _row(
                "MAPIE/Split+Linear",
                "MAPIE",
                cov,
                w,
                iscore,
                None,
                mape_split_s,
                "sklearn LinearRegression + mapie.MapieRegressor(method='base')",
            )
        )
        (lo, hi), mape_cqr_s = timed_call(
            _mapie_cqr_intervals, splits, alpha=cfg.alpha, seed=cfg.seed
        )
        cov, w, iscore = _intervals_metrics(lo, hi, splits["y_test"])
        rows.append(
            _row(
                "MAPIE/CQR+GBR",
                "MAPIE",
                cov,
                w,
                iscore,
                None,
                mape_cqr_s,
                "sklearn GradientBoostingRegressor + mapie.MapieQuantileRegressor",
            )
        )
    else:
        rows.append(
            _row(
                "MAPIE/Split+Linear",
                "MAPIE",
                None,
                None,
                None,
                None,
                None,
                f"skipped: MAPIE not installed ({_MAPIE_ERROR})",
            )
        )
        rows.append(
            _row(
                "MAPIE/CQR+GBR",
                "MAPIE",
                None,
                None,
                None,
                None,
                None,
                "skipped: MAPIE not installed",
            )
        )

    # crepes: split + sklearn LinearRegression
    if _CREPES_AVAILABLE:
        (lo, hi), cr_split_s = timed_call(_crepes_split_intervals, splits, alpha=cfg.alpha)
        cov, w, iscore = _intervals_metrics(lo, hi, splits["y_test"])
        rows.append(
            _row(
                "crepes/Split+Linear",
                "crepes",
                cov,
                w,
                iscore,
                None,
                cr_split_s,
                "sklearn LinearRegression + crepes.ConformalRegressor (residual-based calibration)",
            )
        )
        (lo, hi), cr_cqr_s = timed_call(
            _crepes_cqr_intervals, splits, alpha=cfg.alpha, seed=cfg.seed
        )
        cov, w, iscore = _intervals_metrics(lo, hi, splits["y_test"])
        rows.append(
            _row(
                "crepes/CQR+GBR",
                "crepes",
                cov,
                w,
                iscore,
                None,
                cr_cqr_s,
                "sklearn GradientBoostingRegressor quantile + crepes ConformalRegressor on CQR scores",
            )
        )
    else:
        rows.append(
            _row(
                "crepes/Split+Linear",
                "crepes",
                None,
                None,
                None,
                None,
                None,
                f"skipped: crepes not installed ({_CREPES_ERROR})",
            )
        )
        rows.append(
            _row(
                "crepes/CQR+GBR",
                "crepes",
                None,
                None,
                None,
                None,
                None,
                "skipped: crepes not installed",
            )
        )

    # torchcp: split + sklearn LinearRegression (API-version tolerant)
    if _TORCHCP_AVAILABLE:
        try:
            (lo, hi), tccp_s = timed_call(_torchcp_split_intervals, splits, alpha=cfg.alpha)
            cov, w, iscore = _intervals_metrics(lo, hi, splits["y_test"])
            rows.append(
                _row(
                    "torchcp/Split+Linear",
                    "torchcp",
                    cov,
                    w,
                    iscore,
                    None,
                    tccp_s,
                    "sklearn LinearRegression + torchcp.regression.SplitCP (API-version tolerant)",
                )
            )
        except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
            rows.append(
                _row(
                    "torchcp/Split+Linear",
                    "torchcp",
                    None,
                    None,
                    None,
                    None,
                    None,
                    f"failed: {exc!r} (torchcp API drift across releases)",
                )
            )
    else:
        rows.append(
            _row(
                "torchcp/Split+Linear",
                "torchcp",
                None,
                None,
                None,
                None,
                None,
                f"skipped: torchcp not installed ({_TORCHCP_ERROR})",
            )
        )

    print_fairness_notes(
        title="External Conformal Comparison: torchregress vs MAPIE / crepes / torchcp",
        seed_policy="fixed seed; shared train/calibration/test split",
        train_budget=(
            f"{cfg.epochs} epochs, batch={cfg.batch_size}, lr={cfg.lr}; "
            "torchregress MLP + single-linear baselines; sklearn LinearRegression/GBR for the other libraries"
        ),
        metric_policy=(
            "coverage vs (1-alpha) target 0.9, mean interval width, "
            "interval score (proper scoring rule), and runtime"
        ),
    )
    print_comparison_summary(
        "Conformal: torchregress vs MAPIE vs crepes vs torchcp",
        rows,
        metric_order=[
            "TargetCoverage",
            "Coverage",
            "Width",
            "IntervalScore",
            "train_s",
            "eval_s",
        ],
    )

    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/external_comparison_conformal_vs_mapie.py",
            task="Conformal prediction intervals (vs MAPIE / crepes / torchcp)",
            config=cfg,
            rows=rows,
            notes=[
                f"MAPIE availability: {_MAPIE_AVAILABLE}",
                f"crepes availability: {_CREPES_AVAILABLE}",
                f"torchcp availability: {_TORCHCP_AVAILABLE}",
                f"alpha = {cfg.alpha}",
                "Capacity is not matched between libraries: torchregress uses MLPs; "
                "MAPIE/crepes/torchcp wrap sklearn estimators by design. "
                "torchregress/Split+Linear is the apples-to-apples wrapper comparison "
                "with MAPIE/crepes/torchcp on a single linear layer.",
                "IntervalScore is the proper scoring rule for predictive intervals (lower is better).",
            ],
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="External conformal comparison: torchregress vs MAPIE / crepes / torchcp"
    )
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()
    main(summary_json_path=args.summary_json_path)
