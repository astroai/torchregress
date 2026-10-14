"""
External-comparison benchmark: conformal prediction intervals
(torchregress vs MAPIE / crepes / torchcp).

Canonical task: split-conformal and CQR intervals on a heteroscedastic regression
dataset with a fixed seed and shared train/calibration/test split.

Run::

    uv pip install mapie crepes torchcp
    uv run python examples/external_comparison_conformal_vs_mapie.py \\
        --summary-json-path reports/external_comparison_conformal_vs_mapie_latest.json

Notes
-----
* Four library wrappers are compared on the same split:
  - **torchregress**: small MLP backbones + ``ConformalLoss`` (split / CQR).
  - **MAPIE** (>= 1.0): sklearn estimators + ``SplitConformalRegressor`` /
    ``ConformalizedQuantileRegressor`` (``fit`` -> ``conformalize`` ->
    ``predict_interval``; ``confidence_level = 1 - alpha``).
  - **crepes**: sklearn estimators + ``crepes.ConformalRegressor`` calibration.
  - **torchcp** (>= 1.2): ``SplitPredictor`` with the ``ABS`` score on a torch
    ``nn.Linear`` carrying the fitted sklearn ``LinearRegression`` weights.
* Capacity is intentionally not matched between libraries. torchregress uses an
  MLP backbone; the others wrap sklearn estimators by design. To isolate the
  effect of the wrapper itself, ``torchregress/Split+Linear`` is included as a
  torchregress wrapper around a single linear layer — directly comparable to
  ``MAPIE/Split+Linear``, ``crepes/Split+Linear``, and ``torchcp/Split+Linear``.
* All three external libraries are optional dependencies. A comparator that is
  absent gives ``skipped: <lib> not installed``; one that is present but whose
  API does not match (for example MAPIE 0.x, which has no
  ``SplitConformalRegressor``) gives ``skipped: <lib> <version> incompatible``;
  an unexpected error while running it gives ``failed: ... incompatible``. In
  every case the rows are still emitted with ``Coverage``/``Width``/
  ``IntervalScore`` set to ``null`` so the JSON artifact stays schema-stable.
  The installed comparator versions are printed and recorded in the JSON
  ``notes`` and in each external row's ``Version`` field.
"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata as importlib_metadata

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


@dataclass(frozen=True)
class _Comparator:
    """Availability of an optional comparator package.

    ``status`` is ``"ok"``, ``"not_installed"`` (the package cannot be imported)
    or ``"incompatible"`` (installed, but the API this example targets is
    missing or fails to import).
    """

    label: str
    status: str
    version: str | None
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def skip_note(self) -> str:
        if self.status == "not_installed":
            return f"skipped: {self.label} not installed ({self.detail})"
        version = self.version or "(unknown version)"
        return f"skipped: {self.label} {version} incompatible ({self.detail})"


def _probe_comparator(
    label: str, module: str, dist: str, required: tuple[tuple[str, str], ...]
) -> _Comparator:
    """Import ``module`` and check each ``(submodule, attribute)`` in ``required``.

    Only a failure to import ``module`` itself counts as "not installed"; any
    other import failure or missing attribute means the installed release does
    not match the API this example was written for.
    """
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] == module.split(".")[0]:
            return _Comparator(label, "not_installed", None, str(exc))
        return _Comparator(label, "incompatible", _dist_version(dist), repr(exc))
    except Exception as exc:  # noqa: BLE001 - broken install / import-time failure
        return _Comparator(label, "incompatible", _dist_version(dist), repr(exc))
    version = _dist_version(dist)
    for submodule, attr in required:
        try:
            mod = importlib.import_module(submodule)
        except Exception as exc:  # noqa: BLE001
            return _Comparator(label, "incompatible", version, repr(exc))
        if not hasattr(mod, attr):
            return _Comparator(label, "incompatible", version, f"{submodule}.{attr} not found")
    return _Comparator(label, "ok", version)


def _dist_version(dist: str) -> str | None:
    try:
        return importlib_metadata.version(dist)
    except importlib_metadata.PackageNotFoundError:
        return None


_CREPES = _probe_comparator("crepes", "crepes", "crepes", (("crepes", "ConformalRegressor"),))
_MAPIE = _probe_comparator(
    "MAPIE",
    "mapie",
    "mapie",
    (
        ("mapie.regression", "SplitConformalRegressor"),
        ("mapie.regression", "ConformalizedQuantileRegressor"),
    ),
)
_TORCHCP = _probe_comparator(
    "torchcp",
    "torchcp",
    "torchcp",
    (
        ("torchcp.regression.predictor", "SplitPredictor"),
        ("torchcp.regression.score", "ABS"),
    ),
)

# Backward-compatible flags (the smoke tests and docs refer to these).
_CREPES_AVAILABLE = _CREPES.ok
_MAPIE_AVAILABLE = _MAPIE.ok
_TORCHCP_AVAILABLE = _TORCHCP.ok


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
    lo: np.ndarray, hi: np.ndarray, y: np.ndarray, *, alpha: float
) -> tuple[float | None, float | None, float | None]:
    """Return (coverage, width, interval_score) or Nones if inputs are None."""
    if lo is None or hi is None:
        return None, None, None
    coverage = float(np.mean((y >= lo) & (y <= hi)))
    width = float(np.mean(hi - lo))
    iscore = float(interval_score(lo, hi, y, alpha=alpha).mean().item())
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
    splits: dict[str, np.ndarray], *, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Split conformal with MAPIE >= 1.0 (``confidence_level = 1 - alpha``).

    The base regressor is fitted on the training split and conformalized on the
    shared calibration split, exactly like the other libraries.
    """
    from mapie.regression import SplitConformalRegressor
    from sklearn.linear_model import LinearRegression

    # MAPIE's conformity-score consistency check uses eps=1e-8, which float32
    # round-off (the shared splits are float32) can exceed; promote to float64.
    f64 = {k: np.asarray(v, dtype=np.float64) for k, v in splits.items()}
    mapie = SplitConformalRegressor(
        estimator=LinearRegression(),
        confidence_level=1.0 - alpha,
        prefit=False,
    )
    mapie.fit(f64["X_train"], f64["y_train"])
    mapie.conformalize(f64["X_cal"], f64["y_cal"])
    _, y_pis = mapie.predict_interval(f64["X_test"])
    return np.asarray(y_pis[:, 0, 0]), np.asarray(y_pis[:, 1, 0])


def _mapie_cqr_intervals(
    splits: dict[str, np.ndarray], *, alpha: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Conformalized quantile regression with MAPIE >= 1.0.

    MAPIE clones the single quantile-loss ``GradientBoostingRegressor`` into the
    lower / upper / median models itself (quantile levels ``alpha/2``,
    ``1 - alpha/2`` and ``0.5``). ``symmetric_correction=True`` applies one
    shared correction to both bounds, i.e. the same conformity score
    ``max(q_lo - y, y - q_hi)`` used by the torchregress and crepes CQR rows.
    """
    from mapie.regression import ConformalizedQuantileRegressor
    from sklearn.ensemble import GradientBoostingRegressor

    f64 = {k: np.asarray(v, dtype=np.float64) for k, v in splits.items()}
    mapie = ConformalizedQuantileRegressor(
        estimator=GradientBoostingRegressor(loss="quantile", random_state=seed),
        confidence_level=1.0 - alpha,
        prefit=False,
    )
    mapie.fit(f64["X_train"], f64["y_train"])
    mapie.conformalize(f64["X_cal"], f64["y_cal"])
    _, y_pis = mapie.predict_interval(f64["X_test"], symmetric_correction=True)
    return np.asarray(y_pis[:, 0, 0]), np.asarray(y_pis[:, 1, 0])


def _crepes_split_intervals(
    splits: dict[str, np.ndarray], *, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    from crepes import ConformalRegressor
    from sklearn.linear_model import LinearRegression

    point = LinearRegression().fit(splits["X_train"], splits["y_train"])
    pred_cal = point.predict(splits["X_cal"])
    pred_test = point.predict(splits["X_test"])
    cr = ConformalRegressor()
    cr.fit(np.asarray(splits["y_cal"]).reshape(-1) - pred_cal.reshape(-1))
    # crepes >= 0.9 renamed predict(significance=) to
    # predict_int(confidence=); both return (n, 2) intervals.
    intervals = cr.predict_int(pred_test.reshape(-1), confidence=1.0 - alpha)
    lo = np.asarray(intervals[:, 0])
    hi = np.asarray(intervals[:, 1])
    return lo, hi


def _crepes_cqr_intervals(
    splits: dict[str, np.ndarray], *, alpha: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """CQR via crepes' ``ConformalRegressor``.

    crepes does not ship a turnkey CQR wrapper, but ``ConformalRegressor`` is
    the natural primitive for "calibrate on a 1-D conformity score and read
    back a (1-alpha) conformal quantile". ``ConformalRegressor.fit`` takes the
    *absolute value* of whatever it is given, whereas the CQR score
    ``max(q_lo - y, y - q_hi)`` is signed (negative when ``y`` lies inside the
    quantile band). The score is therefore shifted by its minimum so it is
    non-negative, calibrated through crepes (``predict_int`` at the requested
    confidence returns the finite-sample conformal quantile of the shifted
    scores), and the shift is removed again. Feeding the raw signed score
    would silently turn it into ``|score|`` and over-cover (about 0.99).
    """
    from crepes import ConformalRegressor
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
    # CQR conformity score (signed): max(q_lo - y, y - q_hi).
    score = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
    shift = float(score.min())
    cr = ConformalRegressor()
    cr.fit(score - shift)  # non-negative, so crepes' abs() is the identity
    # Query a dummy point at 0; the interval is [-q', +q'] with q' the
    # (1-alpha) conformal quantile of the shifted scores.
    intervals = cr.predict_int(np.zeros(1), confidence=1.0 - alpha)
    q = float(intervals[0, 1]) + shift
    return q_lo_test - q, q_hi_test + q


def _torchcp_split_intervals(
    splits: dict[str, np.ndarray], *, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Split conformal with torchcp >= 1.2 (``SplitPredictor`` + ``ABS`` score).

    torchcp predictors need a ``torch.nn.Module`` and ``DataLoader`` inputs, so
    the fitted sklearn ``LinearRegression`` coefficients are copied into an
    ``nn.Linear`` (identical point predictions). ``predict`` returns a tensor of
    shape ``(n, n_alpha, 2)``; there is one ``alpha`` so ``[:, 0, :]`` is used.
    """
    from sklearn.linear_model import LinearRegression
    from torch.utils.data import DataLoader, TensorDataset
    from torchcp.regression.predictor import SplitPredictor
    from torchcp.regression.score import ABS

    point = LinearRegression().fit(splits["X_train"], splits["y_train"])
    model = nn.Linear(splits["X_train"].shape[1], 1)
    with torch.no_grad():
        model.weight.copy_(torch.from_numpy(np.asarray(point.coef_)).float().reshape(1, -1))
        model.bias.copy_(torch.tensor([float(point.intercept_)]))

    cp = SplitPredictor(ABS(), model=model, alpha=alpha, device="cpu")
    X_cal = torch.from_numpy(splits["X_cal"]).float()
    y_cal = torch.from_numpy(splits["y_cal"]).float()
    cp.calibrate(DataLoader(TensorDataset(X_cal, y_cal), batch_size=len(X_cal)))
    intervals = cp.predict(torch.from_numpy(splits["X_test"]).float())
    out = intervals.detach().cpu().numpy()[:, 0, :]
    return out[:, 0], out[:, 1]


def _row(
    name: str,
    library: str,
    coverage: float | None,
    width: float | None,
    interval_score_value: float | None,
    train_s: float | None,
    eval_s: float | None,
    notes: str,
    *,
    alpha: float = 0.1,
    version: str | None = None,
) -> dict[str, object]:
    return {
        "Method": name,
        "Library": library,
        "Version": version,
        "TargetCoverage": 1 - alpha,
        "Coverage": coverage,
        "Width": width,
        "IntervalScore": interval_score_value,
        "train_s": train_s,
        "eval_s": eval_s,
        "Notes": notes,
    }


def _external_row(
    comparator: _Comparator,
    name: str,
    runner: Callable[[], tuple[np.ndarray, np.ndarray]],
    y_test: np.ndarray,
    notes: str,
    *,
    alpha: float,
) -> dict[str, object]:
    """Run one comparator row, distinguishing "not installed" from "incompatible".

    ``runner`` returns ``(lower, upper)`` interval arrays. A missing package or
    an API mismatch detected at import time yields a skipped row; any exception
    raised while running an installed comparator yields a ``failed`` row that
    names the installed version, so API drift is never reported as "not
    installed".
    """
    lib = comparator.label
    kwargs = {"alpha": alpha, "version": comparator.version}
    if not comparator.ok:
        return _row(name, lib, None, None, None, None, None, comparator.skip_note(), **kwargs)
    try:
        (lo, hi), eval_s = timed_call(runner)
    except Exception as exc:  # noqa: BLE001 - any failure of an installed comparator
        note = f"failed: {lib} {comparator.version or '(unknown version)'} incompatible ({exc!r})"
        return _row(name, lib, None, None, None, None, None, note, **kwargs)
    cov, w, iscore = _intervals_metrics(lo, hi, y_test, alpha=alpha)
    return _row(name, lib, cov, w, iscore, None, eval_s, notes, **kwargs)


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
    cov, w, iscore = _intervals_metrics(lo, hi, splits["y_test"], alpha=cfg.alpha)
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
            alpha=cfg.alpha,
            version=_dist_version("torchregress"),
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
    cov, w, iscore = _intervals_metrics(lo, hi, splits["y_test"], alpha=cfg.alpha)
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
            alpha=cfg.alpha,
            version=_dist_version("torchregress"),
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
    cov, w, iscore = _intervals_metrics(lo, hi, splits["y_test"], alpha=cfg.alpha)
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
            alpha=cfg.alpha,
            version=_dist_version("torchregress"),
        )
    )

    # External comparators. Each row fits the point/quantile model on the train
    # split and calibrates on the shared calibration split.
    y_test = splits["y_test"]
    alpha = cfg.alpha
    rows.append(
        _external_row(
            _MAPIE,
            "MAPIE/Split+Linear",
            lambda: _mapie_split_intervals(splits, alpha=alpha),
            y_test,
            "sklearn LinearRegression + mapie.SplitConformalRegressor(confidence_level=1-alpha)",
            alpha=alpha,
        )
    )
    rows.append(
        _external_row(
            _MAPIE,
            "MAPIE/CQR+GBR",
            lambda: _mapie_cqr_intervals(splits, alpha=alpha, seed=cfg.seed),
            y_test,
            "sklearn GradientBoostingRegressor + mapie.ConformalizedQuantileRegressor"
            "(symmetric_correction=True)",
            alpha=alpha,
        )
    )
    rows.append(
        _external_row(
            _CREPES,
            "crepes/Split+Linear",
            lambda: _crepes_split_intervals(splits, alpha=alpha),
            y_test,
            "sklearn LinearRegression + crepes.ConformalRegressor (residual-based calibration)",
            alpha=alpha,
        )
    )
    rows.append(
        _external_row(
            _CREPES,
            "crepes/CQR+GBR",
            lambda: _crepes_cqr_intervals(splits, alpha=alpha, seed=cfg.seed),
            y_test,
            "sklearn GradientBoostingRegressor quantile + crepes ConformalRegressor on CQR scores",
            alpha=alpha,
        )
    )
    rows.append(
        _external_row(
            _TORCHCP,
            "torchcp/Split+Linear",
            lambda: _torchcp_split_intervals(splits, alpha=alpha),
            y_test,
            "sklearn LinearRegression weights in nn.Linear + torchcp SplitPredictor(ABS)",
            alpha=alpha,
        )
    )

    comparator_versions = [
        f"{c.label}={c.version or 'n/a'} ({c.status})" for c in (_MAPIE, _CREPES, _TORCHCP)
    ]
    print("\nComparator versions: " + ", ".join(comparator_versions))

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
            "Version",
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
                "Comparator versions: " + ", ".join(comparator_versions),
                f"MAPIE availability: {_MAPIE.status}",
                f"crepes availability: {_CREPES.status}",
                f"torchcp availability: {_TORCHCP.status}",
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
