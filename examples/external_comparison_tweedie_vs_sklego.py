"""
External-comparison benchmark: Tweedie / compound-Poisson regression
(torchregress.TweedieLoss / CompoundPoissonLoss vs a log-link Tweedie GLM).

Canonical task: synthetic zero-inflated continuous response drawn from a
compound Poisson-Gamma distribution with Tweedie power ``p=1.5`` on a shared
train/test split.

Run::

    uv pip install scikit-learn scikit-lego  # scikit-lego is optional (only probed)
    uv run python examples/external_comparison_tweedie_vs_sklego.py \\
        --summary-json-path reports/external_comparison_tweedie_vs_sklego_latest.json

Notes
-----
* The torchregress methods train a small MLP on log-mean (the losses use the
  default log link, so the network output is ``log(mu)``).
* **The GLM baseline is scikit-learn, not scikit-lego.** scikit-lego no longer
  ships a Tweedie/GLM estimator (``sklego.linear_model.GLMRegressor`` is gone in
  0.9.x and nothing equivalent replaced it; its ``linear_model`` module only
  offers e.g. ``LADRegression``, ``QuantileRegression``,
  ``ImbalancedLinearRegression``). The comparison therefore uses
  ``sklearn.linear_model.TweedieRegressor(power=1.5, link="log")``, a
  log-link GLM fitted with the Tweedie deviance. scikit-lego is still probed so
  the summary records whether it is installed, which version, and that it has no
  Tweedie GLM; it is *not* needed to run the script.
* Capacity is not matched: the MLP is nonlinear, the GLM is linear in the
  features. The comparison highlights what each library offers out of the box.
* The installed versions of the comparator packages are printed and recorded in
  the JSON ``notes`` and in each row's ``Version`` field.
"""

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from importlib import metadata as importlib_metadata

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_tweedie_deviance

from torchregress.comparison import (
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
    timed_call,
    write_comparison_summary_json,
)
from torchregress.losses import CompoundPoissonLoss, TweedieLoss


def _dist_version(dist: str) -> str | None:
    try:
        return importlib_metadata.version(dist)
    except importlib_metadata.PackageNotFoundError:
        return None


def _probe_sklego() -> tuple[str, str | None, bool]:
    """Return ``(status, version, has_glm)`` for the optional scikit-lego package.

    ``status`` is ``"not_installed"`` when ``sklego`` cannot be imported and
    ``"installed"`` otherwise; ``has_glm`` says whether
    ``sklego.linear_model.GLMRegressor`` (the class the original comparison used)
    still exists.
    """
    try:
        linear_model = importlib.import_module("sklego.linear_model")
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] == "sklego":
            return "not_installed", None, False
        return "installed", _dist_version("scikit-lego"), False
    except Exception:  # noqa: BLE001 - broken install
        return "installed", _dist_version("scikit-lego"), False
    return "installed", _dist_version("scikit-lego"), hasattr(linear_model, "GLMRegressor")


_SKLEGO_STATUS, _SKLEGO_VERSION, _SKLEGO_HAS_GLM = _probe_sklego()
_SKLEGO_AVAILABLE = _SKLEGO_STATUS == "installed"
_SKLEARN_VERSION = _dist_version("scikit-learn")


@dataclass(frozen=True)
class TweedieExternalConfig:
    seed: int = 260614
    n_train: int = 1500
    n_test: int = 500
    n_features: int = 3
    p_power: float = 1.5
    phi: float = 0.6
    hidden: int = 32
    epochs: int = 60
    batch_size: int = 64
    lr: float = 1e-2
    glm_alpha: float = 0.0


def _simulate(cfg: TweedieExternalConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_train + cfg.n_test
    X = rng.uniform(-1.5, 1.5, size=(n, cfg.n_features)).astype(np.float32)
    log_mu = 0.7 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2]
    mu = np.exp(log_mu).astype(np.float32)
    p = cfg.p_power
    phi = cfg.phi
    lam = mu ** (2 - p) / (phi * (2 - p))
    shape = (2 - p) / (p - 1)
    scale = phi * (p - 1) * mu ** (p - 1)
    y = np.zeros(n, dtype=np.float32)
    for i in range(n):
        n_events = int(rng.poisson(lam[i]))
        if n_events > 0:
            y[i] = float(rng.gamma(shape, scale[i], size=n_events).sum())
    s = cfg.n_train
    return {
        "X_train": X[:s],
        "y_train": y[:s],
        "X_test": X[s:],
        "y_test": y[s:],
    }


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
            # The Tweedie losses default to a log link: they take log(mu) and
            # apply exp internally, so feed the raw network output.
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            opt.step()


def _tweedie_deviance(y: np.ndarray, mu: np.ndarray, p: float) -> float:
    """Mean Tweedie deviance using the sklearn / lightgbm convention.

    This is the same form implemented in ``sklearn.metrics.mean_tweedie_deviance``,
    so the torchregress-vs-scikit-lego comparison is on the same metric convention
    across library boundaries. The formula is

    .. math::

        D(y, \\mu) = 2 \\left[ \\frac{y^{2-p}}{(1-p)(2-p)} - \\frac{y \\cdot \\mu^{p-1}}{1-p} + \\frac{\\mu^{2-p}}{2-p} \\right]

    See https://scikit-learn.org/stable/modules/model_evaluation.html#mean-tweedie-deviance
    for the canonical reference.
    """
    return float(mean_tweedie_deviance(y, mu, sample_weight=None, power=p))


def _eval_torch(
    splits: dict[str, np.ndarray],
    cfg: TweedieExternalConfig,
    *,
    loss_factory,
    name: str,
) -> dict[str, object]:
    X_train = torch.from_numpy(splits["X_train"]).float()
    y_train = torch.from_numpy(splits["y_train"]).float().unsqueeze(1)
    X_test = torch.from_numpy(splits["X_test"]).float()
    model = _MLP(cfg.n_features, cfg.hidden)
    loss_fn = loss_factory(cfg.p_power)
    _, train_s = timed_call(
        _train_torch,
        model,
        loss_fn,
        X_train,
        y_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
    )
    model.eval()

    def _predict() -> np.ndarray:
        with torch.no_grad():
            return torch.exp(model(X_test)).squeeze(-1).numpy()

    mu_pred, eval_s = timed_call(_predict)
    mae = float(np.mean(np.abs(mu_pred - splits["y_test"])))
    deviance = _tweedie_deviance(splits["y_test"], mu_pred, cfg.p_power)
    return {
        "Method": f"torchregress/{name}",
        "Library": "torchregress",
        "Version": _dist_version("torchregress"),
        "MAE": mae,
        "TweedieDeviance": deviance,
        "ZeroFracPred": float(np.mean(mu_pred <= 1e-3)),
        "train_s": train_s,
        "eval_s": eval_s,
        "Notes": f"MLP + torchregress.{name}Loss on log-mean (log link)",
    }


def _eval_sklearn_glm(
    splits: dict[str, np.ndarray], cfg: TweedieExternalConfig
) -> dict[str, object]:
    """Log-link Tweedie GLM from scikit-learn (the replacement for scikit-lego's removed GLM)."""
    glm = TweedieRegressor(
        power=cfg.p_power,
        alpha=cfg.glm_alpha,
        link="log",
        fit_intercept=True,
        solver="lbfgs",
        max_iter=200,
    )
    _, train_s = timed_call(glm.fit, splits["X_train"], splits["y_train"])
    mu_pred, eval_s = timed_call(
        lambda: np.asarray(glm.predict(splits["X_test"]), dtype=np.float64)
    )
    mae = float(np.mean(np.abs(mu_pred - splits["y_test"])))
    deviance = _tweedie_deviance(splits["y_test"], mu_pred, cfg.p_power)
    return {
        "Method": "scikit-learn/TweedieRegressor(log)",
        "Library": "scikit-learn",
        "Version": _SKLEARN_VERSION,
        "MAE": mae,
        "TweedieDeviance": deviance,
        "ZeroFracPred": float(np.mean(mu_pred <= 1e-3)),
        "train_s": train_s,
        "eval_s": eval_s,
        "Notes": (
            "log-link Tweedie GLM (scikit-lego has no Tweedie GLM since 0.9.x); "
            f"alpha={cfg.glm_alpha}; capacity not matched"
        ),
    }


def main(
    cfg: TweedieExternalConfig | None = None,
    summary_json_path: str | None = None,
) -> None:
    cfg = cfg or TweedieExternalConfig()
    set_comparison_seed(cfg.seed)
    splits = _simulate(cfg)
    zero_frac_test = float(np.mean(splits["y_test"] == 0))

    rows: list[dict[str, object]] = []
    set_comparison_seed(cfg.seed + 10)
    rows.append(
        _eval_torch(
            splits,
            cfg,
            loss_factory=lambda p: TweedieLoss(p=p),
            name="Tweedie",
        )
    )
    set_comparison_seed(cfg.seed + 11)
    rows.append(
        _eval_torch(
            splits,
            cfg,
            loss_factory=lambda p: CompoundPoissonLoss(p=p),
            name="CompoundPoisson",
        )
    )

    try:
        rows.append(_eval_sklearn_glm(splits, cfg))
    except Exception as exc:  # noqa: BLE001 - report API drift instead of crashing
        rows.append(
            {
                "Method": "scikit-learn/TweedieRegressor(log)",
                "Library": "scikit-learn",
                "Version": _SKLEARN_VERSION,
                "MAE": None,
                "TweedieDeviance": None,
                "ZeroFracPred": None,
                "train_s": None,
                "eval_s": None,
                "Notes": f"failed: scikit-learn {_SKLEARN_VERSION} incompatible ({exc!r})",
            }
        )

    if _SKLEGO_STATUS == "not_installed":
        sklego_note = "scikit-lego not installed (not needed: it has no Tweedie GLM)"
    elif _SKLEGO_HAS_GLM:
        sklego_note = f"scikit-lego {_SKLEGO_VERSION} still has GLMRegressor, but it is not used"
    else:
        sklego_note = f"scikit-lego {_SKLEGO_VERSION} installed; no GLMRegressor / Tweedie GLM"
    versions = f"scikit-learn={_SKLEARN_VERSION}, scikit-lego={_SKLEGO_VERSION or 'n/a'}"
    print(f"\nComparator versions: {versions} ({sklego_note})")

    print_fairness_notes(
        title="External Tweedie Comparison: torchregress vs scikit-learn TweedieRegressor",
        seed_policy="fixed seed; shared train/test split drawn from compound Poisson-Gamma",
        train_budget=(
            f"{cfg.epochs} epochs, batch={cfg.batch_size}, lr={cfg.lr} for torchregress MLP; "
            f"LBFGS up to 200 iterations (alpha={cfg.glm_alpha}) for the scikit-learn GLM"
        ),
        metric_policy="MAE, Tweedie unit deviance, predicted zero-fraction, runtime",
    )
    print_comparison_summary(
        "Tweedie: torchregress vs scikit-learn TweedieRegressor (scikit-lego has no Tweedie GLM)",
        rows,
        metric_order=["Version", "MAE", "TweedieDeviance", "ZeroFracPred", "train_s", "eval_s"],
    )

    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/external_comparison_tweedie_vs_sklego.py",
            task="Tweedie / compound-Poisson regression (vs scikit-learn TweedieRegressor)",
            config=cfg,
            rows=rows,
            notes=[
                f"Comparator versions: {versions}",
                f"scikit-lego: {sklego_note}",
                f"p_power={cfg.p_power}, phi={cfg.phi}",
                "GLM baseline is sklearn.linear_model.TweedieRegressor (log link) because "
                "scikit-lego 0.9.x ships no Tweedie GLM.",
                "Capacity is not matched: torchregress uses an MLP; the GLM is linear in the features.",
                f"Test zero-fraction (compound Poisson-Gamma draw): {zero_frac_test:.2%}",
            ],
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="External Tweedie comparison: torchregress vs a log-link Tweedie GLM"
    )
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()
    main(summary_json_path=args.summary_json_path)
