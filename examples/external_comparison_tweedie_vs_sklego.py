"""
External-comparison benchmark: Tweedie / compound-Poisson regression
(torchregress.TweedieLoss / CompoundPoissonLoss vs scikit-lego GLMRegressor).

Canonical task: synthetic zero-inflated continuous response drawn from a
compound Poisson-Gamma distribution with Tweedie power ``p=1.5`` on a shared
train/test split.

Run::

    uv pip install "torchregress[external]"
    uv run python examples/external_comparison_tweedie_vs_sklego.py \\
        --summary-json-path reports/external_comparison_tweedie_vs_sklego_latest.json

Notes
-----
* The torchregress methods train a small MLP on log-mean. scikit-lego's
  ``GLMRegressor(distribution="tweedie")`` fits a generalized linear model with
  a log link. Capacity is therefore not matched — the comparison highlights
  what each library offers out of the box.
* If scikit-lego is not installed, the script still runs and emits rows with
  metrics set to ``None`` and a note explaining the skip.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_tweedie_deviance

from torchregress.comparison import (
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
    timed_call,
    write_comparison_summary_json,
)
from torchregress.losses import CompoundPoissonLoss, TweedieLoss

try:
    from sklego.linear_model import GLMRegressor

    _SKLEGO_AVAILABLE = True
    _SKLEGO_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - soft dependency
    _SKLEGO_AVAILABLE = False
    _SKLEGO_ERROR = str(exc)


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
            mu = torch.exp(model(X[idx]))
            loss = loss_fn(mu, y[idx])
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
    train_s, _ = timed_call(
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
        "MAE": mae,
        "TweedieDeviance": deviance,
        "ZeroFracPred": float(np.mean(mu_pred <= 1e-3)),
        "train_s": train_s,
        "eval_s": eval_s,
        "Notes": f"MLP + torchregress.{name}Loss on log-mean",
    }


def _eval_sklego(
    splits: dict[str, np.ndarray], cfg: TweedieExternalConfig, *, seed: int
) -> dict[str, object]:
    glm = GLMRegressor(
        distribution="tweedie",
        power=cfg.p_power,
        alpha=1.0,
        fit_intercept=True,
        solver="lbfgs",
        max_iter=200,
    )
    train_s, _ = timed_call(glm.fit, splits["X_train"], splits["y_train"])
    mu_pred = np.asarray(glm.predict(splits["X_test"]), dtype=np.float64)
    mae = float(np.mean(np.abs(mu_pred - splits["y_test"])))
    deviance = _tweedie_deviance(splits["y_test"], mu_pred, cfg.p_power)
    return {
        "Method": "scikit-lego/GLMRegressor(tweedie)",
        "Library": "scikit-lego",
        "MAE": mae,
        "TweedieDeviance": deviance,
        "ZeroFracPred": float(np.mean(mu_pred <= 1e-3)),
        "train_s": train_s,
        "Notes": "log-link GLM with tweedie deviance; capacity not matched",
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

    if _SKLEGO_AVAILABLE:
        try:
            rows.append(_eval_sklego(splits, cfg, seed=cfg.seed))
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "Method": "scikit-lego/GLMRegressor(tweedie)",
                    "Library": "scikit-lego",
                    "MAE": None,
                    "TweedieDeviance": None,
                    "ZeroFracPred": None,
                    "train_s": None,
                    "Notes": f"failed: {exc!r}",
                }
            )
    else:
        rows.append(
            {
                "Method": "scikit-lego/GLMRegressor(tweedie)",
                "Library": "scikit-lego",
                "MAE": None,
                "TweedieDeviance": None,
                "ZeroFracPred": None,
                "train_s": None,
                "Notes": f"skipped: scikit-lego not installed ({_SKLEGO_ERROR})",
            }
        )

    print_fairness_notes(
        title="External Tweedie Comparison: torchregress vs scikit-lego",
        seed_policy="fixed seed; shared train/test split drawn from compound Poisson-Gamma",
        train_budget=(
            f"{cfg.epochs} epochs, batch={cfg.batch_size}, lr={cfg.lr} for torchregress MLP; "
            "LBFGS up to 200 iterations for scikit-lego GLM"
        ),
        metric_policy="MAE, Tweedie unit deviance, predicted zero-fraction, runtime",
    )
    print_comparison_summary(
        "Tweedie: torchregress vs scikit-lego",
        rows,
        metric_order=["MAE", "TweedieDeviance", "ZeroFracPred", "train_s", "eval_s"],
    )

    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/external_comparison_tweedie_vs_sklego.py",
            task="Tweedie / compound-Poisson regression (vs scikit-lego)",
            config=cfg,
            rows=rows,
            notes=[
                f"scikit-lego availability: {_SKLEGO_AVAILABLE}",
                f"p_power={cfg.p_power}, phi={cfg.phi}",
                "Capacity is not matched: torchregress uses an MLP; scikit-lego fits a log-link GLM.",
                f"Test zero-fraction (compound Poisson-Gamma draw): {zero_frac_test:.2%}",
            ],
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="External Tweedie comparison: torchregress vs scikit-lego"
    )
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()
    main(summary_json_path=args.summary_json_path)
