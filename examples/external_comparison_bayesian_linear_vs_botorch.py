"""
External-comparison benchmark: low-shot Bayesian linear regression
(torchregress.BayesianLinearHead vs BoTorch SingleTaskGP).

Canonical task: low-shot linear regression with known ``w_true``. Reports RMSE,
Gaussian NLL, empirical 95% interval coverage, posterior-mean L2 error to
``w_true`` (torchregress only — GP weights are not directly comparable), and
runtime.

Run::

    uv pip install "torchregress[external]"
    uv run python examples/external_comparison_bayesian_linear_vs_botorch.py \\
        --summary-json-path reports/external_comparison_bayesian_linear_vs_botorch_latest.json

Notes
-----
* The two methods are intentionally different: torchregress offers an
  exact-conjugate BLR posterior in closed form, while BoTorch fits a
  Gaussian-process model with full hyperparameter marginal-likelihood
  optimization. Capacity is therefore not matched — this is an
  apples-to-oranges benchmark meant to highlight the operational gap.
* If BoTorch is not installed, the script still runs and emits rows with
  metrics set to ``None`` and a note explaining the skip.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from torchregress.comparison import (
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
    timed_call,
    write_comparison_summary_json,
)
from torchregress.test_time import BayesianLinearHead

try:
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    _BOTORCH_AVAILABLE = True
    _BOTORCH_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - soft dependency
    _BOTORCH_AVAILABLE = False
    _BOTORCH_ERROR = str(exc)


@dataclass(frozen=True)
class BayesianLinearExternalConfig:
    seed: int = 260613
    dim: int = 5
    n_train: int = 30
    n_test: int = 300
    noise: float = 0.3
    prior_precision: float = 1e-2


def _simulate(cfg: BayesianLinearExternalConfig) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(cfg.seed)
    w_true = torch.randn(cfg.dim, generator=g)
    X_train = torch.randn(cfg.n_train, cfg.dim, generator=g)
    y_train = (X_train @ w_true).unsqueeze(-1) + cfg.noise * torch.randn(
        cfg.n_train, 1, generator=g
    )
    X_test = torch.randn(cfg.n_test, cfg.dim, generator=g)
    y_test = (X_test @ w_true).unsqueeze(-1) + cfg.noise * torch.randn(cfg.n_test, 1, generator=g)
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "w_true": w_true,
    }


def _eval_blr(
    splits: dict[str, torch.Tensor], cfg: BayesianLinearExternalConfig
) -> dict[str, object]:
    head = BayesianLinearHead(
        in_features=cfg.dim,
        fit_intercept=False,
        prior_precision=cfg.prior_precision,
        noise_variance=cfg.noise**2,
    ).fit(splits["X_train"], splits["y_train"])
    pred = head.predict(splits["X_test"], return_std=True, include_noise=True)
    mean = pred["mean"].squeeze(-1)
    std = pred["std"].squeeze(-1).clamp_min(1e-6)
    rmse = torch.sqrt(torch.mean((mean - splits["y_test"].squeeze(-1)) ** 2)).item()
    nll = -(
        torch.distributions.Normal(mean, std).log_prob(splits["y_test"].squeeze(-1)).mean().item()
    )
    cov = (
        (
            (mean - 1.96 * std <= splits["y_test"].squeeze(-1))
            & (splits["y_test"].squeeze(-1) <= mean + 1.96 * std)
        )
        .float()
        .mean()
        .item()
    )
    w_err = float((head.posterior_mean[0] - splits["w_true"]).norm().item())
    return {
        "Method": "torchregress/BayesianLinearHead",
        "Library": "torchregress",
        "RMSE": rmse,
        "NLL": nll,
        "Cov95": cov,
        "PostErrL2": w_err,
    }


def _eval_botorch(
    splits: dict[str, torch.Tensor], cfg: BayesianLinearExternalConfig, *, seed: int
) -> dict[str, object]:
    train_X = splits["X_train"].double()
    train_Y = splits["y_train"].double()
    test_X = splits["X_test"].double()
    test_Y = splits["y_test"].squeeze(-1).double()
    gp = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll, max_attempts=2)
    gp.eval()
    gp.likelihood.eval()
    with torch.no_grad():
        posterior = gp.posterior(test_X)
        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.squeeze(-1).clamp_min(1e-8).sqrt()
    rmse = torch.sqrt(torch.mean((mean - test_Y) ** 2)).item()
    nll = -posterior.log_prob(test_Y.unsqueeze(-1)).div(test_Y.numel()).item()
    cov = ((mean - 1.96 * std <= test_Y) & (test_Y <= mean + 1.96 * std)).float().mean().item()
    return {
        "Method": "BoTorch/SingleTaskGP",
        "Library": "BoTorch",
        "RMSE": rmse,
        "NLL": nll,
        "Cov95": cov,
        "PostErrL2": None,  # GP weights are not directly comparable
    }


def main(
    cfg: BayesianLinearExternalConfig | None = None,
    summary_json_path: str | None = None,
) -> None:
    cfg = cfg or BayesianLinearExternalConfig()
    set_comparison_seed(cfg.seed)
    splits = _simulate(cfg)
    rows: list[dict[str, object]] = []

    blr_result, blr_train_s = timed_call(_eval_blr, splits, cfg)
    blr_result["train_s"] = blr_train_s
    blr_result["Notes"] = "exact-conjugate posterior in closed form"
    rows.append(blr_result)

    if _BOTORCH_AVAILABLE:
        try:
            bt_result, bt_total_s = timed_call(_eval_botorch, splits, cfg, seed=cfg.seed)
            bt_result["train_s"] = bt_total_s
            bt_result["Notes"] = "GP hyperparameter marginal-likelihood fit; capacity not matched"
            rows.append(bt_result)
        except (RuntimeError, ValueError, ImportError) as exc:
            rows.append(
                {
                    "Method": "BoTorch/SingleTaskGP",
                    "Library": "BoTorch",
                    "RMSE": None,
                    "NLL": None,
                    "Cov95": None,
                    "PostErrL2": None,
                    "train_s": None,
                    "Notes": f"failed: {exc!r}",
                }
            )
    else:
        rows.append(
            {
                "Method": "BoTorch/SingleTaskGP",
                "Library": "BoTorch",
                "RMSE": None,
                "NLL": None,
                "Cov95": None,
                "PostErrL2": None,
                "train_s": None,
                "Notes": f"skipped: BoTorch not installed ({_BOTORCH_ERROR})",
            }
        )

    print_fairness_notes(
        title="External Bayesian Linear Comparison: torchregress vs BoTorch",
        seed_policy="fixed seed; shared low-shot linear split",
        train_budget=(
            f"n_train={cfg.n_train}, d={cfg.dim}, noise={cfg.noise}; "
            "torchregress uses exact-conjugate BLR; BoTorch fits a GP via MLL"
        ),
        metric_policy="RMSE, Gaussian NLL, empirical 95% coverage, posterior L2 error to w_true (torchregress only)",
    )
    print_comparison_summary(
        "Low-shot Bayesian Linear: torchregress vs BoTorch",
        rows,
        metric_order=["RMSE", "NLL", "Cov95", "PostErrL2", "train_s"],
    )

    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/external_comparison_bayesian_linear_vs_botorch.py",
            task="Low-shot Bayesian linear regression (vs BoTorch)",
            config=cfg,
            rows=rows,
            notes=[
                f"BoTorch availability: {_BOTORCH_AVAILABLE}",
                f"n_train={cfg.n_train}, d={cfg.dim}, noise={cfg.noise}",
                "Capacity is not matched: torchregress is exact-conjugate linear; "
                "BoTorch fits a flexible GP with hyperparameter MLL.",
            ],
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="External Bayesian linear comparison: torchregress vs BoTorch"
    )
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()
    main(summary_json_path=args.summary_json_path)
