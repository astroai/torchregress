"""Shared-budget comparison for doubly-robust causal ATE/CATE estimation."""

import argparse
from dataclasses import dataclass

import torch
from comparison_utils import (
    print_comparison_summary,
    print_fairness_notes,
    timed_call,
    write_comparison_summary_json,
)

from torchregress.causal import causal_overlap_report, dr_ate, dr_cate


@dataclass(frozen=True)
class CausalDRConfig:
    seed: int = 260227
    n_samples: int = 1200
    n_features: int = 6
    folds: int = 3
    alpha: float = 0.05


def _fit_models():
    from sklearn.linear_model import (  # type: ignore[import-untyped]
        LinearRegression,
        LogisticRegression,
    )

    return LinearRegression, LogisticRegression


def _simulate_uplift(cfg: CausalDRConfig) -> dict[str, torch.Tensor]:
    torch.manual_seed(cfg.seed)
    x = torch.randn(cfg.n_samples, cfg.n_features)
    base = 0.5 * x[:, 0] - 0.4 * x[:, 1] + 0.3 * x[:, 2] ** 2
    tau = 0.4 + 0.3 * torch.tanh(x[:, 0])
    p = torch.sigmoid(0.9 * x[:, 0] - 0.7 * x[:, 1] + 0.25 * x[:, 2]).clamp(0.03, 0.97)
    t = torch.bernoulli(p)
    y0 = base + 0.25 * torch.randn(cfg.n_samples)
    y = y0 + t * tau
    return {"x": x, "t": t, "y": y, "tau": tau, "p": p}


def _simulate_selection_bias(cfg: CausalDRConfig) -> dict[str, torch.Tensor]:
    torch.manual_seed(cfg.seed + 11)
    x = torch.randn(cfg.n_samples, cfg.n_features)
    f1 = 1.1 * x[:, 0] - 0.8 * x[:, 1]
    f2 = 0.9 * x[:, 2] + 0.5 * x[:, 3]
    p = torch.sigmoid(1.2 * f1 - 0.6 * f2).clamp(0.03, 0.97)
    t = torch.bernoulli(p)
    tau = 0.25 + 0.2 * torch.sigmoid(f2)
    y0 = 0.7 * f1 + 0.4 * x[:, 4] - 0.2 * x[:, 5] + 0.3 * torch.randn(cfg.n_samples)
    y = y0 + t * tau
    return {"x": x, "t": t, "y": y, "tau": tau, "p": p}


def _naive_ate(t: torch.Tensor, y: torch.Tensor) -> float:
    treated = t > 0.5
    control = ~treated
    if int(treated.sum().item()) == 0 or int(control.sum().item()) == 0:
        return float("nan")
    return float(y[treated].mean().item() - y[control].mean().item())


def _scenario_rows(
    name: str,
    data: dict[str, torch.Tensor],
    cfg: CausalDRConfig,
) -> list[dict[str, object]]:
    LinearRegression, LogisticRegression = _fit_models()
    true_ate = float(data["tau"].mean().item())
    naive = _naive_ate(data["t"], data["y"])
    overlap = causal_overlap_report(data["p"], data["t"], trim_threshold=0.05)

    rows: list[dict[str, object]] = [
        {
            "Method": f"{name}-NaiveDiff",
            "ATE_true": true_ate,
            "ATE_hat": naive,
            "ATE_abs_error": abs(naive - true_ate),
            "CI_contains_true": 0.0,
            "CI_width": 0.0,
            "OverlapRate": overlap["overlap_rate"],
            "MinESS": overlap["min_group_ess"],
            "train_s": 0.0,
            "eval_s": 0.0,
            "Notes": "difference in means without confounding adjustment",
        }
    ]

    def _run_dr_ate():
        return dr_ate(
            data["x"],
            data["t"],
            data["y"],
            outcome_model=LinearRegression,
            propensity_model=LogisticRegression(max_iter=1000),
            folds=cfg.folds,
            alpha=cfg.alpha,
            seed=cfg.seed,
            trim_threshold=0.05,
        )

    ate_result, ate_s = timed_call(_run_dr_ate)
    ci_contains = float(ate_result["ci_low"] <= true_ate <= ate_result["ci_high"])
    rows.append(
        {
            "Method": f"{name}-DRATE",
            "ATE_true": true_ate,
            "ATE_hat": float(ate_result["estimate"]),
            "ATE_abs_error": abs(float(ate_result["estimate"]) - true_ate),
            "CI_contains_true": ci_contains,
            "CI_width": float(ate_result["ci_high"] - ate_result["ci_low"]),
            "OverlapRate": float(ate_result["diagnostics"]["overlap_rate"]),
            "MinESS": float(ate_result["diagnostics"]["min_group_ess"]),
            "train_s": float(ate_s),
            "eval_s": 0.0,
            "Notes": "cross-fitted doubly-robust ATE with overlap diagnostics",
        }
    )

    def _run_dr_cate():
        return dr_cate(
            data["x"],
            data["t"],
            data["y"],
            cate_model=LinearRegression,
            outcome_model=LinearRegression,
            propensity_model=LogisticRegression(max_iter=1000),
            folds=cfg.folds,
            alpha=cfg.alpha,
            seed=cfg.seed,
            trim_threshold=0.05,
        )

    cate_result, cate_s = timed_call(_run_dr_cate)
    cate_ate = float(cate_result["ate_estimate"])
    ci_contains_cate = float(cate_result["ate_ci_low"] <= true_ate <= cate_result["ate_ci_high"])
    rows.append(
        {
            "Method": f"{name}-DRCATE",
            "ATE_true": true_ate,
            "ATE_hat": cate_ate,
            "ATE_abs_error": abs(cate_ate - true_ate),
            "CI_contains_true": ci_contains_cate,
            "CI_width": float(cate_result["ate_ci_high"] - cate_result["ate_ci_low"]),
            "OverlapRate": float(cate_result["diagnostics"]["overlap_rate"]),
            "MinESS": float(cate_result["diagnostics"]["min_group_ess"]),
            "train_s": float(cate_s),
            "eval_s": 0.0,
            "Notes": "DR pseudo-outcome regression for CATE with aggregate ATE report",
        }
    )
    return rows


def run_comparison(cfg: CausalDRConfig) -> tuple[list[dict[str, object]], list[str]]:
    uplift = _simulate_uplift(cfg)
    selection = _simulate_selection_bias(cfg)
    rows = _scenario_rows("Uplift", uplift, cfg) + _scenario_rows("SelectionBias", selection, cfg)
    notes = [
        "Both scenarios use binary treatment with confounded assignment.",
        "DR methods use cross-fitting by default and report overlap diagnostics.",
        "SelectionBias simulates selection effects based on covariate features.",
    ]
    return rows, notes


def main(cfg: CausalDRConfig | None = None, summary_json_path: str | None = None) -> None:
    cfg = cfg or CausalDRConfig()
    rows, notes = run_comparison(cfg)

    print_fairness_notes(
        title="Causal DR Comparison",
        seed_policy="fixed seeds and shared generated datasets",
        train_budget="shared fold count and nuisance-model classes",
        metric_policy="ATE absolute error, CI coverage/width, overlap diagnostics, runtime",
    )
    print_comparison_summary(
        "Causal DR summary",
        rows,
        metric_order=[
            "ATE_true",
            "ATE_hat",
            "ATE_abs_error",
            "CI_contains_true",
            "CI_width",
            "OverlapRate",
            "MinESS",
            "train_s",
        ],
    )

    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/causal_dr_uplift_comparison.py",
            task="Causal inference regression (DR ATE/CATE)",
            config=cfg,
            rows=rows,
            notes=notes,
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run causal DR uplift comparison")
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()
    main(summary_json_path=args.summary_json_path)
