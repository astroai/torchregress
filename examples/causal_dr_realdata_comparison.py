"""Shared-budget comparison for doubly-robust causal ATE/CATE on real covariates."""

import argparse
from dataclasses import dataclass

import torch
from sklearn.datasets import load_diabetes

from torchregress.causal import causal_overlap_report, dr_ate, dr_cate
from torchregress.comparison import (
    print_comparison_summary,
    print_fairness_notes,
    timed_call,
    write_comparison_summary_json,
)


@dataclass(frozen=True)
class CausalDRRealDataConfig:
    seed: int = 260305
    n_samples: int = 420
    folds: int = 3
    alpha: float = 0.05


def _fit_models():
    from sklearn.linear_model import (  # type: ignore[import-untyped]
        LinearRegression,
        LogisticRegression,
    )

    return LinearRegression, LogisticRegression


def _real_covariates(cfg: CausalDRRealDataConfig) -> tuple[torch.Tensor, torch.Tensor]:
    x_np, y_np = load_diabetes(return_X_y=True)
    x_all = torch.tensor(x_np, dtype=torch.float32)
    y_all = torch.tensor(y_np, dtype=torch.float32)

    if cfg.n_samples > x_all.shape[0]:
        raise ValueError(
            f"Requested {cfg.n_samples} samples but diabetes dataset has {x_all.shape[0]}."
        )

    g = torch.Generator().manual_seed(cfg.seed)
    idx = torch.randperm(x_all.shape[0], generator=g)[: cfg.n_samples]
    x = x_all[idx]
    y_raw = y_all[idx]
    y = (y_raw - y_raw.mean()) / y_raw.std(unbiased=False).clamp_min(1e-6)
    return x, y


def _make_scenario_1(cfg: CausalDRRealDataConfig) -> dict[str, torch.Tensor]:
    x, baseline = _real_covariates(cfg)
    g = torch.Generator().manual_seed(cfg.seed + 19)

    tau = 0.18 + 0.16 * torch.tanh(2.4 * x[:, 0])
    p = torch.sigmoid(1.15 * x[:, 0] - 0.9 * x[:, 1] + 0.25 * x[:, 2]).clamp(0.03, 0.97)
    t = torch.bernoulli(p, generator=g)
    y0 = baseline + 0.18 * torch.randn(baseline.shape, generator=g)
    y = y0 + t * tau
    return {"x": x, "t": t, "y": y, "tau": tau, "p": p}


def _make_scenario_2(cfg: CausalDRRealDataConfig) -> dict[str, torch.Tensor]:
    x, baseline = _real_covariates(cfg)
    g = torch.Generator().manual_seed(cfg.seed + 31)

    tau = 0.10 + 0.20 * torch.sigmoid(2.0 * x[:, 3])
    p = torch.sigmoid(-1.0 * x[:, 0] + 1.2 * x[:, 2] - 0.6 * x[:, 4]).clamp(0.03, 0.97)
    t = torch.bernoulli(p, generator=g)
    y0 = baseline + 0.25 * x[:, 2] - 0.2 * x[:, 4] + 0.16 * torch.randn(baseline.shape, generator=g)
    y = y0 + t * tau
    return {"x": x, "t": t, "y": y, "tau": tau, "p": p}


def _naive_ate(t: torch.Tensor, y: torch.Tensor) -> float:
    treated = t > 0.5
    control = ~treated
    if int(treated.sum().item()) == 0 or int(control.sum().item()) == 0:
        return float("nan")
    return float(y[treated].mean().item() - y[control].mean().item())


def _scenario_rows(
    scenario: str,
    data: dict[str, torch.Tensor],
    cfg: CausalDRRealDataConfig,
) -> list[dict[str, object]]:
    LinearRegression, LogisticRegression = _fit_models()
    true_ate = float(data["tau"].mean().item())
    naive = _naive_ate(data["t"], data["y"])
    overlap = causal_overlap_report(data["p"], data["t"], trim_threshold=0.05)

    rows: list[dict[str, object]] = [
        {
            "Method": f"{scenario}-NaiveDiff",
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

    def _run_dr_ate() -> dict[str, object]:
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
    ate_est = float(ate_result["estimate"])
    ci_low = float(ate_result["ci_low"])
    ci_high = float(ate_result["ci_high"])
    rows.append(
        {
            "Method": f"{scenario}-DRATE",
            "ATE_true": true_ate,
            "ATE_hat": ate_est,
            "ATE_abs_error": abs(ate_est - true_ate),
            "CI_contains_true": float(ci_low <= true_ate <= ci_high),
            "CI_width": ci_high - ci_low,
            "OverlapRate": float(ate_result["diagnostics"]["overlap_rate"]),
            "MinESS": float(ate_result["diagnostics"]["min_group_ess"]),
            "train_s": float(ate_s),
            "eval_s": 0.0,
            "Notes": "cross-fitted doubly-robust ATE with overlap diagnostics",
        }
    )

    def _run_dr_cate() -> dict[str, object]:
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
    cate_ci_low = float(cate_result["ate_ci_low"])
    cate_ci_high = float(cate_result["ate_ci_high"])
    rows.append(
        {
            "Method": f"{scenario}-DRCATE",
            "ATE_true": true_ate,
            "ATE_hat": cate_ate,
            "ATE_abs_error": abs(cate_ate - true_ate),
            "CI_contains_true": float(cate_ci_low <= true_ate <= cate_ci_high),
            "CI_width": cate_ci_high - cate_ci_low,
            "OverlapRate": float(cate_result["diagnostics"]["overlap_rate"]),
            "MinESS": float(cate_result["diagnostics"]["min_group_ess"]),
            "train_s": float(cate_s),
            "eval_s": 0.0,
            "Notes": "DR pseudo-outcome CATE with aggregate ATE diagnostics",
        }
    )

    return rows


def run_comparison(cfg: CausalDRRealDataConfig) -> tuple[list[dict[str, object]], list[str]]:
    scenario_1 = _make_scenario_1(cfg)
    scenario_2 = _make_scenario_2(cfg)
    rows = _scenario_rows("DiabetesProxy", scenario_1, cfg) + _scenario_rows(
        "DiabetesSelectionBias", scenario_2, cfg
    )
    notes = [
        "Uses real Diabetes covariates and baseline outcomes with simulated treatment assignment.",
        "Ground-truth treatment effects are known by construction for ATE/CATE error checks.",
        "DR methods use cross-fitting and overlap diagnostics in both scenarios.",
    ]
    return rows, notes


def main(cfg: CausalDRRealDataConfig | None = None, summary_json_path: str | None = None) -> None:
    cfg = cfg or CausalDRRealDataConfig()
    rows, notes = run_comparison(cfg)

    print_fairness_notes(
        title="Causal DR Comparison (Real Covariates)",
        seed_policy="fixed seed and shared Diabetes covariate sampling",
        train_budget="shared fold count and nuisance-model classes",
        metric_policy="ATE absolute error, CI coverage/width, overlap diagnostics, runtime",
    )
    print_comparison_summary(
        "Causal DR summary (real covariates)",
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
            example="examples/causal_dr_realdata_comparison.py",
            task="Causal inference regression (DR ATE/CATE, real covariates)",
            config=cfg,
            rows=rows,
            notes=notes,
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run causal DR comparison on real covariates")
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()
    main(summary_json_path=args.summary_json_path)
