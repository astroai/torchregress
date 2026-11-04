"""Shared-budget comparison for doubly-robust causal ATE/CATE estimation.

Two datasets share the estimators and metrics; each keeps its own fixed seed and size:

- ``--dataset synthetic`` (default): fully simulated covariates, uplift and
  selection-bias scenarios.
- ``--dataset diabetes``: real ``sklearn`` Diabetes covariates and baseline outcomes
  with simulated treatment assignment, so ground-truth effects stay known.
"""

import argparse
from collections.abc import Callable
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

EXAMPLE_PATH = "examples/causal_dr_uplift_comparison.py"

Scenario = dict[str, torch.Tensor]


@dataclass(frozen=True)
class CausalDRConfig:
    """Synthetic-covariate configuration (``--dataset synthetic``)."""

    seed: int = 260227
    n_samples: int = 1200
    n_features: int = 6
    folds: int = 3
    alpha: float = 0.05


@dataclass(frozen=True)
class CausalDRRealDataConfig:
    """Diabetes-covariate configuration (``--dataset diabetes``)."""

    seed: int = 260305
    n_samples: int = 420
    folds: int = 3
    alpha: float = 0.05


AnyConfig = CausalDRConfig | CausalDRRealDataConfig


def _fit_models():
    from sklearn.linear_model import (  # type: ignore[import-untyped]
        LinearRegression,
        LogisticRegression,
    )

    return LinearRegression, LogisticRegression


# ---------------------------------------------------------------------------
# Synthetic scenarios
# ---------------------------------------------------------------------------


def _simulate_uplift(cfg: CausalDRConfig) -> Scenario:
    torch.manual_seed(cfg.seed)
    x = torch.randn(cfg.n_samples, cfg.n_features)
    base = 0.5 * x[:, 0] - 0.4 * x[:, 1] + 0.3 * x[:, 2] ** 2
    tau = 0.4 + 0.3 * torch.tanh(x[:, 0])
    p = torch.sigmoid(0.9 * x[:, 0] - 0.7 * x[:, 1] + 0.25 * x[:, 2]).clamp(0.03, 0.97)
    t = torch.bernoulli(p)
    y0 = base + 0.25 * torch.randn(cfg.n_samples)
    y = y0 + t * tau
    return {"x": x, "t": t, "y": y, "tau": tau, "p": p}


def _simulate_selection_bias(cfg: CausalDRConfig) -> Scenario:
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


def _synthetic_scenarios(cfg: CausalDRConfig) -> list[tuple[str, Scenario]]:
    return [
        ("Uplift", _simulate_uplift(cfg)),
        ("SelectionBias", _simulate_selection_bias(cfg)),
    ]


# ---------------------------------------------------------------------------
# Diabetes-covariate scenarios
# ---------------------------------------------------------------------------


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


def _diabetes_proxy(cfg: CausalDRRealDataConfig) -> Scenario:
    x, baseline = _real_covariates(cfg)
    g = torch.Generator().manual_seed(cfg.seed + 19)

    tau = 0.18 + 0.16 * torch.tanh(2.4 * x[:, 0])
    p = torch.sigmoid(1.15 * x[:, 0] - 0.9 * x[:, 1] + 0.25 * x[:, 2]).clamp(0.03, 0.97)
    t = torch.bernoulli(p, generator=g)
    y0 = baseline + 0.18 * torch.randn(baseline.shape, generator=g)
    y = y0 + t * tau
    return {"x": x, "t": t, "y": y, "tau": tau, "p": p}


def _diabetes_selection_bias(cfg: CausalDRRealDataConfig) -> Scenario:
    x, baseline = _real_covariates(cfg)
    g = torch.Generator().manual_seed(cfg.seed + 31)

    tau = 0.10 + 0.20 * torch.sigmoid(2.0 * x[:, 3])
    p = torch.sigmoid(-1.0 * x[:, 0] + 1.2 * x[:, 2] - 0.6 * x[:, 4]).clamp(0.03, 0.97)
    t = torch.bernoulli(p, generator=g)
    y0 = baseline + 0.25 * x[:, 2] - 0.2 * x[:, 4] + 0.16 * torch.randn(baseline.shape, generator=g)
    y = y0 + t * tau
    return {"x": x, "t": t, "y": y, "tau": tau, "p": p}


def _diabetes_scenarios(cfg: CausalDRRealDataConfig) -> list[tuple[str, Scenario]]:
    return [
        ("DiabetesProxy", _diabetes_proxy(cfg)),
        ("DiabetesSelectionBias", _diabetes_selection_bias(cfg)),
    ]


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    config_cls: type[AnyConfig]
    make_scenarios: Callable[..., list[tuple[str, Scenario]]]
    title: str
    seed_policy: str
    summary_title: str
    task: str
    notes: tuple[str, ...]


DATASETS: dict[str, DatasetSpec] = {
    "synthetic": DatasetSpec(
        config_cls=CausalDRConfig,
        make_scenarios=_synthetic_scenarios,
        title="Causal DR Comparison",
        seed_policy="fixed seeds and shared generated datasets",
        summary_title="Causal DR summary",
        task="Causal inference regression (DR ATE/CATE)",
        notes=(
            "Both scenarios use binary treatment with confounded assignment.",
            "DR methods use cross-fitting by default and report overlap diagnostics.",
            "SelectionBias simulates selection effects based on covariate features.",
        ),
    ),
    "diabetes": DatasetSpec(
        config_cls=CausalDRRealDataConfig,
        make_scenarios=_diabetes_scenarios,
        title="Causal DR Comparison (Real Covariates)",
        seed_policy="fixed seed and shared Diabetes covariate sampling",
        summary_title="Causal DR summary (real covariates)",
        task="Causal inference regression (DR ATE/CATE, real covariates)",
        notes=(
            "Uses real Diabetes covariates and baseline outcomes with simulated treatment assignment.",
            "Ground-truth treatment effects are known by construction for ATE/CATE error checks.",
            "DR methods use cross-fitting and overlap diagnostics in both scenarios.",
        ),
    ),
}


def _dataset_for_config(cfg: AnyConfig) -> DatasetSpec:
    for spec in DATASETS.values():
        if type(cfg) is spec.config_cls:
            return spec
    raise TypeError(f"Unsupported config type: {type(cfg).__name__}")


# ---------------------------------------------------------------------------
# Shared estimators and metrics
# ---------------------------------------------------------------------------


def _naive_ate(t: torch.Tensor, y: torch.Tensor) -> float:
    treated = t > 0.5
    control = ~treated
    if int(treated.sum().item()) == 0 or int(control.sum().item()) == 0:
        return float("nan")
    return float(y[treated].mean().item() - y[control].mean().item())


def _scenario_rows(name: str, data: Scenario, cfg: AnyConfig) -> list[dict[str, object]]:
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
            "Method": f"{name}-DRATE",
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
            "Method": f"{name}-DRCATE",
            "ATE_true": true_ate,
            "ATE_hat": cate_ate,
            "ATE_abs_error": abs(cate_ate - true_ate),
            "CI_contains_true": float(cate_ci_low <= true_ate <= cate_ci_high),
            "CI_width": cate_ci_high - cate_ci_low,
            "OverlapRate": float(cate_result["diagnostics"]["overlap_rate"]),
            "MinESS": float(cate_result["diagnostics"]["min_group_ess"]),
            "train_s": float(cate_s),
            "eval_s": 0.0,
            "Notes": "DR pseudo-outcome regression for CATE with aggregate ATE report",
        }
    )
    return rows


def run_comparison(cfg: AnyConfig) -> tuple[list[dict[str, object]], list[str]]:
    spec = _dataset_for_config(cfg)
    rows: list[dict[str, object]] = []
    for name, scenario in spec.make_scenarios(cfg):
        rows += _scenario_rows(name, scenario, cfg)
    return rows, list(spec.notes)


def main(
    cfg: AnyConfig | None = None,
    summary_json_path: str | None = None,
    *,
    dataset: str = "synthetic",
) -> None:
    """Run the comparison; ``cfg`` (if given) selects the dataset by its type."""
    cfg = cfg or DATASETS[dataset].config_cls()
    spec = _dataset_for_config(cfg)
    rows, notes = run_comparison(cfg)

    print_fairness_notes(
        title=spec.title,
        seed_policy=spec.seed_policy,
        train_budget="shared fold count and nuisance-model classes",
        metric_policy="ATE absolute error, CI coverage/width, overlap diagnostics, runtime",
    )
    print_comparison_summary(
        spec.summary_title,
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
            example=EXAMPLE_PATH,
            task=spec.task,
            config=cfg,
            rows=rows,
            notes=notes,
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run causal DR ATE/CATE comparison")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="synthetic")
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()
    main(dataset=args.dataset, summary_json_path=args.summary_json_path)
