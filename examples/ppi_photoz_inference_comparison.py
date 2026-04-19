"""Prediction-powered inference on a photo-z style synthetic scenario."""

import argparse
from dataclasses import dataclass
from typing import Optional

import torch
from comparison_utils import (
    print_comparison_summary,
    print_fairness_notes,
    write_comparison_summary_json,
)

from torchregress.inference import PPIConfig, ppi_diagnostics, ppi_mean_ci, ppi_quantile_ci


@dataclass(frozen=True)
class PPIPhotoZConfig:
    seed: int = 260227
    n_labeled: int = 200
    n_unlabeled: int = 3000
    alpha: float = 0.1
    q: float = 0.9
    n_boot: int = 500


def _simulate(cfg: PPIPhotoZConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(cfg.seed)
    x_l = torch.randn(cfg.n_labeled, 4)
    x_u = torch.randn(cfg.n_unlabeled, 4)
    beta = torch.tensor([0.4, -0.25, 0.12, 0.05])
    y_l = x_l @ beta + 0.25 * torch.randn(cfg.n_labeled)
    y_u_true = x_u @ beta + 0.25 * torch.randn(cfg.n_unlabeled)
    # Imperfect predictor to emulate realistic photo-z regression noise.
    pred_l = x_l @ beta + 0.33 * torch.randn(cfg.n_labeled)
    pred_u = x_u @ beta + 0.33 * torch.randn(cfg.n_unlabeled)
    return y_l, y_u_true, pred_l, pred_u


def run_comparison(cfg: PPIPhotoZConfig) -> tuple[list[dict[str, object]], list[str]]:
    y_l, y_u_true, pred_l, pred_u = _simulate(cfg)

    # Baseline: labeled-only intervals.
    mean_est = float(y_l.mean().item())
    mean_se = float(y_l.std(unbiased=True).item() / max(cfg.n_labeled**0.5, 1.0))
    z = 1.6448536269514722  # 90% two-sided normal CI
    baseline_mean_lo = mean_est - z * mean_se
    baseline_mean_hi = mean_est + z * mean_se

    q_est = float(torch.quantile(y_l, cfg.q).item())
    q_boot = torch.quantile(
        y_l[torch.randint(0, cfg.n_labeled, (cfg.n_boot, cfg.n_labeled))],
        cfg.q,
        dim=1,
    )
    baseline_q_lo = float(torch.quantile(q_boot, cfg.alpha / 2).item())
    baseline_q_hi = float(torch.quantile(q_boot, 1 - cfg.alpha / 2).item())

    config = PPIConfig(alpha=cfg.alpha, n_boot=cfg.n_boot, seed=cfg.seed)
    ppi_mean = ppi_mean_ci(y_l, pred_l, pred_u, config=config)
    ppi_q = ppi_quantile_ci(
        y_l,
        pred_l,
        pred_u,
        q=cfg.q,
        config=config,
    )
    diag = ppi_diagnostics(y_l, pred_l, pred_u)

    true_mean = float(y_u_true.mean().item())
    true_q = float(torch.quantile(y_u_true, cfg.q).item())

    rows = [
        {
            "Method": "LabeledOnlyMeanCI",
            "Target": "mean",
            "Estimate": mean_est,
            "AbsError": abs(mean_est - true_mean),
            "CIWidth": baseline_mean_hi - baseline_mean_lo,
            "CoversTruth": float(baseline_mean_lo <= true_mean <= baseline_mean_hi),
            "train_s": 0.0,
            "eval_s": 0.0,
            "Notes": "naive labeled-only normal CI",
        },
        {
            "Method": "PPIMeanCI",
            "Target": "mean",
            "Estimate": float(ppi_mean["estimate"]),
            "AbsError": abs(float(ppi_mean["estimate"]) - true_mean),
            "CIWidth": float(ppi_mean["ci_upper"]) - float(ppi_mean["ci_lower"]),
            "CoversTruth": float(
                float(ppi_mean["ci_lower"]) <= true_mean <= float(ppi_mean["ci_upper"])
            ),
            "train_s": 0.0,
            "eval_s": 0.0,
            "Notes": "prediction-powered mean CI",
        },
        {
            "Method": "LabeledOnlyQuantileCI",
            "Target": f"q{int(cfg.q * 100)}",
            "Estimate": q_est,
            "AbsError": abs(q_est - true_q),
            "CIWidth": baseline_q_hi - baseline_q_lo,
            "CoversTruth": float(baseline_q_lo <= true_q <= baseline_q_hi),
            "train_s": 0.0,
            "eval_s": 0.0,
            "Notes": "naive labeled-only bootstrap quantile CI",
        },
        {
            "Method": "PPIQuantileCI",
            "Target": f"q{int(cfg.q * 100)}",
            "Estimate": float(ppi_q["estimate"]),
            "AbsError": abs(float(ppi_q["estimate"]) - true_q),
            "CIWidth": float(ppi_q["ci_upper"]) - float(ppi_q["ci_lower"]),
            "CoversTruth": float(float(ppi_q["ci_lower"]) <= true_q <= float(ppi_q["ci_upper"])),
            "train_s": 0.0,
            "eval_s": 0.0,
            "Notes": "prediction-powered quantile CI",
        },
    ]

    notes = [
        "PPI combines labeled outcomes and unlabeled predictions for interval estimation.",
        (
            "Diagnostics: corr="
            f"{diag['prediction_label_correlation']:.3f}, "
            f"shift={diag['prediction_mean_shift_unlabeled_vs_labeled']:.3f}, "
            f"overlap={diag['prediction_range_overlap_ratio']:.3f}"
        ),
    ]
    return rows, notes


def main(cfg: Optional[PPIPhotoZConfig] = None, summary_json_path: str | None = None) -> None:
    cfg = cfg or PPIPhotoZConfig()
    rows, notes = run_comparison(cfg)

    print_fairness_notes(
        title="PPI Photo-z Inference Comparison",
        seed_policy="fixed seed and shared synthetic generator",
        train_budget="closed-form estimators + bootstrap",
        metric_policy="estimate error, CI width, truth coverage",
    )
    print_comparison_summary(
        "PPI inference summary",
        rows,
        metric_order=["Estimate", "AbsError", "CIWidth", "CoversTruth", "train_s", "eval_s"],
    )

    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/ppi_photoz_inference_comparison.py",
            task="Photo-z population/quantile inference with limited labels",
            config=cfg,
            rows=rows,
            notes=notes,
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction-powered inference comparison.")
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()
    main(summary_json_path=args.summary_json_path)
