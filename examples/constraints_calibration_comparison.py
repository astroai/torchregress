"""Comparison of constrained heads and post-hoc calibration transforms."""

import argparse
from dataclasses import dataclass

import torch
from torch import Tensor

from torchregress.calibration import (
    IsotonicMeanCalibrator,
    PITCalibrator,
    VarianceTemperatureScaler,
)
from torchregress.comparison import (
    print_comparison_summary,
    print_fairness_notes,
    timed_call,
    write_comparison_summary_json,
)
from torchregress.constraints import (
    BoundedHead,
    NonCrossingSort,
    NonNegativeHead,
    SimplexHead,
    SpectralNormWrapper,
)


@dataclass(frozen=True)
class ConstraintCalibrationConfig:
    seed: int = 260227
    n_cal: int = 512
    n_test: int = 256
    n_features: int = 4


def _gaussian_nll(mean: Tensor, var: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    safe_var = var.clamp_min(eps)
    return 0.5 * (
        torch.log(safe_var)
        + (target - mean) ** 2 / safe_var
        + torch.log(torch.tensor(2.0 * torch.pi))
    )


def _pit_chi2(pit_values: Tensor, bins: int = 10) -> float:
    pit = pit_values.detach().float().reshape(-1)
    hist = torch.histc(pit, bins=bins, min=0.0, max=1.0)
    expected = pit.numel() / bins
    chi2 = torch.sum((hist - expected) ** 2 / max(expected, 1e-8))
    return float(chi2.item())


def _crossing_rate(values: Tensor) -> float:
    if values.shape[-1] < 2:
        return 0.0
    diffs = values[..., 1:] - values[..., :-1]
    return float((diffs < 0).any(dim=-1).float().mean().item())


def run_comparison(cfg: ConstraintCalibrationConfig) -> tuple[list[dict[str, object]], list[str]]:
    torch.manual_seed(cfg.seed)
    n_total = cfg.n_cal + cfg.n_test
    x = torch.randn(n_total, cfg.n_features)

    true_mean = 0.7 * x[:, 0] - 0.5 * x[:, 1] + 0.2 * x[:, 2] ** 2
    true_std = 0.15 + 0.2 * torch.sigmoid(x[:, 0])
    y = true_mean + true_std * torch.randn_like(true_mean)

    # Miscalibrated raw predictions.
    pred_mean_raw = 1.2 * true_mean + 0.25 * torch.randn_like(true_mean)
    pred_var_raw = (0.6 * true_std).pow(2).clamp_min(1e-6)

    mean_cal, mean_test = pred_mean_raw[: cfg.n_cal], pred_mean_raw[cfg.n_cal :]
    var_cal, var_test = pred_var_raw[: cfg.n_cal], pred_var_raw[cfg.n_cal :]
    y_cal, y_test = y[: cfg.n_cal], y[cfg.n_cal :]

    raw_nll = float(_gaussian_nll(mean_test, var_test, y_test).mean().item())
    raw_mae = float(torch.mean(torch.abs(mean_test - y_test)).item())
    raw_pit_test = PITCalibrator.pit_from_gaussian(mean_test, torch.sqrt(var_test), y_test)
    raw_pit_chi2 = _pit_chi2(raw_pit_test)

    # Constraint demo tensors.
    q_raw = torch.stack([mean_test + 0.2, mean_test - 0.1, mean_test + 0.05], dim=-1)
    cross_raw = _crossing_rate(q_raw)

    base_linear = torch.nn.Linear(cfg.n_features, 1)
    raw_head_out = base_linear(x[cfg.n_cal :]).squeeze(-1)
    bound_violation_raw = float(((raw_head_out < 0.0) | (raw_head_out > 1.0)).float().mean().item())

    # Calibration transforms.
    temp_scaler = VarianceTemperatureScaler()
    isotonic = IsotonicMeanCalibrator()
    pit_cal = PITCalibrator()

    def _fit_calibrators() -> None:
        isotonic.fit(mean_cal, y_cal)
        mean_cal_iso = isotonic.transform(mean_cal)
        temp_scaler.fit(mean_cal_iso, var_cal, y_cal)
        pit_train = PITCalibrator.pit_from_gaussian(
            mean_cal_iso,
            torch.sqrt(temp_scaler.transform(var_cal)),
            y_cal,
        )
        pit_cal.fit(pit_train)

    _, train_s = timed_call(_fit_calibrators)

    def _eval_calibrated() -> dict[str, float]:
        mean_iso = isotonic.transform(mean_test)
        var_temp = temp_scaler.transform(var_test)
        pit = PITCalibrator.pit_from_gaussian(mean_iso, torch.sqrt(var_temp), y_test)
        pit_adj = pit_cal.transform(pit)

        q_sorted = NonCrossingSort(dim=-1)(q_raw)
        cross_sorted = _crossing_rate(q_sorted)

        bounded_head = BoundedHead(torch.nn.Linear(cfg.n_features, 1), low=0.0, high=1.0)
        bounded_out = bounded_head(x[cfg.n_cal :]).squeeze(-1)
        bound_violation_bounded = float(
            ((bounded_out < 0.0) | (bounded_out > 1.0)).float().mean().item()
        )

        return {
            "MAE": float(torch.mean(torch.abs(mean_iso - y_test)).item()),
            "NLL": float(_gaussian_nll(mean_iso, var_temp, y_test).mean().item()),
            "PITChi2": _pit_chi2(pit_adj),
            "CrossingRate": cross_sorted,
            "BoundViolation": bound_violation_bounded,
        }

    calibrated_metrics, eval_s = timed_call(_eval_calibrated)

    # Smoke usage for the remaining constraint APIs.
    _ = NonNegativeHead(torch.nn.Linear(cfg.n_features, 1))(x[:8])
    simplex_out = SimplexHead(torch.nn.Linear(cfg.n_features, 3))(x[:8])
    _ = SpectralNormWrapper(torch.nn.Linear(cfg.n_features, 1))(x[:8])
    simplex_sum_err = float(torch.mean(torch.abs(simplex_out.sum(dim=-1) - 1.0)).item())

    rows = [
        {
            "Method": "Raw",
            "MAE": raw_mae,
            "NLL": raw_nll,
            "PITChi2": raw_pit_chi2,
            "CrossingRate": cross_raw,
            "BoundViolation": bound_violation_raw,
            "train_s": 0.0,
            "eval_s": 0.0,
            "Notes": "uncalibrated, unconstrained outputs",
        },
        {
            "Method": "Calibrated+Constrained",
            **calibrated_metrics,
            "train_s": float(train_s),
            "eval_s": float(eval_s),
            "Notes": "variance temp + isotonic + PIT + sorted/bounded heads",
        },
    ]

    notes = [
        f"Simplex head average sum error: {simplex_sum_err:.6f}",
        "Constraint APIs used: NonNegativeHead, BoundedHead, SimplexHead, NonCrossingSort, SpectralNormWrapper.",
        "Calibration APIs used: VarianceTemperatureScaler, IsotonicMeanCalibrator, PITCalibrator.",
    ]
    return rows, notes


def main(
    cfg: ConstraintCalibrationConfig | None = None,
    summary_json_path: str | None = None,
) -> None:
    cfg = cfg or ConstraintCalibrationConfig()
    rows, notes = run_comparison(cfg)

    print_fairness_notes(
        title="Constraints + Calibration Comparison",
        seed_policy="fixed seed and shared synthetic split",
        train_budget="shared calibrator fitting stage",
        metric_policy="MAE, NLL, PIT chi-square, crossing/bound violations, runtime",
    )
    print_comparison_summary(
        "Constraint/calibration summary",
        rows,
        metric_order=[
            "MAE",
            "NLL",
            "PITChi2",
            "CrossingRate",
            "BoundViolation",
            "train_s",
            "eval_s",
        ],
    )

    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/constraints_calibration_comparison.py",
            task="Output constraints + post-hoc calibration transforms",
            config=cfg,
            rows=rows,
            notes=notes,
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run constraints/calibration comparison")
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()
    main(summary_json_path=args.summary_json_path)
