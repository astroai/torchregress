"""Shared-budget comparison for uncertain labels and density-aware conformal methods."""

import argparse
from dataclasses import dataclass

import torch
from comparison_utils import (
    print_comparison_summary,
    print_fairness_notes,
    timed_call,
    write_comparison_summary_json,
)
from torch import Tensor

from torchregress.losses import (
    ConsistencyRegLoss,
    DensityConformal,
    MonteCarloConformal,
    NoisyTargetGaussianNLL,
    PrevalenceAdjustedCP,
    PseudoLabelNLL,
    SplitConformal,
)


@dataclass(frozen=True)
class UncertainGTConformalConfig:
    seed: int = 260227
    n_cal: int = 400
    n_test: int = 300
    n_features: int = 5
    n_mc_samples: int = 24
    alpha: float = 0.1


def _simulate(cfg: UncertainGTConformalConfig) -> dict[str, Tensor]:
    torch.manual_seed(cfg.seed)
    n = cfg.n_cal + cfg.n_test
    x = torch.randn(n, cfg.n_features)

    # Latent clean target with long-tail component.
    base = 0.8 * x[:, 0] - 0.5 * x[:, 1] + 0.3 * x[:, 2] ** 2 + 0.25 * x[:, 3] * x[:, 4]
    tail = 0.9 * torch.sign(x[:, 0]) * torch.abs(x[:, 0]) ** 1.5
    y_true = base + 0.2 * tail

    # Heteroscedastic observation noise (uncertain ground-truth labels).
    obs_std = (0.08 + 0.25 * torch.sigmoid(1.2 * x[:, 0])).clamp_min(0.03)
    y_obs = y_true + obs_std * torch.randn_like(y_true)
    y_obs_var = obs_std**2

    # Imperfect point predictor and predictive variance proxy.
    pred_mean = y_true + 0.18 * torch.randn_like(y_true)
    pred_std = (0.12 + 0.15 * torch.sigmoid(1.0 * x[:, 1])).clamp_min(0.04)
    pred_var = pred_std**2

    # Teacher/pseudo labels for weak supervision signals.
    teacher_mean = y_true + 0.1 * torch.randn_like(y_true)
    pseudo_target = teacher_mean + 0.05 * torch.randn_like(y_true)
    pseudo_conf = torch.sigmoid(1.5 - 2.0 * torch.abs(pseudo_target - y_obs)).clamp(0.05, 0.95)
    label_mask = torch.zeros(n, dtype=torch.bool)
    label_mask[: (n // 2)] = True

    # Monte Carlo predictive samples (e.g., dropout/ensemble draws).
    mc_samples = pred_mean.unsqueeze(0) + pred_std.unsqueeze(0) * torch.randn(
        cfg.n_mc_samples,
        n,
    )

    split = cfg.n_cal
    return {
        "x_cal": x[:split],
        "x_test": x[split:],
        "y_true_cal": y_true[:split],
        "y_true_test": y_true[split:],
        "y_obs_cal": y_obs[:split],
        "y_obs_test": y_obs[split:],
        "y_obs_var_cal": y_obs_var[:split],
        "y_obs_var_test": y_obs_var[split:],
        "pred_mean_cal": pred_mean[:split],
        "pred_mean_test": pred_mean[split:],
        "pred_var_cal": pred_var[:split],
        "pred_var_test": pred_var[split:],
        "teacher_mean_test": teacher_mean[split:],
        "pseudo_target_test": pseudo_target[split:],
        "pseudo_conf_test": pseudo_conf[split:],
        "label_mask_test": label_mask[split:],
        "mc_cal": mc_samples[:, :split],
        "mc_test": mc_samples[:, split:],
    }


def _coverage_and_width(
    lower: Tensor,
    upper: Tensor,
    y_true: Tensor,
) -> tuple[float, float]:
    covered = ((y_true >= lower) & (y_true <= upper)).float().mean().item()
    width = (upper - lower).mean().item()
    return float(covered), float(width)


def _uncertain_losses(
    mean: Tensor,
    pred_var: Tensor,
    y_obs: Tensor,
    y_obs_var: Tensor,
    teacher_mean: Tensor,
    pseudo_target: Tensor,
    pseudo_conf: Tensor,
    label_mask: Tensor,
) -> dict[str, float]:
    y_pred_gauss = torch.stack([mean, torch.log(pred_var.clamp_min(1e-8))], dim=-1).reshape(
        mean.shape[0],
        2,
    )
    noisy_nll = NoisyTargetGaussianNLL()
    consistency = ConsistencyRegLoss(consistency_weight=0.6)
    pseudo_nll = PseudoLabelNLL(pseudo_weight=0.6)
    noisy_val = noisy_nll(
        y_pred_gauss, y_obs.unsqueeze(-1), target_variance=y_obs_var.unsqueeze(-1)
    )
    consistency_val = consistency(
        mean.unsqueeze(-1), y_obs.unsqueeze(-1), teacher_mean.unsqueeze(-1)
    )
    pseudo_val = pseudo_nll(
        y_pred_gauss,
        y_obs.unsqueeze(-1),
        pseudo_target=pseudo_target.unsqueeze(-1),
        pseudo_confidence=pseudo_conf.unsqueeze(-1),
        label_mask=label_mask.unsqueeze(-1),
    )
    return {
        "NoisyTargetNLL": float(noisy_val.item()),
        "ConsistencyLoss": float(consistency_val.item()),
        "PseudoLabelNLL": float(pseudo_val.item()),
    }


def run_comparison(cfg: UncertainGTConformalConfig) -> tuple[list[dict[str, object]], list[str]]:
    data = _simulate(cfg)
    y_cal = data["y_obs_cal"].unsqueeze(-1)
    y_test_true = data["y_true_test"].unsqueeze(-1)
    pred_cal = data["pred_mean_cal"].unsqueeze(-1)
    pred_test = data["pred_mean_test"].unsqueeze(-1)

    rows: list[dict[str, object]] = []

    split_cp = SplitConformal(alpha=cfg.alpha)
    _, split_train_s = timed_call(split_cp.calibrate, pred_cal, y_cal)
    (split_lower, split_upper), split_eval_s = timed_call(split_cp.predict_interval, pred_test)
    split_cov, split_w = _coverage_and_width(split_lower, split_upper, y_test_true)
    split_losses = _uncertain_losses(
        data["pred_mean_test"],
        data["pred_var_test"],
        data["y_obs_test"],
        data["y_obs_var_test"],
        data["teacher_mean_test"],
        data["pseudo_target_test"],
        data["pseudo_conf_test"],
        data["label_mask_test"],
    )
    rows.append(
        {
            "Method": "SplitConformal",
            "Coverage90": split_cov,
            "Width90": split_w,
            **split_losses,
            "train_s": float(split_train_s),
            "eval_s": float(split_eval_s),
            "Notes": "absolute-residual split conformal baseline",
        }
    )

    density_cp = DensityConformal(alpha=cfg.alpha, bandwidth=0.25)
    _, density_train_s = timed_call(density_cp.calibrate, pred_cal, y_cal)
    (density_lower, density_upper), density_eval_s = timed_call(
        density_cp.predict_interval, pred_test
    )
    density_cov, density_w = _coverage_and_width(density_lower, density_upper, y_test_true)
    rows.append(
        {
            "Method": "DensityConformal",
            "Coverage90": density_cov,
            "Width90": density_w,
            **split_losses,
            "train_s": float(density_train_s),
            "eval_s": float(density_eval_s),
            "Notes": "density-adaptive residual scaling for long-tail coverage",
        }
    )

    prev_cp = PrevalenceAdjustedCP(alpha=cfg.alpha, n_bins=5)
    cal_groups = torch.bucketize(
        data["y_obs_cal"], torch.quantile(data["y_obs_cal"], torch.linspace(0, 1, 6))[1:-1]
    )
    test_groups = torch.bucketize(
        data["pred_mean_test"], torch.quantile(data["y_obs_cal"], torch.linspace(0, 1, 6))[1:-1]
    )
    _, prev_train_s = timed_call(prev_cp.calibrate, pred_cal, y_cal, groups=cal_groups)
    (prev_lower, prev_upper), prev_eval_s = timed_call(
        prev_cp.predict_interval,
        pred_test,
        groups=test_groups,
    )
    prev_cov, prev_w = _coverage_and_width(prev_lower, prev_upper, y_test_true)
    rows.append(
        {
            "Method": "PrevalenceAdjustedCP",
            "Coverage90": prev_cov,
            "Width90": prev_w,
            **split_losses,
            "train_s": float(prev_train_s),
            "eval_s": float(prev_eval_s),
            "Notes": "groupwise alpha scaling by prevalence for rare-target regions",
        }
    )

    mc_cp = MonteCarloConformal(alpha=cfg.alpha)
    _, mc_train_s = timed_call(mc_cp.calibrate, data["mc_cal"].unsqueeze(-1), y_cal)
    (mc_lower, mc_upper), mc_eval_s = timed_call(
        mc_cp.predict_interval,
        data["mc_test"].unsqueeze(-1),
    )
    mc_cov, mc_w = _coverage_and_width(mc_lower, mc_upper, y_test_true)
    mc_mean_test = data["mc_test"].mean(dim=0)
    mc_losses = _uncertain_losses(
        mc_mean_test,
        data["pred_var_test"],
        data["y_obs_test"],
        data["y_obs_var_test"],
        data["teacher_mean_test"],
        data["pseudo_target_test"],
        data["pseudo_conf_test"],
        data["label_mask_test"],
    )
    rows.append(
        {
            "Method": "MonteCarloConformal",
            "Coverage90": mc_cov,
            "Width90": mc_w,
            **mc_losses,
            "train_s": float(mc_train_s),
            "eval_s": float(mc_eval_s),
            "Notes": "MC-sample uncertainty-normalized conformal intervals",
        }
    )

    notes = [
        "Coverage is evaluated against latent clean targets; calibration uses noisy observed labels.",
        "NoisyTargetNLL adds target-variance to predictive variance before scoring.",
        "PseudoLabelNLL uses observed labels where available and pseudo labels elsewhere.",
    ]
    return rows, notes


def main(
    cfg: UncertainGTConformalConfig | None = None,
    summary_json_path: str | None = None,
) -> None:
    cfg = cfg or UncertainGTConformalConfig()
    rows, notes = run_comparison(cfg)

    print_fairness_notes(
        title="Uncertain-GT + DensityConformal Comparison",
        seed_policy="fixed seed and shared synthetic split",
        train_budget="shared calibration set and MC budget across methods",
        metric_policy="coverage/width + uncertain-GT losses + runtime",
    )
    print_comparison_summary(
        "Uncertain-GT/density-conformal summary",
        rows,
        metric_order=[
            "Coverage90",
            "Width90",
            "NoisyTargetNLL",
            "ConsistencyLoss",
            "PseudoLabelNLL",
            "train_s",
            "eval_s",
        ],
    )

    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/uncertain_gt_density_conformal_comparison.py",
            task="Uncertain ground-truth + density-aware conformal regression",
            config=cfg,
            rows=rows,
            notes=notes,
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run uncertain-ground-truth + density-conformal comparison"
    )
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()
    main(summary_json_path=args.summary_json_path)
