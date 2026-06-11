"""Shared-budget uncertain-GT + density conformal comparison on real tabular data."""

import argparse
from dataclasses import dataclass

import torch
from sklearn.datasets import load_diabetes
from torch import Tensor

from torchregress.comparison import (
    print_comparison_summary,
    print_fairness_notes,
    timed_call,
    write_comparison_summary_json,
)
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
class UncertainGTConformalRealDataConfig:
    seed: int = 260305
    n_train: int = 240
    n_cal: int = 120
    n_test: int = 80
    hidden: int = 32
    epochs: int = 24
    lr: float = 8e-3
    n_mc_samples: int = 24
    alpha: float = 0.1


class _MLP(torch.nn.Module):
    def __init__(self, n_features: int, hidden: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _split_realdata(cfg: UncertainGTConformalRealDataConfig) -> dict[str, Tensor]:
    x_np, y_np = load_diabetes(return_X_y=True)
    x = torch.tensor(x_np, dtype=torch.float32)
    y_raw = torch.tensor(y_np, dtype=torch.float32).unsqueeze(-1)

    need = cfg.n_train + cfg.n_cal + cfg.n_test
    if need > x.shape[0]:
        raise ValueError(f"Requested {need} samples but diabetes dataset has {x.shape[0]}.")

    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(x.shape[0], generator=g)[:need]
    x = x[perm]
    y_raw = y_raw[perm]

    # Standardize on train-only clean labels.
    y_train_raw = y_raw[: cfg.n_train]
    y_mean = y_train_raw.mean()
    y_std = y_train_raw.std(unbiased=False).clamp_min(1e-6)
    y_clean = (y_raw - y_mean) / y_std

    # Feature-dependent annotation uncertainty.
    obs_std = (0.08 + 0.22 * torch.sigmoid(2.0 * x[:, 0:1])).clamp_min(0.03)
    y_obs = y_clean + obs_std * torch.randn(y_clean.shape, generator=g)

    train_end = cfg.n_train
    cal_end = train_end + cfg.n_cal

    return {
        "x_train": x[:train_end],
        "x_cal": x[train_end:cal_end],
        "x_test": x[cal_end:],
        "y_clean_train": y_clean[:train_end],
        "y_clean_cal": y_clean[train_end:cal_end],
        "y_clean_test": y_clean[cal_end:],
        "y_obs_train": y_obs[:train_end],
        "y_obs_cal": y_obs[train_end:cal_end],
        "y_obs_test": y_obs[cal_end:],
        "y_obs_var_train": obs_std[:train_end].pow(2),
        "y_obs_var_cal": obs_std[train_end:cal_end].pow(2),
        "y_obs_var_test": obs_std[cal_end:].pow(2),
    }


def _fit_predictor(cfg: UncertainGTConformalRealDataConfig, data: dict[str, Tensor]) -> _MLP:
    model = _MLP(int(data["x_train"].shape[1]), cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()
    for _ in range(cfg.epochs):
        opt.zero_grad(set_to_none=True)
        pred = model(data["x_train"])
        loss = torch.mean((pred - data["y_obs_train"]) ** 2)
        loss.backward()
        opt.step()
    model.eval()
    return model


def _coverage_and_width(lower: Tensor, upper: Tensor, y_true: Tensor) -> tuple[float, float]:
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
    y_pred = torch.cat([mean, torch.log(pred_var.clamp_min(1e-8))], dim=-1)
    noisy_nll = NoisyTargetGaussianNLL()
    consistency = ConsistencyRegLoss(consistency_weight=0.6)
    pseudo_nll = PseudoLabelNLL(pseudo_weight=0.6)

    noisy_val = noisy_nll(y_pred, y_obs, target_variance=y_obs_var)
    consistency_val = consistency(mean, y_obs, teacher_mean)
    pseudo_val = pseudo_nll(
        y_pred,
        y_obs,
        pseudo_target=pseudo_target,
        pseudo_confidence=pseudo_conf,
        label_mask=label_mask,
    )
    return {
        "NoisyTargetNLL": float(noisy_val.item()),
        "ConsistencyLoss": float(consistency_val.item()),
        "PseudoLabelNLL": float(pseudo_val.item()),
    }


def run_comparison(
    cfg: UncertainGTConformalRealDataConfig,
) -> tuple[list[dict[str, object]], list[str]]:
    data = _split_realdata(cfg)
    model, predictor_train_s = timed_call(_fit_predictor, cfg, data)

    with torch.no_grad():
        pred_train = model(data["x_train"])
        pred_cal = model(data["x_cal"])
        pred_test = model(data["x_test"])

    train_resid_var = (pred_train - data["y_obs_train"]).pow(2).mean().clamp_min(1e-4)
    pred_var_cal = (train_resid_var + 0.1 * data["y_obs_var_cal"]).clamp_min(1e-6)
    pred_var_test = (train_resid_var + 0.1 * data["y_obs_var_test"]).clamp_min(1e-6)

    g = torch.Generator().manual_seed(cfg.seed + 13)
    teacher_mean_test = 0.7 * pred_test + 0.3 * data["y_obs_test"]
    pseudo_target_test = teacher_mean_test + 0.05 * torch.randn(
        teacher_mean_test.shape, generator=g
    )
    pseudo_conf_test = torch.sigmoid(2.0 - 3.0 * torch.abs(pseudo_target_test - data["y_obs_test"]))
    pseudo_conf_test = pseudo_conf_test.clamp(0.05, 0.95)

    label_mask_test = torch.zeros_like(data["y_obs_test"], dtype=torch.bool)
    label_mask_test[: (label_mask_test.shape[0] // 2)] = True

    mc_cal = pred_cal.unsqueeze(0) + pred_var_cal.sqrt().unsqueeze(0) * torch.randn(
        cfg.n_mc_samples,
        pred_cal.shape[0],
        pred_cal.shape[1],
        generator=g,
    )
    mc_test = pred_test.unsqueeze(0) + pred_var_test.sqrt().unsqueeze(0) * torch.randn(
        cfg.n_mc_samples,
        pred_test.shape[0],
        pred_test.shape[1],
        generator=g,
    )

    rows: list[dict[str, object]] = []

    split_cp = SplitConformal(alpha=cfg.alpha)
    _, split_train_s = timed_call(split_cp.calibrate, pred_cal, data["y_obs_cal"])
    (split_l, split_u), split_eval_s = timed_call(split_cp.predict_interval, pred_test)
    split_cov, split_w = _coverage_and_width(split_l, split_u, data["y_clean_test"])
    split_losses = _uncertain_losses(
        pred_test,
        pred_var_test,
        data["y_obs_test"],
        data["y_obs_var_test"],
        teacher_mean_test,
        pseudo_target_test,
        pseudo_conf_test,
        label_mask_test,
    )
    rows.append(
        {
            "Method": "SplitConformal",
            "Coverage90": split_cov,
            "Width90": split_w,
            **split_losses,
            "train_s": float(predictor_train_s + split_train_s),
            "eval_s": float(split_eval_s),
            "Notes": "real-data predictor + split conformal",
        }
    )

    density_cp = DensityConformal(alpha=cfg.alpha, bandwidth=0.25)
    _, density_train_s = timed_call(density_cp.calibrate, pred_cal, data["y_obs_cal"])
    (density_l, density_u), density_eval_s = timed_call(density_cp.predict_interval, pred_test)
    density_cov, density_w = _coverage_and_width(density_l, density_u, data["y_clean_test"])
    rows.append(
        {
            "Method": "DensityConformal",
            "Coverage90": density_cov,
            "Width90": density_w,
            **split_losses,
            "train_s": float(predictor_train_s + density_train_s),
            "eval_s": float(density_eval_s),
            "Notes": "density-adaptive residual scaling",
        }
    )

    prev_cp = PrevalenceAdjustedCP(alpha=cfg.alpha, n_bins=5)
    bins = torch.quantile(data["y_obs_cal"].view(-1), torch.linspace(0, 1, 6))[1:-1]
    cal_groups = torch.bucketize(data["y_obs_cal"].view(-1), bins)
    test_groups = torch.bucketize(pred_test.view(-1), bins)

    _, prev_train_s = timed_call(prev_cp.calibrate, pred_cal, data["y_obs_cal"], groups=cal_groups)
    (prev_l, prev_u), prev_eval_s = timed_call(
        prev_cp.predict_interval, pred_test, groups=test_groups
    )
    prev_cov, prev_w = _coverage_and_width(prev_l, prev_u, data["y_clean_test"])
    rows.append(
        {
            "Method": "PrevalenceAdjustedCP",
            "Coverage90": prev_cov,
            "Width90": prev_w,
            **split_losses,
            "train_s": float(predictor_train_s + prev_train_s),
            "eval_s": float(prev_eval_s),
            "Notes": "groupwise alpha scaling by prevalence",
        }
    )

    mc_cp = MonteCarloConformal(alpha=cfg.alpha)
    _, mc_train_s = timed_call(mc_cp.calibrate, mc_cal, data["y_obs_cal"])
    (mc_l, mc_u), mc_eval_s = timed_call(mc_cp.predict_interval, mc_test)
    mc_cov, mc_w = _coverage_and_width(mc_l, mc_u, data["y_clean_test"])

    mc_mean_test = mc_test.mean(dim=0)
    mc_losses = _uncertain_losses(
        mc_mean_test,
        pred_var_test,
        data["y_obs_test"],
        data["y_obs_var_test"],
        teacher_mean_test,
        pseudo_target_test,
        pseudo_conf_test,
        label_mask_test,
    )
    rows.append(
        {
            "Method": "MonteCarloConformal",
            "Coverage90": mc_cov,
            "Width90": mc_w,
            **mc_losses,
            "train_s": float(predictor_train_s + mc_train_s),
            "eval_s": float(mc_eval_s),
            "Notes": "MC sample-based conformal with uncertainty-normalized scores",
        }
    )

    notes = [
        "Uses Diabetes covariates and targets with feature-dependent annotation noise.",
        "Coverage is evaluated on clean targets; calibration uses noisy observed labels.",
        "All conformal variants share the same predictor and calibration split.",
    ]
    return rows, notes


def main(
    cfg: UncertainGTConformalRealDataConfig | None = None,
    summary_json_path: str | None = None,
) -> None:
    cfg = cfg or UncertainGTConformalRealDataConfig()
    rows, notes = run_comparison(cfg)

    print_fairness_notes(
        title="Uncertain-GT + DensityConformal Comparison (Real Data)",
        seed_policy="fixed seed and shared Diabetes train/cal/test split",
        train_budget="shared predictor fit + shared conformal calibration budget",
        metric_policy="coverage/width + uncertain-GT losses + runtime",
    )
    print_comparison_summary(
        "Uncertain-GT/density-conformal summary (real data)",
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
            example="examples/uncertain_gt_density_conformal_realdata_comparison.py",
            task="Uncertain ground-truth + density-aware conformal regression (real-data)",
            config=cfg,
            rows=rows,
            notes=notes,
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run uncertain-ground-truth + density-conformal real-data comparison"
    )
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()
    main(summary_json_path=args.summary_json_path)
