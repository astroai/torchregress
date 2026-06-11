"""Shared-budget comparison for multimodal regression: SLS vs. CQR vs. CTI."""

import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from comparison_utils import (
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
    timed_call,
    write_comparison_summary_json,
)
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from torchregress.losses import CQR, CTI, SLSConformal, SLSLoss


@dataclass(frozen=True)
class MultimodalComparisonConfig:
    seed: int = 260227
    n_train: int = 800
    n_cal: int = 400
    n_test: int = 400
    n_features: int = 1
    hidden: int = 32
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-2
    tau: float = 0.8  # 80% target coverage


class _MLP(nn.Module):
    def __init__(self, n_features: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _simulate(cfg: MultimodalComparisonConfig) -> dict[str, Tensor]:
    set_comparison_seed(cfg.seed)
    n = cfg.n_train + cfg.n_cal + cfg.n_test
    x = torch.rand(n, cfg.n_features) * 2.0 - 1.0  # U([-1, 1])

    # Conditional bimodal distribution
    # If x < 0: unimodal around -1.0
    # If x >= 0: bimodal around -1.0 and 1.0
    y = torch.empty(n, 1)
    for i in range(n):
        xi = x[i, 0].item()
        noise = torch.randn(1) * 0.15
        if xi < 0:
            y[i, 0] = -1.0 + noise
        else:
            if torch.rand(1).item() < 0.5:
                y[i, 0] = -1.0 + noise
            else:
                y[i, 0] = 1.0 + noise

    split_train = cfg.n_train
    split_cal = cfg.n_train + cfg.n_cal

    return {
        "x_train": x[:split_train],
        "x_cal": x[split_train:split_cal],
        "x_test": x[split_cal:],
        "y_train": y[:split_train],
        "y_cal": y[split_train:split_cal],
        "y_test": y[split_cal:],
    }


def _train_cqr(data: dict[str, Tensor], cfg: MultimodalComparisonConfig) -> tuple[nn.Module, CQR]:
    # CQR model outputs lower/upper quantiles (2 outputs)
    model = _MLP(cfg.n_features, cfg.hidden, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    ds = TensorDataset(data["x_train"], data["y_train"])
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    # Levels: alpha/2 and 1 - alpha/2
    alpha = 1.0 - cfg.tau
    q_lo = alpha / 2.0
    q_hi = 1.0 - alpha / 2.0

    for _ in range(cfg.epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)

            # Pinball loss for quantiles
            diff = yb - preds
            loss_lo = torch.max(q_lo * diff[:, 0], (q_lo - 1.0) * diff[:, 0])
            loss_hi = torch.max(q_hi * diff[:, 1], (q_hi - 1.0) * diff[:, 1])
            loss = (loss_lo + loss_hi).mean()

            loss.backward()
            optimizer.step()

    # Conformalize
    cqr = CQR(alpha=alpha)
    model.eval()
    with torch.no_grad():
        cal_preds = model(data["x_cal"])
    cqr.calibrate(cal_preds, data["y_cal"])
    return model, cqr


def _train_cti(data: dict[str, Tensor], cfg: MultimodalComparisonConfig) -> tuple[nn.Module, CTI]:
    # CTI baseline: fit an MDN/GMM (predict mean1, log_var1, mean2, log_var2, logits)
    # 5 outputs total for a 2-component mixture
    model = _MLP(cfg.n_features, cfg.hidden, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    ds = TensorDataset(data["x_train"], data["y_train"])
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    for _ in range(cfg.epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            m1, lv1, m2, lv2, logits = (
                out[:, 0:1],
                out[:, 1:2],
                out[:, 2:3],
                out[:, 3:4],
                out[:, 4:5],
            )

            # MDN loss (NLL of GMM)
            v1 = torch.exp(lv1).clamp(min=1e-4)
            v2 = torch.exp(lv2).clamp(min=1e-4)
            pi = torch.sigmoid(logits)

            log_p1 = -0.5 * (lv1 + (yb - m1) ** 2 / v1 + np.log(2.0 * np.pi))
            log_p2 = -0.5 * (lv2 + (yb - m2) ** 2 / v2 + np.log(2.0 * np.pi))

            # log sum exp
            log_p = torch.log(pi * torch.exp(log_p1) + (1.0 - pi) * torch.exp(log_p2) + 1e-8)
            loss = -log_p.mean()

            loss.backward()
            optimizer.step()

    # Conformalize
    model.eval()
    with torch.no_grad():
        out_cal = model(data["x_cal"])
        m1, lv1, m2, lv2, logits = (
            out_cal[:, 0:1],
            out_cal[:, 1:2],
            out_cal[:, 2:3],
            out_cal[:, 3:4],
            out_cal[:, 4:5],
        )
        v1 = torch.exp(lv1).clamp(min=1e-4)
        v2 = torch.exp(lv2).clamp(min=1e-4)
        pi = torch.sigmoid(logits)

        log_p1 = -0.5 * (lv1 + (data["y_cal"] - m1) ** 2 / v1 + np.log(2.0 * np.pi))
        log_p2 = -0.5 * (lv2 + (data["y_cal"] - m2) ** 2 / v2 + np.log(2.0 * np.pi))
        log_p_cal = torch.log(pi * torch.exp(log_p1) + (1.0 - pi) * torch.exp(log_p2) + 1e-8)

    alpha = 1.0 - cfg.tau
    cti = CTI(alpha=alpha, grid_size=100)
    cti.calibrate(log_p_cal.squeeze(-1), data["y_cal"])
    return model, cti


def _train_sls(
    data: dict[str, Tensor], cfg: MultimodalComparisonConfig
) -> tuple[SLSLoss, SLSConformal]:
    # SLS: train direct frontier and quantiles using SLSLoss
    # The context prediction model extracts features from X (identity here since X is 1D)
    # The loss function contains the SLS networks
    sls_loss = SLSLoss(
        d=1,
        context_dim=cfg.n_features,
        K=2,  # Union of K=2 components to handle modality
        mode="full",
        hidden_dim=cfg.hidden,
        tau=cfg.tau,
        warmup_steps=100,
        reduction="mean",
    )

    # We optimize the loss function parameters directly (which contains self.frontier and self.quantile_net)
    optimizer = torch.optim.Adam(sls_loss.parameters(), lr=cfg.lr)

    ds = TensorDataset(data["x_train"], data["y_train"])
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    for _ in range(cfg.epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = sls_loss(xb, yb)
            loss.backward()
            optimizer.step()

    # Conformalize
    alpha = 1.0 - cfg.tau
    sls_conformal = SLSConformal(sls_loss, alpha=alpha, grid_size=100)
    sls_conformal.calibrate(data["x_cal"], data["y_cal"])
    return sls_loss, sls_conformal


def _eval_coverage_and_width(
    lower: Tensor,
    upper: Tensor,
    y_test: Tensor,
) -> tuple[float, float]:
    covered = (y_test >= lower) & (y_test <= upper)
    coverage = covered.float().mean().item()
    width = (upper - lower).mean().item()
    return coverage, width


def run_comparison(cfg: MultimodalComparisonConfig) -> list[dict[str, object]]:
    data = _simulate(cfg)

    # 1. Train CQR
    (model_cqr, cqr), train_cqr_s = timed_call(_train_cqr, data, cfg)
    model_cqr.eval()
    with torch.no_grad():
        test_preds = model_cqr(data["x_test"])
    (lower_cqr, upper_cqr), eval_cqr_s = timed_call(cqr.predict_interval, test_preds)
    cov_cqr, width_cqr = _eval_coverage_and_width(lower_cqr, upper_cqr, data["y_test"])

    # 2. Train CTI
    (model_cti, cti), train_cti_s = timed_call(_train_cti, data, cfg)
    model_cti.eval()

    def cti_density_fn(y_g: Tensor, x_t: Tensor) -> Tensor:
        # evaluates density log p(y_g | x_t) for grid
        # y_g: [grid_size], x_t: [batch, context_dim]
        # Returns: [batch, grid_size]
        with torch.no_grad():
            out = model_cti(x_t)
        m1, lv1, m2, lv2, logits = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4], out[:, 4:5]
        v1 = torch.exp(lv1).clamp(min=1e-4)
        v2 = torch.exp(lv2).clamp(min=1e-4)
        pi = torch.sigmoid(logits)

        # Expand shapes for vectorized grid evaluation
        y_g = y_g.unsqueeze(0)  # [1, grid_size]
        log_p1 = -0.5 * (lv1 + (y_g - m1) ** 2 / v1 + np.log(2.0 * np.pi))
        log_p2 = -0.5 * (lv2 + (y_g - m2) ** 2 / v2 + np.log(2.0 * np.pi))

        return torch.log(pi * torch.exp(log_p1) + (1.0 - pi) * torch.exp(log_p2) + 1e-8)

    (lower_cti, upper_cti), eval_cti_s = timed_call(
        cti.predict_intervals_from_density, cti_density_fn, data["x_test"], y_min=-3.0, y_max=3.0
    )
    cov_cti, width_cti = _eval_coverage_and_width(lower_cti, upper_cti, data["y_test"])

    # 3. Train SLS
    (sls_loss, sls_conformal), train_sls_s = timed_call(_train_sls, data, cfg)
    (lower_sls, upper_sls), eval_sls_s = timed_call(
        sls_conformal.predict_interval_from_grid, data["x_test"], y_min=-3.0, y_max=3.0
    )
    cov_sls, width_sls = _eval_coverage_and_width(lower_sls, upper_sls, data["y_test"])

    return [
        {
            "Method": "CQR (Baseline)",
            "Coverage": cov_cqr,
            "AvgWidth": width_cqr,
            "train_s": train_cqr_s,
            "eval_s": eval_cqr_s,
            "Notes": "Equal-tailed quantiles, ignores bimodal gaps",
        },
        {
            "Method": "CTI (Plug-in)",
            "Coverage": cov_cti,
            "AvgWidth": width_cti,
            "train_s": train_cti_s,
            "eval_s": eval_cti_s,
            "Notes": "Density GMM estimation + thresholding",
        },
        {
            "Method": "SLS (Direct)",
            "Coverage": cov_sls,
            "AvgWidth": width_sls,
            "train_s": train_sls_s,
            "eval_s": eval_sls_s,
            "Notes": "Direct volume minimization on Union of 2 flows",
        },
    ]


def main(
    cfg: MultimodalComparisonConfig | None = None, summary_json_path: str | None = None
) -> None:
    cfg = cfg or MultimodalComparisonConfig()
    rows = run_comparison(cfg)

    print_fairness_notes(
        title="Multimodal SLS Regression Comparison",
        seed_policy="fixed seed and shared bimodal split",
        train_budget="comparable epochs and model capacities",
        metric_policy="test set coverage, average prediction interval width, runtime",
    )

    print_comparison_summary(
        "Multimodal method summary",
        rows,
        metric_order=["Coverage", "AvgWidth", "train_s", "eval_s"],
    )

    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/sls_multimodal_regression.py",
            task="Multimodal conditional interval prediction",
            config=cfg,
            rows=rows,
            notes=[
                "SLS uses UnionFrontier with K=2 components.",
                "CTI uses a 2-component MDN as the plug-in density model.",
                "CQR is equal-tailed and must cover the unimodal-to-bimodal gap.",
            ],
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SLS multimodal regression example")
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()
    main(summary_json_path=args.summary_json_path)
