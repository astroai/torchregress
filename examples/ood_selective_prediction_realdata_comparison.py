"""
Real-data OOD/selective prediction comparison (Diabetes regression).

This example uses a deterministic covariate-shift split on the sklearn Diabetes
dataset to compare uncertainty methods on shared budgets with common OOD/selective
metrics and runtime tracking.
"""

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from comparison_utils import (
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
    timed_call,
    write_comparison_summary_json,
)
from sklearn.datasets import load_diabetes

from torchregress.ensemble import SWAG, BayesianNeuralNetwork
from torchregress.metrics import (
    RejectionPolicy,
    ensemble_mean,
    ensemble_variance_decomposition,
    ood_metrics_report,
    risk_coverage_curve,
)


@dataclass(frozen=True)
class OODRealDataConfig:
    seed: int = 321
    n_train: int = 200
    n_cal: int = 60
    n_id_test: int = 60
    n_ood_test: int = 60
    epochs: int = 20
    ensemble_size: int = 3
    mc_samples: int = 12
    lr: float = 0.01
    swag_samples: int = 10
    swag_scale: float = 0.5
    bnn_samples: int = 12
    bnn_beta: float = 0.2
    shift_feature_idx: int = 2
    train_target_noise: float = 0.03
    conformal_alpha: float = 0.1


class PointMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HeteroMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_realdata_splits(cfg: OODRealDataConfig) -> dict[str, torch.Tensor]:
    x_np, y_np = load_diabetes(return_X_y=True)
    x_all = torch.tensor(x_np, dtype=torch.float32)
    y_all = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
    n_total = x_all.shape[0]
    need = cfg.n_train + cfg.n_cal + cfg.n_id_test + cfg.n_ood_test
    if need > n_total:
        raise ValueError(f"Requested {need} samples but diabetes dataset has {n_total}.")

    # OOD pool: largest absolute magnitude along a chosen standardized feature.
    shift_scores = x_all[:, cfg.shift_feature_idx].abs()
    sorted_idx = torch.argsort(shift_scores, descending=True)
    ood_idx = sorted_idx[: cfg.n_ood_test]
    non_ood_idx = sorted_idx[cfg.n_ood_test :]

    g = torch.Generator().manual_seed(cfg.seed)
    non_ood_perm = non_ood_idx[torch.randperm(non_ood_idx.shape[0], generator=g)]
    train_idx = non_ood_perm[: cfg.n_train]
    cal_idx = non_ood_perm[cfg.n_train : cfg.n_train + cfg.n_cal]
    id_idx = non_ood_perm[cfg.n_train + cfg.n_cal : cfg.n_train + cfg.n_cal + cfg.n_id_test]
    ood_idx = ood_idx[torch.randperm(ood_idx.shape[0], generator=g)]

    x_train = x_all[train_idx]
    x_cal = x_all[cal_idx]
    x_id = x_all[id_idx]
    x_ood = x_all[ood_idx]
    y_train_raw = y_all[train_idx]
    y_cal_raw = y_all[cal_idx]
    y_id_raw = y_all[id_idx]
    y_ood_raw = y_all[ood_idx]

    # Standardize targets using clean train labels.
    y_mean = y_train_raw.mean()
    y_std = y_train_raw.std(unbiased=False).clamp_min(1e-6)
    y_train = (y_train_raw - y_mean) / y_std
    y_cal = (y_cal_raw - y_mean) / y_std
    y_id = (y_id_raw - y_mean) / y_std
    y_ood = (y_ood_raw - y_mean) / y_std

    # Mild observation noise on training targets only to avoid overfitting to a tiny table.
    if cfg.train_target_noise > 0:
        noise = torch.randn(y_train.shape, generator=g, device=y_train.device, dtype=y_train.dtype)
        y_train = y_train + cfg.train_target_noise * noise

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_cal": x_cal,
        "y_cal": y_cal,
        "x_id": x_id,
        "y_id": y_id,
        "x_ood": x_ood,
        "y_ood": y_ood,
    }


def _train_point_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    epochs: int,
    lr: float,
) -> nn.Module:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        opt.step()
    model.eval()
    return model


def _train_hetero_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    epochs: int,
    lr: float,
) -> nn.Module:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(x_train)
        mean = out[:, :1]
        log_var = out[:, 1:2].clamp(min=-8.0, max=4.0)
        var = torch.exp(log_var)
        loss = 0.5 * (log_var + (y_train - mean) ** 2 / (var + 1e-6))
        loss.mean().backward()
        opt.step()
    model.eval()
    return model


def _train_swag_model(
    swag: SWAG,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    epochs: int,
    lr: float,
) -> SWAG:
    opt = torch.optim.Adam(swag.base_model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    warmup_epoch = max(1, epochs // 3)
    swag.base_model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        pred = swag.base_model(x_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        opt.step()
        if epoch >= warmup_epoch:
            swag.collect_model(swag.base_model)
    if int(swag.n_models.item()) == 0:
        swag.collect_model(swag.base_model)
    swag.base_model.eval()
    return swag


def _train_bnn_model(
    model: BayesianNeuralNetwork,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    beta: float,
) -> BayesianNeuralNetwork:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(x_train)
        loss = model.elbo_loss(pred, y_train, n_data=x_train.shape[0], beta=beta)
        loss.backward()
        opt.step()
    model.eval()
    return model


def _mc_dropout_predict(
    model: nn.Module, x: torch.Tensor, n_samples: int
) -> tuple[torch.Tensor, torch.Tensor]:
    model.train()
    preds = [model(x) for _ in range(n_samples)]
    stacked = torch.stack(preds)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0).clamp(min=1e-8)
    model.eval()
    return mean, std


def _swag_predict(
    swag_model: SWAG, x: torch.Tensor, n_samples: int, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            swag_model.sample(scale=scale)
            preds.append(swag_model(x))
    stacked = torch.stack(preds)
    return stacked.mean(dim=0), stacked.std(dim=0).clamp(min=1e-8)


def _compute_selective_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    uncertainty: torch.Tensor,
) -> dict[str, float]:
    rcc = risk_coverage_curve(y_pred, y_true, uncertainty.view(-1), n_points=25)
    policy = RejectionPolicy(fraction=0.2)
    policy.update(y_pred, y_true, uncertainty.view(-1))
    policy_res = policy.compute()
    return {
        "AURC": float(rcc["aurc"].item()),
        "rej20_risk": float(policy_res["mean_risk"].item()),
        "rej20_cov": float(policy_res["coverage"].item()),
    }


def _conformal_interval_metrics(
    pred_cal: torch.Tensor,
    y_cal: torch.Tensor,
    pred_id: torch.Tensor,
    y_id: torch.Tensor,
    pred_ood: torch.Tensor,
    y_ood: torch.Tensor,
    *,
    alpha: float,
) -> dict[str, float]:
    residuals = (y_cal - pred_cal).abs().view(-1)
    q = torch.quantile(residuals, 1.0 - alpha)
    lower_id = pred_id - q
    upper_id = pred_id + q
    lower_ood = pred_ood - q
    upper_ood = pred_ood + q
    cov_id = ((y_id >= lower_id) & (y_id <= upper_id)).float().mean()
    cov_ood = ((y_ood >= lower_ood) & (y_ood <= upper_ood)).float().mean()
    width = torch.mean(upper_id - lower_id)
    return {
        "ConformalCov90_ID": float(cov_id.item()),
        "ConformalCov90_OOD": float(cov_ood.item()),
        "ConformalWidth90_ID": float(width.item()),
    }


def _ood_summary_stats(
    x_train: torch.Tensor,
    x_id: torch.Tensor,
    x_ood: torch.Tensor,
    pred_mean_id: torch.Tensor,
    pred_std_id: torch.Tensor,
    pred_mean_ood: torch.Tensor,
    pred_std_ood: torch.Tensor,
) -> dict[str, float]:
    id_unc = pred_std_id.mean().item()
    ood_unc = pred_std_ood.mean().item()
    gap = ood_unc - id_unc

    train_mean = x_train.mean(dim=0)
    x_centered = x_train - train_mean
    cov = (x_centered.T @ x_centered) / max(1, x_train.shape[0] - 1)

    ood_report = ood_metrics_report(
        model_output=(pred_mean_id, pred_std_id.pow(2)),
        x_test=x_id,
        x_reference=x_train,
        mean=train_mean,
        cov=cov,
    )
    _ = ood_metrics_report(
        model_output=(pred_mean_ood, pred_std_ood.pow(2)),
        x_test=x_ood,
        x_reference=x_train,
        mean=train_mean,
        cov=cov,
    )
    return {
        "unc_id": float(id_unc),
        "unc_ood": float(ood_unc),
        "ood_unc_gap": float(gap),
        "typicality_id": float(
            ood_report.get("typicality_score", torch.tensor(float("nan"))).item()
        ),
        "mahalanobis_id": float(
            ood_report.get("mahalanobis_distance", torch.tensor(float("nan"))).item()
        ),
        "kde_id": float(ood_report.get("kernel_density", torch.tensor(float("nan"))).item()),
    }


def run_comparison(cfg: OODRealDataConfig) -> list[dict[str, float | str]]:
    data = _make_realdata_splits(cfg)
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_cal = data["x_cal"]
    y_cal = data["y_cal"]
    x_id = data["x_id"]
    y_id = data["y_id"]
    x_ood = data["x_ood"]
    y_ood = data["y_ood"]
    d_in = int(x_train.shape[1])

    rows: list[dict[str, float | str]] = []

    # Deep ensemble
    ensemble_models: list[nn.Module] = []
    train_total = 0.0
    for i in range(cfg.ensemble_size):
        set_comparison_seed(cfg.seed + 10 + i)
        model = PointMLP(d_in, dropout=0.0)
        _, t = timed_call(_train_point_model, model, x_train, y_train, epochs=cfg.epochs, lr=cfg.lr)
        train_total += t
        ensemble_models.append(model)

    def _eval_deep_ensemble() -> dict[str, float]:
        preds_cal = torch.stack([m(x_cal) for m in ensemble_models])
        preds_id = torch.stack([m(x_id) for m in ensemble_models])
        preds_ood = torch.stack([m(x_ood) for m in ensemble_models])
        mean_cal = ensemble_mean(preds_cal)
        mean_id = ensemble_mean(preds_id)
        mean_ood = ensemble_mean(preds_ood)
        std_id = preds_id.std(dim=0).clamp(min=1e-8)
        std_ood = preds_ood.std(dim=0).clamp(min=1e-8)
        out = {
            "MSE_ID": float(torch.mean((mean_id - y_id) ** 2).item()),
            "MSE_OOD": float(torch.mean((mean_ood - y_ood) ** 2).item()),
        }
        out.update(_compute_selective_metrics(mean_id, y_id, std_id))
        out.update(
            _conformal_interval_metrics(
                mean_cal,
                y_cal,
                mean_id,
                y_id,
                mean_ood,
                y_ood,
                alpha=cfg.conformal_alpha,
            )
        )
        out.update(_ood_summary_stats(x_train, x_id, x_ood, mean_id, std_id, mean_ood, std_ood))
        return out

    deep_stats, deep_eval_s = timed_call(_eval_deep_ensemble)
    rows.append(
        {
            "Method": "DeepEnsemble",
            **deep_stats,
            "train_s": train_total,
            "eval_s": deep_eval_s,
            "Notes": "point ensemble uncertainty via member std",
        }
    )

    # Heteroscedastic ensemble
    hetero_models: list[nn.Module] = []
    train_total = 0.0
    for i in range(cfg.ensemble_size):
        set_comparison_seed(cfg.seed + 100 + i)
        model = HeteroMLP(d_in)
        _, t = timed_call(
            _train_hetero_model, model, x_train, y_train, epochs=cfg.epochs, lr=cfg.lr
        )
        train_total += t
        hetero_models.append(model)

    def _eval_hetero_ensemble() -> dict[str, float]:
        outs_cal = [m(x_cal) for m in hetero_models]
        outs_id = [m(x_id) for m in hetero_models]
        outs_ood = [m(x_ood) for m in hetero_models]
        means_cal = torch.stack([o[:, :1] for o in outs_cal])
        means_id = torch.stack([o[:, :1] for o in outs_id])
        vars_id = torch.stack([torch.exp(o[:, 1:2].clamp(-8.0, 4.0)) for o in outs_id])
        means_ood = torch.stack([o[:, :1] for o in outs_ood])
        vars_ood = torch.stack([torch.exp(o[:, 1:2].clamp(-8.0, 4.0)) for o in outs_ood])
        pred_cal = ensemble_mean(means_cal)
        pred_id = ensemble_mean(means_id)
        pred_ood = ensemble_mean(means_ood)
        epi_id, ale_id = ensemble_variance_decomposition(means_id, vars_id)
        epi_ood, ale_ood = ensemble_variance_decomposition(means_ood, vars_ood)
        std_id = torch.sqrt((epi_id + ale_id).clamp(min=1e-8))
        std_ood = torch.sqrt((epi_ood + ale_ood).clamp(min=1e-8))
        out = {
            "MSE_ID": float(torch.mean((pred_id - y_id) ** 2).item()),
            "MSE_OOD": float(torch.mean((pred_ood - y_ood) ** 2).item()),
            "epi_id": float(torch.sqrt(epi_id.clamp(min=1e-8)).mean().item()),
            "ale_id": float(torch.sqrt(ale_id.clamp(min=1e-8)).mean().item()),
        }
        out.update(_compute_selective_metrics(pred_id, y_id, std_id))
        out.update(
            _conformal_interval_metrics(
                pred_cal,
                y_cal,
                pred_id,
                y_id,
                pred_ood,
                y_ood,
                alpha=cfg.conformal_alpha,
            )
        )
        out.update(_ood_summary_stats(x_train, x_id, x_ood, pred_id, std_id, pred_ood, std_ood))
        return out

    hetero_stats, hetero_eval_s = timed_call(_eval_hetero_ensemble)
    rows.append(
        {
            "Method": "HeteroscedasticEnsemble",
            **hetero_stats,
            "train_s": train_total,
            "eval_s": hetero_eval_s,
            "Notes": "epistemic+aleatoric via ensemble decomposition",
        }
    )

    # MC dropout
    set_comparison_seed(cfg.seed + 500)
    mc_model = PointMLP(d_in, dropout=0.15)
    _, mc_train_s = timed_call(
        _train_point_model, mc_model, x_train, y_train, epochs=cfg.epochs, lr=cfg.lr
    )

    def _eval_mc() -> dict[str, float]:
        mean_cal, _ = _mc_dropout_predict(mc_model, x_cal, cfg.mc_samples)
        mean_id, std_id = _mc_dropout_predict(mc_model, x_id, cfg.mc_samples)
        mean_ood, std_ood = _mc_dropout_predict(mc_model, x_ood, cfg.mc_samples)
        out = {
            "MSE_ID": float(torch.mean((mean_id - y_id) ** 2).item()),
            "MSE_OOD": float(torch.mean((mean_ood - y_ood) ** 2).item()),
        }
        out.update(_compute_selective_metrics(mean_id, y_id, std_id))
        out.update(
            _conformal_interval_metrics(
                mean_cal,
                y_cal,
                mean_id,
                y_id,
                mean_ood,
                y_ood,
                alpha=cfg.conformal_alpha,
            )
        )
        out.update(_ood_summary_stats(x_train, x_id, x_ood, mean_id, std_id, mean_ood, std_ood))
        return out

    mc_stats, mc_eval_s = timed_call(_eval_mc)
    rows.append(
        {
            "Method": "MCDropoutWrapper (proxy)",
            **mc_stats,
            "train_s": mc_train_s,
            "eval_s": mc_eval_s,
            "Notes": "single model + MC sampling; compare against ensembles",
        }
    )

    # SWAG
    set_comparison_seed(cfg.seed + 700)
    swag = SWAG(copy.deepcopy(PointMLP(d_in, dropout=0.0)), max_num_models=max(4, cfg.epochs))
    _, swag_train_s = timed_call(
        _train_swag_model, swag, x_train, y_train, epochs=cfg.epochs, lr=cfg.lr
    )

    def _eval_swag() -> dict[str, float]:
        mean_cal, _ = _swag_predict(swag, x_cal, cfg.swag_samples, cfg.swag_scale)
        mean_id, std_id = _swag_predict(swag, x_id, cfg.swag_samples, cfg.swag_scale)
        mean_ood, std_ood = _swag_predict(swag, x_ood, cfg.swag_samples, cfg.swag_scale)
        out = {
            "MSE_ID": float(torch.mean((mean_id - y_id) ** 2).item()),
            "MSE_OOD": float(torch.mean((mean_ood - y_ood) ** 2).item()),
        }
        out.update(_compute_selective_metrics(mean_id, y_id, std_id))
        out.update(
            _conformal_interval_metrics(
                mean_cal,
                y_cal,
                mean_id,
                y_id,
                mean_ood,
                y_ood,
                alpha=cfg.conformal_alpha,
            )
        )
        out.update(_ood_summary_stats(x_train, x_id, x_ood, mean_id, std_id, mean_ood, std_ood))
        return out

    swag_stats, swag_eval_s = timed_call(_eval_swag)
    rows.append(
        {
            "Method": "SWAG",
            **swag_stats,
            "train_s": swag_train_s,
            "eval_s": swag_eval_s,
            "Notes": "posterior weight sampling; epistemic-focused",
        }
    )

    # BNN
    set_comparison_seed(cfg.seed + 800)
    bnn = BayesianNeuralNetwork(
        input_dim=d_in,
        hidden_dims=[64, 64],
        output_dim=1,
        n_samples=cfg.bnn_samples,
    )
    _, bnn_train_s = timed_call(
        _train_bnn_model,
        bnn,
        x_train,
        y_train,
        epochs=cfg.epochs,
        lr=cfg.lr * 0.75,
        beta=cfg.bnn_beta,
    )

    def _eval_bnn() -> dict[str, float]:
        mean_cal, _ = bnn.predict_with_uncertainty(x_cal, n_samples=cfg.bnn_samples)
        mean_id, std_id = bnn.predict_with_uncertainty(x_id, n_samples=cfg.bnn_samples)
        mean_ood, std_ood = bnn.predict_with_uncertainty(x_ood, n_samples=cfg.bnn_samples)
        std_id = std_id.clamp(min=1e-8)
        std_ood = std_ood.clamp(min=1e-8)
        out = {
            "MSE_ID": float(torch.mean((mean_id - y_id) ** 2).item()),
            "MSE_OOD": float(torch.mean((mean_ood - y_ood) ** 2).item()),
        }
        out.update(_compute_selective_metrics(mean_id, y_id, std_id))
        out.update(
            _conformal_interval_metrics(
                mean_cal,
                y_cal,
                mean_id,
                y_id,
                mean_ood,
                y_ood,
                alpha=cfg.conformal_alpha,
            )
        )
        out.update(_ood_summary_stats(x_train, x_id, x_ood, mean_id, std_id, mean_ood, std_ood))
        return out

    bnn_stats, bnn_eval_s = timed_call(_eval_bnn)
    rows.append(
        {
            "Method": "BayesianNeuralNetwork",
            **bnn_stats,
            "train_s": bnn_train_s,
            "eval_s": bnn_eval_s,
            "Notes": "variational BNN (ELBO) with MC predictive samples",
        }
    )

    return rows


def main(cfg: OODRealDataConfig | None = None, summary_json_path: str | None = None) -> None:
    cfg = cfg or OODRealDataConfig()
    rows = run_comparison(cfg)
    print("OOD / Selective Prediction Comparison (Real Data)")
    print("=" * 58)
    print_fairness_notes(
        title="OOD + Selective Prediction (real-data)",
        seed_policy="fixed seed; shared Diabetes split with deterministic covariate-shift OOD pool",
        train_budget=(
            f"same MLP depth/width; {cfg.epochs} epochs; ensemble size={cfg.ensemble_size}; "
            f"MC samples={cfg.mc_samples}; SWAG samples={cfg.swag_samples}; BNN samples={cfg.bnn_samples}"
        ),
        metric_policy=(
            "ID/OOD MSE, risk-coverage AURC, rejection-policy risk/coverage, OOD uncertainty gap, "
            "split-conformal interval coverage/width, and aggregate OOD metrics runtime-tracked"
        ),
    )
    print_comparison_summary(
        "OOD / Selective Prediction Summary (Real Data)",
        rows,
        metric_order=[
            "MSE_ID",
            "MSE_OOD",
            "AURC",
            "rej20_risk",
            "rej20_cov",
            "ConformalCov90_ID",
            "ConformalCov90_OOD",
            "ConformalWidth90_ID",
            "unc_id",
            "unc_ood",
            "ood_unc_gap",
            "train_s",
            "eval_s",
        ],
    )
    print("\nCaveats:")
    print(
        "- Real-data example uses deterministic covariate-shift split, not a labeled OOD benchmark suite."
    )
    print(
        "- OOD pool is defined by extreme values of a single feature; validate on your deployment shift."
    )
    print(
        "- Use multiple signals (risk-coverage, uncertainty gap, task error), not a single OOD score."
    )
    print("- Conformal coverage can degrade under shift; track ID and OOD coverage separately.")

    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/ood_selective_prediction_realdata_comparison.py",
            task="OOD robustness / selective prediction (real-data)",
            config=cfg,
            rows=rows,
            notes=[
                "Real-data OOD split on sklearn Diabetes via extreme-value covariate shift",
                "Shared budget includes DeepEnsemble, HeteroscedasticEnsemble, MCDropout, SWAG, BNN",
                "Each method reports split-conformal interval quality from a shared calibration split",
            ],
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    main()
