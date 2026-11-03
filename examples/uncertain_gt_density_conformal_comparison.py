"""Shared-budget comparison for uncertain labels and density-aware conformal methods.

Two datasets are available::

    python examples/uncertain_gt_density_conformal_comparison.py --dataset synthetic
    python examples/uncertain_gt_density_conformal_comparison.py --dataset diabetes

``synthetic`` simulates latent clean targets, heteroscedastic label noise, an
imperfect predictor, teacher/pseudo labels, and MC predictive draws directly.
``diabetes`` adds feature-dependent annotation noise to sklearn Diabetes targets,
fits a small MLP on the noisy train labels, and derives the predictive variance,
pseudo labels, and MC draws from that predictor. Both produce the same set of
calibration/test inputs, which every conformal method and uncertain-label loss
then shares.
"""

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

DATASETS = ("synthetic", "diabetes")
EXAMPLE = "examples/uncertain_gt_density_conformal_comparison.py"


@dataclass(frozen=True)
class UncertainGTConformalConfig:
    """Fully simulated task (``--dataset synthetic``)."""

    seed: int = 260227
    n_cal: int = 400
    n_test: int = 300
    n_features: int = 5
    n_mc_samples: int = 24
    alpha: float = 0.1


@dataclass(frozen=True)
class UncertainGTConformalRealDataConfig:
    """sklearn Diabetes with annotation noise and a fitted MLP (``--dataset diabetes``)."""

    seed: int = 260305
    n_train: int = 240
    n_cal: int = 120
    n_test: int = 80
    hidden: int = 32
    epochs: int = 24
    lr: float = 8e-3
    n_mc_samples: int = 24
    alpha: float = 0.1


UncertainGTConfig = UncertainGTConformalConfig | UncertainGTConformalRealDataConfig


def default_config(dataset: str) -> UncertainGTConfig:
    if dataset == "synthetic":
        return UncertainGTConformalConfig()
    if dataset == "diabetes":
        return UncertainGTConformalRealDataConfig()
    raise ValueError(f"Unknown dataset {dataset!r}; expected one of {DATASETS}.")


@dataclass
class ConformalInputs:
    """Everything the shared comparison needs; tensors are ``[N, 1]`` (MC: ``[S, N, 1]``)."""

    pred_cal: Tensor
    pred_test: Tensor
    pred_var_test: Tensor
    y_obs_cal: Tensor
    y_obs_test: Tensor
    y_obs_var_test: Tensor
    y_clean_test: Tensor
    teacher_mean_test: Tensor
    pseudo_target_test: Tensor
    pseudo_conf_test: Tensor
    label_mask_test: Tensor
    mc_cal: Tensor
    mc_test: Tensor
    predictor_train_s: float = 0.0


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------


def _synthetic_inputs(cfg: UncertainGTConformalConfig) -> ConformalInputs:
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

    def cal(t: Tensor) -> Tensor:
        return t[:split].unsqueeze(-1)

    def test(t: Tensor) -> Tensor:
        return t[split:].unsqueeze(-1)

    return ConformalInputs(
        pred_cal=cal(pred_mean),
        pred_test=test(pred_mean),
        pred_var_test=test(pred_var),
        y_obs_cal=cal(y_obs),
        y_obs_test=test(y_obs),
        y_obs_var_test=test(y_obs_var),
        y_clean_test=test(y_true),
        teacher_mean_test=test(teacher_mean),
        pseudo_target_test=test(pseudo_target),
        pseudo_conf_test=test(pseudo_conf),
        label_mask_test=test(label_mask),
        mc_cal=mc_samples[:, :split].unsqueeze(-1),
        mc_test=mc_samples[:, split:].unsqueeze(-1),
    )


# ---------------------------------------------------------------------------
# Diabetes dataset
# ---------------------------------------------------------------------------


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
        "y_clean_test": y_clean[cal_end:],
        "y_obs_train": y_obs[:train_end],
        "y_obs_cal": y_obs[train_end:cal_end],
        "y_obs_test": y_obs[cal_end:],
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


def _diabetes_inputs(cfg: UncertainGTConformalRealDataConfig) -> ConformalInputs:
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

    return ConformalInputs(
        pred_cal=pred_cal,
        pred_test=pred_test,
        pred_var_test=pred_var_test,
        y_obs_cal=data["y_obs_cal"],
        y_obs_test=data["y_obs_test"],
        y_obs_var_test=data["y_obs_var_test"],
        y_clean_test=data["y_clean_test"],
        teacher_mean_test=teacher_mean_test,
        pseudo_target_test=pseudo_target_test,
        pseudo_conf_test=pseudo_conf_test,
        label_mask_test=label_mask_test,
        mc_cal=mc_cal,
        mc_test=mc_test,
        predictor_train_s=float(predictor_train_s),
    )


# ---------------------------------------------------------------------------
# Shared comparison
# ---------------------------------------------------------------------------


def _coverage_and_width(lower: Tensor, upper: Tensor, y_true: Tensor) -> tuple[float, float]:
    covered = ((y_true >= lower) & (y_true <= upper)).float().mean().item()
    width = (upper - lower).mean().item()
    return float(covered), float(width)


def _uncertain_losses(mean: Tensor, inputs: ConformalInputs) -> dict[str, float]:
    y_pred = torch.cat([mean, torch.log(inputs.pred_var_test.clamp_min(1e-8))], dim=-1)
    noisy_nll = NoisyTargetGaussianNLL()
    consistency = ConsistencyRegLoss(consistency_weight=0.6)
    pseudo_nll = PseudoLabelNLL(pseudo_weight=0.6)

    noisy_val = noisy_nll(y_pred, inputs.y_obs_test, target_variance=inputs.y_obs_var_test)
    consistency_val = consistency(mean, inputs.y_obs_test, inputs.teacher_mean_test)
    pseudo_val = pseudo_nll(
        y_pred,
        inputs.y_obs_test,
        pseudo_target=inputs.pseudo_target_test,
        pseudo_confidence=inputs.pseudo_conf_test,
        label_mask=inputs.label_mask_test,
    )
    return {
        "NoisyTargetNLL": float(noisy_val.item()),
        "ConsistencyLoss": float(consistency_val.item()),
        "PseudoLabelNLL": float(pseudo_val.item()),
    }


# Per-dataset row notes and summary notes (kept verbatim from the published runs).
_ROW_NOTES = {
    "synthetic": {
        "SplitConformal": "absolute-residual split conformal baseline",
        "DensityConformal": "density-adaptive residual scaling for long-tail coverage",
        "PrevalenceAdjustedCP": "groupwise alpha scaling by prevalence for rare-target regions",
        "MonteCarloConformal": "MC-sample uncertainty-normalized conformal intervals",
    },
    "diabetes": {
        "SplitConformal": "real-data predictor + split conformal",
        "DensityConformal": "density-adaptive residual scaling",
        "PrevalenceAdjustedCP": "groupwise alpha scaling by prevalence",
        "MonteCarloConformal": "MC sample-based conformal with uncertainty-normalized scores",
    },
}
_SUMMARY_NOTES = {
    "synthetic": [
        "Coverage is evaluated against latent clean targets; calibration uses noisy observed labels.",
        "NoisyTargetNLL adds target-variance to predictive variance before scoring.",
        "PseudoLabelNLL uses observed labels where available and pseudo labels elsewhere.",
    ],
    "diabetes": [
        "Uses Diabetes covariates and targets with feature-dependent annotation noise.",
        "Coverage is evaluated on clean targets; calibration uses noisy observed labels.",
        "All conformal variants share the same predictor and calibration split.",
    ],
}


def dataset_name(cfg: UncertainGTConfig) -> str:
    return "diabetes" if isinstance(cfg, UncertainGTConformalRealDataConfig) else "synthetic"


def run_comparison(cfg: UncertainGTConfig) -> tuple[list[dict[str, object]], list[str]]:
    name = dataset_name(cfg)
    if isinstance(cfg, UncertainGTConformalRealDataConfig):
        inputs = _diabetes_inputs(cfg)
    else:
        inputs = _synthetic_inputs(cfg)
    row_notes = _ROW_NOTES[name]
    pred_cal, pred_test, y_cal = inputs.pred_cal, inputs.pred_test, inputs.y_obs_cal
    y_clean = inputs.y_clean_test

    rows: list[dict[str, object]] = []

    def add_row(
        method: str, cov: float, width: float, losses: dict[str, float], fit_s: float, eval_s: float
    ) -> None:
        rows.append(
            {
                "Method": method,
                "Coverage90": cov,
                "Width90": width,
                **losses,
                # Fitting the shared predictor (Diabetes only) is charged to every method.
                "train_s": float(inputs.predictor_train_s + fit_s),
                "eval_s": float(eval_s),
                "Notes": row_notes[method],
            }
        )

    point_losses = _uncertain_losses(pred_test, inputs)

    split_cp = SplitConformal(alpha=cfg.alpha)
    _, fit_s = timed_call(split_cp.calibrate, pred_cal, y_cal)
    (lower, upper), eval_s = timed_call(split_cp.predict_interval, pred_test)
    add_row(
        "SplitConformal", *_coverage_and_width(lower, upper, y_clean), point_losses, fit_s, eval_s
    )

    density_cp = DensityConformal(alpha=cfg.alpha, bandwidth=0.25)
    _, fit_s = timed_call(density_cp.calibrate, pred_cal, y_cal)
    (lower, upper), eval_s = timed_call(density_cp.predict_interval, pred_test)
    add_row(
        "DensityConformal", *_coverage_and_width(lower, upper, y_clean), point_losses, fit_s, eval_s
    )

    # Prevalence groups: quintiles of the observed calibration labels.
    prev_cp = PrevalenceAdjustedCP(alpha=cfg.alpha, n_bins=5)
    bins = torch.quantile(y_cal.view(-1), torch.linspace(0, 1, 6))[1:-1]
    cal_groups = torch.bucketize(y_cal.view(-1), bins)
    test_groups = torch.bucketize(pred_test.view(-1), bins)
    _, fit_s = timed_call(prev_cp.calibrate, pred_cal, y_cal, groups=cal_groups)
    (lower, upper), eval_s = timed_call(prev_cp.predict_interval, pred_test, groups=test_groups)
    add_row(
        "PrevalenceAdjustedCP",
        *_coverage_and_width(lower, upper, y_clean),
        point_losses,
        fit_s,
        eval_s,
    )

    mc_cp = MonteCarloConformal(alpha=cfg.alpha)
    _, fit_s = timed_call(mc_cp.calibrate, inputs.mc_cal, y_cal)
    (lower, upper), eval_s = timed_call(mc_cp.predict_interval, inputs.mc_test)
    mc_losses = _uncertain_losses(inputs.mc_test.mean(dim=0), inputs)
    add_row(
        "MonteCarloConformal", *_coverage_and_width(lower, upper, y_clean), mc_losses, fit_s, eval_s
    )

    return rows, list(_SUMMARY_NOTES[name])


def main(
    cfg: UncertainGTConfig | None = None,
    summary_json_path: str | None = None,
    *,
    dataset: str = "synthetic",
) -> None:
    """Run the comparison; ``cfg`` (when given) determines the dataset."""
    cfg = cfg or default_config(dataset)
    is_real = dataset_name(cfg) == "diabetes"
    rows, notes = run_comparison(cfg)

    print_fairness_notes(
        title="Uncertain-GT + DensityConformal Comparison" + (" (Real Data)" if is_real else ""),
        seed_policy=(
            "fixed seed and shared Diabetes train/cal/test split"
            if is_real
            else "fixed seed and shared synthetic split"
        ),
        train_budget=(
            "shared predictor fit + shared conformal calibration budget"
            if is_real
            else "shared calibration set and MC budget across methods"
        ),
        metric_policy="coverage/width + uncertain-GT losses + runtime",
    )
    print_comparison_summary(
        "Uncertain-GT/density-conformal summary" + (" (real data)" if is_real else ""),
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
            example=f"{EXAMPLE} --dataset diabetes" if is_real else EXAMPLE,
            task="Uncertain ground-truth + density-aware conformal regression"
            + (" (real-data)" if is_real else ""),
            config=cfg,
            rows=rows,
            notes=notes,
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run uncertain-ground-truth + density-conformal comparison"
    )
    parser.add_argument("--dataset", choices=DATASETS, default="synthetic")
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()
    main(summary_json_path=args.summary_json_path, dataset=args.dataset)
