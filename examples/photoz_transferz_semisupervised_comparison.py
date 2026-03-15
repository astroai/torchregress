"""Real-data semi-supervised photo-z benchmark on TransferZ fixed splits."""

import argparse
import copy
from dataclasses import dataclass
from typing import Literal

import photoz_benchmark_comparison as pzbase
import torch
from comparison_utils import (
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
    timed_call,
    write_comparison_summary_json,
)
from torch.utils.data import DataLoader, TensorDataset

from torchregress.calibration import VarianceTemperatureScaler
from torchregress.losses import GaussianNLLLoss, PseudoLabelConsistencyLoss, WeightedHuberLoss
from torchregress.utils import generate_pseudo_labels, update_ema_teacher_

LabelPolicy = Literal["random", "highz_scarce"]


@dataclass(frozen=True)
class PhotoZTransferZSemiSupervisedConfig:
    seed: int = 260308
    n_train: int = 1024
    n_cal: int = 256
    n_test: int = 512
    batch_size: int = 64
    epochs: int = 10
    teacher_epochs: int = 12
    lr: float = 2e-3
    hidden: int = 64
    labeled_fractions: tuple[float, ...] = (0.1, 0.25, 0.5)
    label_policy: LabelPolicy = "highz_scarce"
    highz_quantile: float = 0.8
    label_bias_strength: float = 3.0
    label_min_probability: float = 0.05
    pseudo_confidence_threshold: float = 0.35
    selective_confidence_threshold: float = 0.55
    highz_threshold_boost: float = 0.12
    teacher_ensemble_size: int = 3
    perturbation_samples: int = 3
    ema_momentum: float = 0.985
    consistency_noise_scale: float = 0.6
    variance_temperature_max_iter: int = 200
    variance_temperature_lr: float = 0.05
    train_dataset_path: str | None = None
    cal_dataset_path: str | None = None
    test_dataset_path: str | None = None
    require_real_data: bool = False


def _make_splits(cfg: PhotoZTransferZSemiSupervisedConfig) -> dict[str, torch.Tensor]:
    base_cfg = pzbase.PhotoZBenchmarkConfig(
        seed=cfg.seed,
        n_train=cfg.n_train,
        n_cal=cfg.n_cal,
        n_test=cfg.n_test,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        lr=cfg.lr,
        hidden=cfg.hidden,
        train_dataset_path=cfg.train_dataset_path,
        cal_dataset_path=cfg.cal_dataset_path,
        test_dataset_path=cfg.test_dataset_path,
        require_real_data=cfg.require_real_data,
    )
    return pzbase._make_splits(base_cfg)


def _select_label_mask(
    *,
    y_train_raw: torch.Tensor,
    labeled_fraction: float,
    seed: int,
    policy: LabelPolicy,
    highz_quantile: float,
    bias_strength: float,
    min_probability: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    n_train = int(y_train_raw.shape[0])
    n_labeled = max(16, int(round(labeled_fraction * n_train)))
    n_labeled = min(max(n_labeled, 1), n_train - 1)
    z = y_train_raw[:, 0]
    q_hi = torch.quantile(z, highz_quantile)
    high_mask = z >= q_hi

    if policy == "random":
        probs = torch.ones_like(z, dtype=torch.float32)
    elif policy == "highz_scarce":
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
        probs = (1.0 - z_norm).clamp_min(0.0).pow(bias_strength) + min_probability
    else:
        raise ValueError(f"Unsupported label policy: {policy}")

    generator = torch.Generator().manual_seed(seed)
    selected = torch.multinomial(probs, n_labeled, replacement=False, generator=generator)
    mask = torch.zeros(n_train, 1, dtype=torch.bool)
    mask[selected] = True

    labeled_high = high_mask[mask[:, 0]]
    labeled_high_share = float(labeled_high.float().mean().item()) if labeled_high.numel() else 0.0
    train_high_share = float(high_mask.float().mean().item())
    stats = {
        "LabeledFraction": float(n_labeled / n_train),
        "LabeledCount": float(n_labeled),
        "LabeledHighZShare": labeled_high_share,
        "TrainHighZShare": train_high_share,
    }
    return mask, stats


def _labeled_loader(
    splits: dict[str, torch.Tensor],
    label_mask: torch.Tensor,
    *,
    batch_size: int,
    seed: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    labeled_x = splits["x_train"][label_mask[:, 0]]
    labeled_y = splits["y_train"][label_mask[:, 0]]
    return DataLoader(
        TensorDataset(labeled_x, labeled_y),
        batch_size=min(batch_size, int(labeled_x.shape[0])),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )


def _uncertainty_to_confidence(values: torch.Tensor, *, quantile: float = 0.75) -> torch.Tensor:
    scale = torch.quantile(values.detach().reshape(-1), quantile).clamp_min(1e-6)
    return (1.0 - values / scale).clamp(min=0.0, max=1.0)


def _calibrated_teacher_outputs(
    cfg: PhotoZTransferZSemiSupervisedConfig,
    splits: dict[str, torch.Tensor],
    label_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    teacher = pzbase.PhotoZRegressor(
        int(splits["x_train"].shape[1]),
        out_dim=2,
        hidden=cfg.hidden,
    )
    loader = _labeled_loader(
        splits,
        label_mask,
        batch_size=cfg.batch_size,
        seed=cfg.seed + 31,
    )
    _, teacher_train_s = timed_call(
        pzbase._train_supervised_tuple,
        teacher,
        GaussianNLLLoss(),
        loader,
        epochs=max(cfg.teacher_epochs, cfg.epochs),
        lr=cfg.lr,
    )
    with torch.no_grad():
        cal_out = teacher(splits["x_cal"])
        train_out = teacher(splits["x_train"])
    mean_cal, logvar_cal = cal_out[:, :1], cal_out[:, 1:2].clamp(-8.0, 6.0)
    mean_train, logvar_train = train_out[:, :1], train_out[:, 1:2].clamp(-8.0, 6.0)
    scaler, varcal_s = timed_call(
        VarianceTemperatureScaler().fit,
        mean_cal,
        torch.exp(logvar_cal),
        splits["y_cal"],
        max_iter=cfg.variance_temperature_max_iter,
        lr=cfg.variance_temperature_lr,
    )
    calibrated_var_train = scaler.transform(torch.exp(logvar_train))
    calibrated_logvar_train = calibrated_var_train.clamp_min(1e-8).log()
    return mean_train, calibrated_logvar_train, float(teacher_train_s + varcal_s)


def _fit_calibrated_teacher(
    cfg: PhotoZTransferZSemiSupervisedConfig,
    splits: dict[str, torch.Tensor],
    label_mask: torch.Tensor,
    *,
    seed_offset: int,
) -> tuple[torch.nn.Module, VarianceTemperatureScaler, float]:
    labeled_x = splits["x_train"][label_mask[:, 0]]
    labeled_y = splits["y_train"][label_mask[:, 0]]
    labeled_xerr = splits["xerr_train"][label_mask[:, 0]]
    g = torch.Generator().manual_seed(cfg.seed + seed_offset)
    aug_x = labeled_x + 0.35 * torch.randn(labeled_x.shape, generator=g) * labeled_xerr
    teacher = pzbase.PhotoZRegressor(
        int(splits["x_train"].shape[1]),
        out_dim=2,
        hidden=cfg.hidden,
    )
    loader = DataLoader(
        TensorDataset(aug_x, labeled_y),
        batch_size=min(cfg.batch_size, int(aug_x.shape[0])),
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed + seed_offset + 1),
    )
    _, teacher_train_s = timed_call(
        pzbase._train_supervised_tuple,
        teacher,
        GaussianNLLLoss(),
        loader,
        epochs=max(cfg.teacher_epochs, cfg.epochs),
        lr=cfg.lr,
    )
    with torch.no_grad():
        cal_out = teacher(splits["x_cal"])
    mean_cal, logvar_cal = cal_out[:, :1], cal_out[:, 1:2].clamp(-8.0, 6.0)
    scaler, varcal_s = timed_call(
        VarianceTemperatureScaler().fit,
        mean_cal,
        torch.exp(logvar_cal),
        splits["y_cal"],
        max_iter=cfg.variance_temperature_max_iter,
        lr=cfg.variance_temperature_lr,
    )
    return teacher, scaler, float(teacher_train_s + varcal_s)


def _ensemble_predict(
    teachers: list[tuple[torch.nn.Module, VarianceTemperatureScaler]],
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    means: list[torch.Tensor] = []
    vars_: list[torch.Tensor] = []
    with torch.no_grad():
        for teacher, scaler in teachers:
            out = teacher(x)
            mean = out[:, :1]
            logvar = out[:, 1:2].clamp(-8.0, 6.0)
            var = scaler.transform(torch.exp(logvar))
            means.append(mean)
            vars_.append(var)
    return torch.stack(means, dim=0), torch.stack(vars_, dim=0)


def _pseudo_targets(
    cfg: PhotoZTransferZSemiSupervisedConfig,
    splits: dict[str, torch.Tensor],
    label_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]:
    teacher_mean, teacher_logvar, teacher_s = _calibrated_teacher_outputs(cfg, splits, label_mask)
    pseudo_target = splits["y_train"].clone()
    pseudo_confidence = torch.zeros_like(pseudo_target)
    unlabeled_mask = ~label_mask
    if bool(unlabeled_mask.any().item()):
        pseudo_u, conf_u, accepted_u = generate_pseudo_labels(
            teacher_mean[unlabeled_mask[:, 0]],
            log_variance=teacher_logvar[unlabeled_mask[:, 0]],
            confidence_threshold=cfg.pseudo_confidence_threshold,
        )
        if not bool(accepted_u.any().item()):
            accepted_u = torch.ones_like(accepted_u, dtype=torch.bool)
            conf_u = torch.full_like(conf_u, 0.5)
        pseudo_target[unlabeled_mask[:, 0]] = pseudo_u
        pseudo_confidence[unlabeled_mask[:, 0]] = conf_u * accepted_u.to(conf_u.dtype)

    accepted_conf = pseudo_confidence[pseudo_confidence > 0]
    accept_rate = float((pseudo_confidence > 0).float().mean().item())
    mean_conf = float(accepted_conf.mean().item()) if accepted_conf.numel() > 0 else 0.0
    return teacher_mean, pseudo_target, pseudo_confidence, accept_rate, mean_conf, teacher_s


def _selective_feature_aware_pseudo_targets(
    cfg: PhotoZTransferZSemiSupervisedConfig,
    splits: dict[str, torch.Tensor],
    label_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float], float]:
    teachers: list[tuple[torch.nn.Module, VarianceTemperatureScaler]] = []
    teacher_cost = 0.0
    for idx in range(cfg.teacher_ensemble_size):
        teacher, scaler, cost = _fit_calibrated_teacher(
            cfg,
            splits,
            label_mask,
            seed_offset=211 + 17 * idx,
        )
        teachers.append((teacher, scaler))
        teacher_cost += cost

    means, vars_ = _ensemble_predict(teachers, splits["x_train"])
    ensemble_mean = means.mean(dim=0)
    aleatoric_std = vars_.mean(dim=0).sqrt()
    disagreement_std = means.std(dim=0, unbiased=False)

    perturb_preds: list[torch.Tensor] = []
    for k in range(cfg.perturbation_samples):
        g = torch.Generator().manual_seed(cfg.seed + 701 + k)
        x_pert = (
            splits["x_train"]
            + torch.randn(
                splits["x_train"].shape,
                generator=g,
                device=splits["x_train"].device,
                dtype=splits["x_train"].dtype,
            )
            * splits["xerr_train"]
        )
        pert_means, _ = _ensemble_predict(teachers, x_pert)
        perturb_preds.append(pert_means.mean(dim=0))
    feature_stability = torch.stack(perturb_preds, dim=0).std(dim=0, unbiased=False)

    feature_error_mag = splits["xerr_train"].pow(2).mean(dim=1, keepdim=True).sqrt()
    conf_alea = _uncertainty_to_confidence(aleatoric_std)
    conf_dis = _uncertainty_to_confidence(disagreement_std)
    conf_stab = _uncertainty_to_confidence(feature_stability)
    conf_feat = _uncertainty_to_confidence(feature_error_mag)
    combined_conf = (conf_alea * conf_dis * conf_stab * conf_feat).clamp_min(1e-8).pow(0.25)

    pseudo_target = splits["y_train"].clone()
    pseudo_confidence = torch.zeros_like(pseudo_target)
    unlabeled_mask = ~label_mask
    highz_threshold = torch.quantile(splits["y_train_raw"][:, 0], cfg.highz_quantile)
    predicted_raw = pzbase._to_raw_y(ensemble_mean, splits)
    predicted_highz = predicted_raw[:, 0:1] >= highz_threshold
    selective_threshold = torch.full_like(combined_conf, cfg.selective_confidence_threshold)
    selective_threshold = selective_threshold + cfg.highz_threshold_boost * predicted_highz.to(
        combined_conf.dtype
    )
    accepted = (combined_conf >= selective_threshold) & unlabeled_mask
    accept_rate = float(accepted.float().mean().item())
    if accept_rate < 0.05:
        unlabeled_conf = combined_conf[unlabeled_mask[:, 0]]
        fallback_cut = torch.quantile(unlabeled_conf, 0.85) if unlabeled_conf.numel() > 0 else 1.0
        fallback_cut = min(float(fallback_cut), max(cfg.pseudo_confidence_threshold, 0.20))
        fallback_accept = (combined_conf >= fallback_cut) & unlabeled_mask
        accepted = fallback_accept

    pseudo_target[accepted[:, 0]] = ensemble_mean[accepted[:, 0]]
    pseudo_confidence[accepted[:, 0]] = combined_conf[accepted[:, 0]]

    low_err_cut = torch.quantile(feature_error_mag[:, 0], 0.25)
    accepted_mask = accepted[:, 0]
    accepted_high_share = (
        float(predicted_highz[accepted_mask].float().mean().item())
        if bool(accepted_mask.any().item())
        else 0.0
    )
    accepted_low_err_share = (
        float((feature_error_mag[accepted_mask, 0] <= low_err_cut).float().mean().item())
        if bool(accepted_mask.any().item())
        else 0.0
    )
    accepted_conf = pseudo_confidence[accepted_mask]
    meta = {
        "PseudoAcceptRate": float(accepted.float().mean().item()),
        "PseudoMeanConfidence": (
            float(accepted_conf.mean().item()) if accepted_conf.numel() > 0 else 0.0
        ),
        "AcceptedHighZShare": accepted_high_share,
        "AcceptedLowErrShare": accepted_low_err_share,
        "TeacherDisagreement": float(disagreement_std.mean().item()),
        "FeatureStability": float(feature_stability.mean().item()),
    }
    return ensemble_mean, pseudo_target, pseudo_confidence, meta, teacher_cost


def _evaluate_point_row(
    *,
    method: str,
    pred: torch.Tensor,
    splits: dict[str, torch.Tensor],
    meta: dict[str, float],
    train_s: float,
    eval_s: float,
    notes: str,
) -> dict[str, object]:
    point = pzbase._point_metrics(pred, splits["y_test"])
    pz = pzbase._photoz_metrics(pred, splits["y_test"], splits)
    return {
        "Method": method,
        **meta,
        **point,
        **pz,
        "NLL": None,
        "Cov90": None,
        "Width90": None,
        "train_s": float(train_s),
        "eval_s": float(eval_s),
        "DataSource": splits["data_source"],
        "Notes": notes,
    }


def _evaluate_catalog_photoz_row(
    splits: dict[str, torch.Tensor],
    meta: dict[str, float],
    labeled_fraction: float,
) -> dict[str, object] | None:
    """Evaluate catalog photo-z (z_phot from table) on the same test samples as all other methods."""
    if "y_phot_test" not in splits or "y_phot_err_test" not in splits:
        return None
    mean_test = splits["y_phot_test"]
    err_test = splits["y_phot_err_test"].clamp_min(1e-8)
    point = pzbase._point_metrics(mean_test, splits["y_test"])
    pz = pzbase._photoz_metrics(mean_test, splits["y_test"], splits)
    lower = mean_test - 1.645 * err_test
    upper = mean_test + 1.645 * err_test
    cov90, width90 = pzbase._coverage_width(lower, upper, splits["y_test"])
    return {
        "Method": "CatalogPhotoZ",
        **meta,
        "LabeledFraction": labeled_fraction,
        **point,
        **pz,
        "NLL": None,
        "Cov90": cov90,
        "Width90": width90,
        "train_s": 0.0,
        "eval_s": 0.0,
        "DataSource": splits["data_source"],
        "Notes": "Catalog photo-z (e.g. Phosphoros) on same test set; all metrics comparable.",
    }


def _train_ema_selective_consistency(
    model: torch.nn.Module,
    splits: dict[str, torch.Tensor],
    *,
    pseudo_target: torch.Tensor,
    pseudo_confidence: torch.Tensor,
    label_mask: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
    ema_momentum: float,
    consistency_noise_scale: float,
) -> torch.nn.Module:
    ema_teacher = copy.deepcopy(model)
    ema_teacher.eval()
    loss_fn = PseudoLabelConsistencyLoss(
        pseudo_weight=0.9,
        consistency_weight=0.35,
        supervised_loss="huber",
        confidence_threshold=0.05,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(
            splits["x_train"],
            splits["xerr_train"],
            splits["y_train"],
            pseudo_target,
            pseudo_confidence,
            label_mask,
        ),
        batch_size=min(batch_size, int(splits["x_train"].shape[0])),
        shuffle=True,
    )
    model.train()
    for _ in range(epochs):
        for xb, xerr_b, yb, pseudo_b, conf_b, label_b in loader:
            opt.zero_grad()
            x_student = xb + consistency_noise_scale * torch.randn_like(xb) * xerr_b
            x_teacher = xb + 0.25 * consistency_noise_scale * torch.randn_like(xb) * xerr_b
            pred = model(x_student)
            with torch.no_grad():
                teacher_pred = ema_teacher(x_teacher)
            loss = loss_fn(
                pred,
                yb,
                pseudo_target=pseudo_b,
                pseudo_confidence=conf_b,
                teacher_pred=teacher_pred,
                label_mask=label_b.to(torch.bool),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            update_ema_teacher_(ema_teacher, model, momentum=ema_momentum)
    model.eval()
    return model


def _evaluate_gaussian_row(
    *,
    method: str,
    model: torch.nn.Module,
    splits: dict[str, torch.Tensor],
    cfg: PhotoZTransferZSemiSupervisedConfig,
    meta: dict[str, float],
    train_s: float,
    notes: str,
) -> dict[str, object]:
    with torch.no_grad():
        cal_out = model(splits["x_cal"])
        test_out = model(splits["x_test"])
    mean_cal, logvar_cal = cal_out[:, :1], cal_out[:, 1:2].clamp(-8.0, 6.0)
    mean_test, logvar_test = test_out[:, :1], test_out[:, 1:2].clamp(-8.0, 6.0)
    scaler, cal_s = timed_call(
        VarianceTemperatureScaler().fit,
        mean_cal,
        torch.exp(logvar_cal),
        splits["y_cal"],
        max_iter=cfg.variance_temperature_max_iter,
        lr=cfg.variance_temperature_lr,
    )
    test_var = scaler.transform(torch.exp(logvar_test))
    test_std = test_var.sqrt()
    resid2 = (splits["y_test"] - mean_test) ** 2
    nll = (0.5 * (torch.log(test_var.clamp_min(1e-8)) + resid2 / test_var)).mean().item()
    lower = mean_test - 1.645 * test_std
    upper = mean_test + 1.645 * test_std
    cov90, width90 = pzbase._coverage_width(lower, upper, splits["y_test"])
    point = pzbase._point_metrics(mean_test, splits["y_test"])
    pz = pzbase._photoz_metrics(mean_test, splits["y_test"], splits)
    return {
        "Method": method,
        **meta,
        **point,
        **pz,
        "NLL": float(nll),
        "Cov90": cov90,
        "Width90": width90,
        "train_s": float(train_s + cal_s),
        "eval_s": 0.0,
        "DataSource": splits["data_source"],
        "Notes": notes,
    }


def run_comparison(
    cfg: PhotoZTransferZSemiSupervisedConfig,
) -> tuple[list[dict[str, object]], list[str]]:
    splits = _make_splits(cfg)
    rows: list[dict[str, object]] = []
    notes = [
        "TransferZ train/cal/test splits are preserved; only the training split is partially labeled.",
        "VALIDATION is used for variance calibration of Gaussian teachers/students before pseudo-label confidence scoring.",
        "The default label policy is high-z-scarce, so labeled coverage of the rare high-z tail is intentionally worse than random masking.",
    ]

    for frac_idx, labeled_fraction in enumerate(cfg.labeled_fractions):
        set_comparison_seed(cfg.seed + frac_idx)
        label_mask, label_stats = _select_label_mask(
            y_train_raw=splits["y_train_raw"],
            labeled_fraction=labeled_fraction,
            seed=cfg.seed + 101 * (frac_idx + 1),
            policy=cfg.label_policy,
            highz_quantile=cfg.highz_quantile,
            bias_strength=cfg.label_bias_strength,
            min_probability=cfg.label_min_probability,
        )
        meta = {
            **label_stats,
            "PseudoAcceptRate": 0.0,
            "PseudoMeanConfidence": 0.0,
            "AcceptedHighZShare": 0.0,
            "AcceptedLowErrShare": 0.0,
            "TeacherDisagreement": 0.0,
            "FeatureStability": 0.0,
        }
        # Catalog photo-z baseline on same test samples (so all metrics are comparable)
        catalog_row = _evaluate_catalog_photoz_row(splits, meta, labeled_fraction)
        if catalog_row is not None:
            rows.append(catalog_row)
        labeled_loader = _labeled_loader(
            splits,
            label_mask,
            batch_size=cfg.batch_size,
            seed=cfg.seed + 11 * (frac_idx + 1),
        )

        huber = pzbase.PhotoZRegressor(int(splits["x_train"].shape[1]), hidden=cfg.hidden)
        _, huber_train_s = timed_call(
            pzbase._train_supervised,
            huber,
            WeightedHuberLoss(delta=1.0),
            labeled_loader,
            epochs=cfg.epochs,
            lr=cfg.lr,
        )
        with torch.no_grad():
            huber_pred = huber(splits["x_test"])
        rows.append(
            _evaluate_point_row(
                method="HuberLabeledOnly",
                pred=huber_pred,
                splits=splits,
                meta=meta,
                train_s=huber_train_s,
                eval_s=0.0,
                notes="labeled-only robust baseline on the observed-label subset",
            )
        )

        gauss = pzbase.PhotoZRegressor(
            int(splits["x_train"].shape[1]), out_dim=2, hidden=cfg.hidden
        )
        _, gauss_train_s = timed_call(
            pzbase._train_supervised_tuple,
            gauss,
            GaussianNLLLoss(),
            labeled_loader,
            epochs=cfg.epochs,
            lr=cfg.lr,
        )
        rows.append(
            _evaluate_gaussian_row(
                method="GaussianLabeledOnly",
                model=gauss,
                splits=splits,
                cfg=cfg,
                meta=meta,
                train_s=gauss_train_s,
                notes="labeled-only Gaussian baseline with validation variance calibration",
            )
        )

        (
            teacher_mean,
            pseudo_target,
            pseudo_confidence,
            accept_rate,
            mean_conf,
            teacher_train_s,
        ) = _pseudo_targets(
            cfg,
            splits,
            label_mask,
        )
        ssl_meta = {
            **label_stats,
            "PseudoAcceptRate": float(accept_rate),
            "PseudoMeanConfidence": float(mean_conf),
            "AcceptedHighZShare": 0.0,
            "AcceptedLowErrShare": 0.0,
            "TeacherDisagreement": 0.0,
            "FeatureStability": 0.0,
        }
        target_all = splits["y_train"].clone()

        pseudo_gauss = pzbase.PhotoZRegressor(
            int(splits["x_train"].shape[1]),
            out_dim=2,
            hidden=cfg.hidden,
        )
        _, pseudo_gauss_train_s = timed_call(
            pzbase._train_pseudo_label_gaussian,
            pseudo_gauss,
            splits,
            target_all=target_all,
            pseudo_target=pseudo_target,
            pseudo_confidence=pseudo_confidence,
            label_mask=label_mask,
            epochs=cfg.epochs,
            lr=cfg.lr,
        )
        rows.append(
            _evaluate_gaussian_row(
                method="PseudoLabelNLL",
                model=pseudo_gauss,
                splits=splits,
                cfg=cfg,
                meta=ssl_meta,
                train_s=teacher_train_s + pseudo_gauss_train_s,
                notes="Gaussian student with calibrated teacher pseudo labels on unlabeled train examples",
            )
        )

        pseudo_consistency = pzbase.PhotoZRegressor(
            int(splits["x_train"].shape[1]),
            hidden=cfg.hidden,
        )
        _, pseudo_consistency_train_s = timed_call(
            pzbase._train_pseudo_label_consistency,
            pseudo_consistency,
            splits,
            target_all=target_all,
            pseudo_target=pseudo_target,
            pseudo_confidence=pseudo_confidence,
            teacher_pred=teacher_mean,
            label_mask=label_mask,
            epochs=cfg.epochs,
            lr=cfg.lr,
        )
        with torch.no_grad():
            consistency_pred = pseudo_consistency(splits["x_test"])
        rows.append(
            _evaluate_point_row(
                method="PseudoLabelConsistency",
                pred=consistency_pred,
                splits=splits,
                meta=ssl_meta,
                train_s=teacher_train_s + pseudo_consistency_train_s,
                eval_s=0.0,
                notes="point student with calibrated pseudo labels plus teacher consistency",
            )
        )

        (
            selective_teacher_mean,
            selective_pseudo_target,
            selective_pseudo_confidence,
            selective_meta,
            selective_teacher_s,
        ) = _selective_feature_aware_pseudo_targets(cfg, splits, label_mask)
        selective_ssl_meta = {**label_stats, **selective_meta}

        selective_gauss = pzbase.PhotoZRegressor(
            int(splits["x_train"].shape[1]),
            out_dim=2,
            hidden=cfg.hidden,
        )
        _, selective_gauss_train_s = timed_call(
            pzbase._train_pseudo_label_gaussian,
            selective_gauss,
            splits,
            target_all=target_all,
            pseudo_target=selective_pseudo_target,
            pseudo_confidence=selective_pseudo_confidence,
            label_mask=label_mask,
            epochs=cfg.epochs,
            lr=cfg.lr,
        )
        rows.append(
            _evaluate_gaussian_row(
                method="SelectivePseudoLabelNLL",
                model=selective_gauss,
                splits=splits,
                cfg=cfg,
                meta=selective_ssl_meta,
                train_s=selective_teacher_s + selective_gauss_train_s,
                notes=(
                    "Gaussian student with ensemble, feature-error-aware selective pseudo labels "
                    "and stricter high-z acceptance"
                ),
            )
        )

        selective_consistency = pzbase.PhotoZRegressor(
            int(splits["x_train"].shape[1]),
            hidden=cfg.hidden,
        )
        _, selective_consistency_train_s = timed_call(
            pzbase._train_pseudo_label_consistency,
            selective_consistency,
            splits,
            target_all=target_all,
            pseudo_target=selective_pseudo_target,
            pseudo_confidence=selective_pseudo_confidence,
            teacher_pred=selective_teacher_mean,
            label_mask=label_mask,
            epochs=cfg.epochs,
            lr=cfg.lr,
        )
        with torch.no_grad():
            clean_pred = selective_consistency(splits["x_test"])
            feature_noise = torch.randn_like(splits["x_test"]) * splits["xerr_test"]
            pert_pred = selective_consistency(splits["x_test"] + feature_noise)
            feature_aware_pred = 0.5 * (clean_pred + pert_pred)
        rows.append(
            _evaluate_point_row(
                method="FeatureAwarePseudoLabelConsistency",
                pred=feature_aware_pred,
                splits=splits,
                meta=selective_ssl_meta,
                train_s=selective_teacher_s + selective_consistency_train_s,
                eval_s=0.0,
                notes=(
                    "point student with selective pseudo labels and feature-error-aware "
                    "prediction averaging at evaluation"
                ),
            )
        )

        ema_selective = pzbase.PhotoZRegressor(
            int(splits["x_train"].shape[1]),
            hidden=cfg.hidden,
        )
        _, ema_train_s = timed_call(
            _train_ema_selective_consistency,
            ema_selective,
            splits,
            pseudo_target=selective_pseudo_target,
            pseudo_confidence=selective_pseudo_confidence,
            label_mask=label_mask,
            epochs=cfg.epochs,
            lr=cfg.lr,
            batch_size=cfg.batch_size,
            ema_momentum=cfg.ema_momentum,
            consistency_noise_scale=cfg.consistency_noise_scale,
        )
        with torch.no_grad():
            ema_clean = ema_selective(splits["x_test"])
            ema_preds = [ema_clean]
            for _ in range(max(cfg.perturbation_samples, 2)):
                x_aug = (
                    splits["x_test"]
                    + 0.5
                    * cfg.consistency_noise_scale
                    * torch.randn_like(splits["x_test"])
                    * splits["xerr_test"]
                )
                ema_preds.append(ema_selective(x_aug))
            ema_pred = torch.stack(ema_preds, dim=0).mean(dim=0)
        rows.append(
            _evaluate_point_row(
                method="EMASelectiveConsistency",
                pred=ema_pred,
                splits=splits,
                meta=selective_ssl_meta,
                train_s=selective_teacher_s + ema_train_s,
                eval_s=0.0,
                notes=(
                    "EMA teacher-student consistency with feature-error perturbations and "
                    "selective pseudo labels"
                ),
            )
        )

    return rows, notes


def main(
    cfg: PhotoZTransferZSemiSupervisedConfig | None = None,
    summary_json_path: str | None = None,
) -> None:
    cfg = cfg or PhotoZTransferZSemiSupervisedConfig()
    rows, notes = run_comparison(cfg)
    print_fairness_notes(
        title="TransferZ Semi-Supervised Photo-z Comparison",
        seed_policy="fixed seed; released TransferZ train/cal/test splits preserved when real data is used",
        train_budget=(
            f"{cfg.epochs} student epochs with the same architecture per method; "
            f"{cfg.teacher_epochs} teacher epochs for pseudo-label rows"
        ),
        metric_policy=(
            "Point metrics plus photo-z domain metrics; Gaussian rows also report NLL and native 90% interval diagnostics"
        ),
    )
    print_comparison_summary(
        "TransferZ Semi-Supervised Photo-z Summary",
        rows,
        metric_order=[
            "LabeledFraction",
            "NMAD",
            "CatastrophicRate",
            "HighZ_MAE",
            "PseudoAcceptRate",
            "PseudoMeanConfidence",
            "train_s",
        ],
    )
    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/photoz_transferz_semisupervised_comparison.py",
            task="TransferZ real-data semi-supervised photometric redshift comparison",
            config=cfg,
            rows=rows,
            notes=notes,
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run semi-supervised photo-z comparison on TransferZ train/cal/test splits."
    )
    parser.add_argument("--summary-json-path", type=str, default=None)
    parser.add_argument("--train-dataset-path", type=str, default=None)
    parser.add_argument("--cal-dataset-path", type=str, default=None)
    parser.add_argument("--test-dataset-path", type=str, default=None)
    parser.add_argument("--require-real-data", action="store_true")
    parser.add_argument("--n-train", type=int, default=1024)
    parser.add_argument("--n-cal", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=512)
    parser.add_argument(
        "--labeled-fractions",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5],
        help="Labeled fractions for SSL (e.g. 0.05 0.1 0.25 0.5 0.75 1.0).",
    )
    args = parser.parse_args()
    main(
        PhotoZTransferZSemiSupervisedConfig(
            n_train=args.n_train,
            n_cal=args.n_cal,
            n_test=args.n_test,
            labeled_fractions=tuple(args.labeled_fractions),
            train_dataset_path=args.train_dataset_path,
            cal_dataset_path=args.cal_dataset_path,
            test_dataset_path=args.test_dataset_path,
            require_real_data=args.require_real_data,
        ),
        summary_json_path=args.summary_json_path,
    )
