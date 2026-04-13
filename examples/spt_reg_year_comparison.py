"""YearPredictionMSD-style real-data benchmark for Shift-Factored Predictive Transport."""

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import spt_reg_synthetic_comparison as sptbase
from comparison_utils import (
    print_comparison_summary,
    print_fairness_notes,
    timed_call,
    write_comparison_summary_json,
)

from torchregress.test_time import (
    FeatureStatNormalizer,
    ShiftFactoredPredictiveTransport,
    ShiftFactoredTransportConfig,
    SignificantSubspaceAligner,
)
from torchregress.utils.openml_relaxed import fetch_openml_regression_with_sklearn_fallback


@dataclass(frozen=True)
class SPTRegYearConfig:
    seed: int = 260410
    dataset_path: str | None = None
    cache_path: str | None = None
    allow_download: bool = False
    target_column: str = "target"
    n_source: int = 4096
    n_target_unlabeled: int = 8192
    n_target_cal: int = 2048
    n_target_test: int = 4096
    shift_feature_idx: int = 0
    alpha: float = 0.1
    ppi_quantile: float = 0.9
    n_support: int = 128
    n_bins: int = 20
    n_samples_eval: int = 64
    target_label_budget: int = 512
    openml_data_id: int | None = None
    openml_dataset_name: str | None = None
    openml_version: int = 1
    max_dataset_rows: int | None = None
    prior_ratio_clip: float = 2.0
    prior_transport_strength: float = 0.5


def _spt_transport_config(cfg: SPTRegYearConfig) -> ShiftFactoredTransportConfig:
    return ShiftFactoredTransportConfig(
        n_support=cfg.n_support,
        alpha=cfg.alpha,
        random_state=cfg.seed,
        prior_ratio_clip=float(cfg.prior_ratio_clip),
        prior_transport_strength=float(cfg.prior_transport_strength),
    )


def spt_year_split_row_budget(cfg: SPTRegYearConfig) -> int:
    return cfg.n_source + cfg.n_target_unlabeled + cfg.n_target_cal + cfg.n_target_test


def spt_year_scale_split_sizes(cfg: SPTRegYearConfig, factor: int) -> SPTRegYearConfig:
    """Multiply pool sizes (and label budget) for stronger Year-track experiments."""
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1:
        return cfg
    return replace(
        cfg,
        n_source=cfg.n_source * factor,
        n_target_unlabeled=cfg.n_target_unlabeled * factor,
        n_target_cal=cfg.n_target_cal * factor,
        n_target_test=cfg.n_target_test * factor,
        target_label_budget=max(1, cfg.target_label_budget * factor),
    )


def _openml_regression_frame_spt(cfg: SPTRegYearConfig) -> tuple[pd.DataFrame, str]:
    if cfg.openml_data_id is None and cfg.openml_dataset_name is None:
        raise ValueError("openml regression fetch requires openml_data_id or openml_dataset_name")
    return fetch_openml_regression_with_sklearn_fallback(
        data_id=cfg.openml_data_id,
        name=cfg.openml_dataset_name,
        version=int(cfg.openml_version),
        target_column=cfg.target_column,
    )


def _load_dataset_frame(cfg: SPTRegYearConfig) -> tuple[pd.DataFrame, str]:
    if cfg.dataset_path and (cfg.openml_data_id is not None or cfg.openml_dataset_name is not None):
        raise ValueError("dataset_path cannot be combined with openml_data_id/openml_dataset_name")

    if cfg.dataset_path:
        path = Path(cfg.dataset_path)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path), str(path)
        return pd.read_csv(path), str(path)

    cache_path = Path(cfg.cache_path) if cfg.cache_path else None
    if cache_path is not None and cache_path.exists():
        if cache_path.suffix.lower() == ".parquet":
            return pd.read_parquet(cache_path), str(cache_path)
        return pd.read_csv(cache_path), str(cache_path)

    if cfg.openml_data_id is not None or cfg.openml_dataset_name is not None:
        if not cfg.allow_download:
            raise FileNotFoundError(
                "OpenML fetch requested but allow_download is False and cache is missing"
            )
        frame, tag = _openml_regression_frame_spt(cfg)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if cache_path.suffix.lower() == ".parquet":
                frame.to_parquet(cache_path, index=False)
            else:
                frame.to_csv(cache_path, index=False)
        return frame, tag

    if not cfg.allow_download:
        raise FileNotFoundError("dataset_path/cache_path missing and allow_download is False")

    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(name="year", version=1, as_frame=True)
    features = cast(pd.DataFrame, bunch.data).copy()
    target = pd.to_numeric(cast(pd.Series, bunch.target), errors="raise")
    frame = features.copy()
    frame[cfg.target_column] = target.to_numpy()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.suffix.lower() == ".parquet":
            frame.to_parquet(cache_path, index=False)
        else:
            frame.to_csv(cache_path, index=False)
    return frame, "OpenML:year"


def _make_year_split(cfg: SPTRegYearConfig) -> tuple[dict[str, np.ndarray], str]:
    frame, dataset_name = _load_dataset_frame(cfg)
    if cfg.target_column not in frame.columns:
        raise ValueError(f"target column {cfg.target_column!r} not found in dataset")

    feature_frame = frame.drop(columns=[cfg.target_column]).select_dtypes(include=["number"])
    if feature_frame.empty:
        raise ValueError("dataset must contain numeric feature columns")
    x = feature_frame.to_numpy(dtype=np.float32, copy=True)
    y = frame[cfg.target_column].to_numpy(dtype=np.float32, copy=True)

    n_total = int(x.shape[0])
    rng = np.random.default_rng(cfg.seed)
    if cfg.max_dataset_rows is not None and n_total > cfg.max_dataset_rows:
        pick = rng.choice(n_total, size=cfg.max_dataset_rows, replace=False)
        x = x[pick]
        y = y[pick]
        n_total = int(x.shape[0])

    need = spt_year_split_row_budget(cfg)
    if need > n_total:
        raise ValueError(f"Requested {need} rows but dataset has {n_total}.")
    if not 0 <= cfg.shift_feature_idx < x.shape[1]:
        raise ValueError("shift_feature_idx is out of bounds for the dataset features")

    shift_scores = np.abs(x[:, cfg.shift_feature_idx])
    sorted_idx = np.argsort(shift_scores)[::-1]
    target_pool_size = cfg.n_target_unlabeled + cfg.n_target_cal + cfg.n_target_test
    target_pool_idx = sorted_idx[:target_pool_size]
    source_idx_pool = sorted_idx[target_pool_size:]
    source_idx = rng.choice(source_idx_pool, size=cfg.n_source, replace=False)
    source_idx = np.sort(source_idx)

    target_perm = rng.permutation(target_pool_idx.shape[0])
    target_pool_idx = target_pool_idx[target_perm]
    unlabeled_stop = cfg.n_target_unlabeled
    cal_stop = unlabeled_stop + cfg.n_target_cal

    source_x = x[source_idx]
    source_y_raw = y[source_idx]
    y_mean = float(source_y_raw.mean())
    y_std = float(np.clip(source_y_raw.std(), 1.0e-6, None))
    source_y = (source_y_raw - y_mean) / y_std

    target_pool_x = x[target_pool_idx]
    target_pool_y = (y[target_pool_idx] - y_mean) / y_std
    return (
        {
            "source_x": source_x,
            "source_y": source_y,
            "target_pool_x": target_pool_x,
            "target_pool_y": target_pool_y,
            "unlabeled_stop": np.array([unlabeled_stop]),
            "cal_stop": np.array([cal_stop]),
        },
        dataset_name,
    )


def run_comparison(cfg: SPTRegYearConfig) -> tuple[list[dict[str, object]], list[str]]:
    splits, dataset_name = _make_year_split(cfg)
    source_x = splits["source_x"]
    source_y = splits["source_y"]
    target_pool_x = splits["target_pool_x"]
    target_pool_y = splits["target_pool_y"]
    unlabeled_stop = int(splits["unlabeled_stop"][0])
    cal_stop = int(splits["cal_stop"][0])

    y_cal = target_pool_y[unlabeled_stop:cal_stop]
    y_test = target_pool_y[cal_stop:]
    x_cal = target_pool_x[unlabeled_stop:cal_stop]
    x_test = target_pool_x[cal_stop:]

    fit_out, fit_s = timed_call(sptbase._fit_linear_gaussian, source_x, source_y)
    beta, sigma = fit_out
    bin_edges = np.quantile(source_y, np.linspace(0.0, 1.0, cfg.n_bins + 1))
    bin_edges = np.unique(bin_edges)
    if bin_edges.size < 3:
        bin_edges = np.linspace(
            float(source_y.min()) - 1.0,
            float(source_y.max()) + 1.0,
            cfg.n_bins + 1,
        )
    predictor = sptbase.SyntheticPredictor(beta, sigma, bin_edges)

    source_gaussian_pool = predictor.predict_distribution(target_pool_x, family="gaussian")
    source_binned_pool = predictor.predict_distribution(target_pool_x, family="binnedpdf")
    source_gaussian_cal = sptbase._slice_batch(source_gaussian_pool, unlabeled_stop, cal_stop)
    source_gaussian_test = sptbase._slice_batch(
        source_gaussian_pool, cal_stop, target_pool_x.shape[0]
    )
    source_binned_cal = sptbase._slice_batch(source_binned_pool, unlabeled_stop, cal_stop)
    source_binned_test = sptbase._slice_batch(source_binned_pool, cal_stop, target_pool_x.shape[0])

    rows: list[dict[str, object]] = []
    rows.append(
        sptbase._evaluate_row(
            method="SourceGaussian",
            family="Gaussian",
            batch_cal=source_gaussian_cal,
            batch_test=source_gaussian_test,
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            train_s=fit_s,
            eval_s=0.0,
            notes=f"source linear-Gaussian predictor on {dataset_name}",
        )
    )

    normalizer = FeatureStatNormalizer().fit(source_x)
    norm_pool = predictor.predict_distribution(
        normalizer.transform(target_pool_x), family="gaussian"
    )
    rows.append(
        sptbase._evaluate_row(
            method="FeatureStatNormGaussian",
            family="Gaussian",
            batch_cal=sptbase._slice_batch(norm_pool, unlabeled_stop, cal_stop),
            batch_test=sptbase._slice_batch(norm_pool, cal_stop, target_pool_x.shape[0]),
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            train_s=fit_s,
            eval_s=0.0,
            notes=f"feature-stat normalization baseline on {dataset_name}",
        )
    )

    aligner = SignificantSubspaceAligner(rank=2, random_state=cfg.seed).fit(source_x, source_y)
    aligned_pool = predictor.predict_distribution(
        aligner.transform(target_pool_x), family="gaussian"
    )
    rows.append(
        sptbase._evaluate_row(
            method="SignificantSubspaceGaussian",
            family="Gaussian",
            batch_cal=sptbase._slice_batch(aligned_pool, unlabeled_stop, cal_stop),
            batch_test=sptbase._slice_batch(aligned_pool, cal_stop, target_pool_x.shape[0]),
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            train_s=fit_s,
            eval_s=0.0,
            notes=f"significance-weighted subspace alignment only on {dataset_name}",
        )
    )

    prior_transport = ShiftFactoredPredictiveTransport(
        replace(
            _spt_transport_config(cfg),
            enable_alignment=False,
            enable_uncertainty_inflation=False,
        )
    ).fit_source(
        predictor.predict_distribution(source_x, family="gaussian"),
        source_y,
        source_inputs=source_x,
    )
    prior_pool, prior_eval_s = timed_call(
        prior_transport.adapt_unlabeled_target,
        target_predictions=source_gaussian_pool,
        target_inputs=target_pool_x,
    )
    rows.append(
        sptbase._evaluate_row(
            method="PriorTransportGaussian",
            family="Gaussian",
            batch_cal=sptbase._slice_batch(prior_pool, unlabeled_stop, cal_stop),
            batch_test=sptbase._slice_batch(prior_pool, cal_stop, target_pool_x.shape[0]),
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            train_s=fit_s,
            eval_s=prior_eval_s,
            notes=f"output-space prior transport only on {dataset_name}",
        )
    )

    raw_split = sptbase._manual_split_conformal(
        source_gaussian_cal,
        y_cal,
        source_gaussian_test,
        cfg.alpha,
    )
    rows.append(
        sptbase._evaluate_row(
            method="RawSplitConformalGaussian",
            family="Gaussian",
            batch_cal=source_gaussian_cal,
            batch_test=raw_split,
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            train_s=fit_s,
            eval_s=0.0,
            notes=f"raw source predictions with split conformal on {dataset_name}",
        )
    )

    w_cal = sptbase._covariate_density_ratio_weights(
        source_x,
        target_pool_x[:unlabeled_stop],
        target_pool_x[unlabeled_stop:cal_stop],
        seed=cfg.seed,
    )
    weighted_split = sptbase._weighted_split_conformal(
        source_gaussian_cal,
        y_cal,
        source_gaussian_test,
        cfg.alpha,
        w_cal,
    )
    rows.append(
        sptbase._evaluate_row(
            method="WeightedSplitConformalGaussian",
            family="Gaussian",
            batch_cal=source_gaussian_cal,
            batch_test=weighted_split,
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            train_s=fit_s,
            eval_s=0.0,
            notes=(
                f"source Gaussian + covariate-weighted split conformal on {dataset_name} "
                f"(logistic density-ratio: target-unlabeled vs source)"
            ),
        )
    )

    spt = ShiftFactoredPredictiveTransport(_spt_transport_config(cfg)).fit_source(
        predictor.predict_distribution(source_x, family="gaussian"),
        source_y,
        source_inputs=source_x,
    )
    spt_pool, spt_eval_s = timed_call(
        spt.adapt_unlabeled_target,
        target_predictions=source_gaussian_pool,
        target_inputs=target_pool_x,
        predictor=predictor,
    )
    spt_cal = sptbase._slice_batch(spt_pool, unlabeled_stop, cal_stop)
    spt_test = sptbase._slice_batch(spt_pool, cal_stop, target_pool_x.shape[0])
    rows.append(
        sptbase._evaluate_row(
            method="SPTTransportGaussian",
            family="Gaussian",
            batch_cal=spt_cal,
            batch_test=spt_test,
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            train_s=fit_s,
            eval_s=spt_eval_s,
            notes="SPT adaptation without conformal wrapping (transport-only path)",
        )
    )
    _, spt_cal_s = timed_call(spt.calibrate_target, spt_cal, y_cal)
    rows.append(
        sptbase._evaluate_row(
            method="SPTRegGaussian",
            family="Gaussian",
            batch_cal=spt_cal,
            batch_test=spt.apply_conformal(spt_test),
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            train_s=fit_s + spt_cal_s,
            eval_s=spt_eval_s,
            notes=f"full SPT-Reg on {dataset_name}",
        )
    )

    small_batch, small_s = timed_call(
        sptbase._refit_batch,
        x_cal[: cfg.target_label_budget],
        y_cal[: cfg.target_label_budget],
        x_test,
    )
    rows.append(
        sptbase._evaluate_row(
            method="TargetRefitSmallGaussian",
            family="Gaussian",
            batch_cal=sptbase._refit_batch(
                x_cal[: cfg.target_label_budget],
                y_cal[: cfg.target_label_budget],
                x_cal,
            ),
            batch_test=small_batch,
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            train_s=small_s,
            eval_s=0.0,
            notes=f"small target-label refit baseline on {dataset_name}",
        )
    )

    rows.append(
        sptbase._evaluate_row(
            method="SourceBinnedPDF",
            family="BinnedPDF",
            batch_cal=source_binned_cal,
            batch_test=source_binned_test,
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            train_s=fit_s,
            eval_s=0.0,
            notes=f"ordered-bin predictive law without target adaptation on {dataset_name}",
        )
    )

    spt_binned = ShiftFactoredPredictiveTransport(_spt_transport_config(cfg)).fit_source(
        predictor.predict_distribution(source_x, family="binnedpdf"),
        source_y,
        source_inputs=source_x,
    )
    spt_binned_pool, spt_binned_eval_s = timed_call(
        spt_binned.adapt_unlabeled_target,
        target_predictions=source_binned_pool,
        target_inputs=target_pool_x,
    )
    spt_binned_cal = sptbase._slice_batch(spt_binned_pool, unlabeled_stop, cal_stop)
    spt_binned_test = sptbase._slice_batch(spt_binned_pool, cal_stop, target_pool_x.shape[0])
    _, spt_binned_cal_s = timed_call(spt_binned.calibrate_target, spt_binned_cal, y_cal)
    rows.append(
        sptbase._evaluate_row(
            method="SPTRegBinnedPDF",
            family="BinnedPDF",
            batch_cal=spt_binned_cal,
            batch_test=spt_binned.apply_conformal(spt_binned_test),
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            train_s=fit_s + spt_binned_cal_s,
            eval_s=spt_binned_eval_s,
            notes=f"SPT-Reg applied to ordered-bin predictions on {dataset_name}",
        )
    )

    notes = [
        f"The benchmark uses a deterministic covariate-shift split on {dataset_name}.",
        "Source rows are sampled from the lower-shift region; target rows come from the high-shift region.",
        "The loader supports local CSV/Parquet paths and optional OpenML Year downloads.",
    ]
    return rows, notes


def main(
    cfg: SPTRegYearConfig | None = None,
    summary_json_path: str | None = None,
) -> None:
    cfg = cfg or SPTRegYearConfig()
    rows, notes = run_comparison(cfg)

    print_fairness_notes(
        title="SPT-Reg YearPredictionMSD-style comparison",
        seed_policy=f"fixed seed = {cfg.seed}",
        train_budget="shared linear-Gaussian source backbone and matched target-label budget",
        metric_policy="point, probabilistic, interval, selective, and PPI summaries",
    )
    print_comparison_summary(
        "SPT-Reg YearPredictionMSD-style summary",
        rows,
        metric_order=[
            "MSE",
            "MAE",
            "TailRMSE90",
            "NLL",
            "CRPS",
            "Cov90",
            "Width90",
            "AURC",
            "PPIMeanCIWidth",
            "PPIQuantileCIWidth",
            "train_s",
            "eval_s",
        ],
    )

    if summary_json_path is not None:
        write_comparison_summary_json(
            summary_json_path,
            example="examples/spt_reg_year_comparison.py",
            task="SPT-Reg YearPredictionMSD-style real-data benchmark with BinnedPDF",
            config=cfg,
            rows=rows,
            notes=notes,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the YearPredictionMSD-style real-data benchmark for SPT-Reg."
    )
    parser.add_argument("--summary-json-path", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--cache-path", type=str, default="")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--scale-split-factor",
        type=int,
        default=1,
        help="Multiply n_source / n_target_* pools and target_label_budget after base config.",
    )
    parser.add_argument("--openml-data-id", type=int, default=None)
    parser.add_argument("--openml-dataset-name", type=str, default="")
    parser.add_argument("--openml-version", type=int, default=SPTRegYearConfig.openml_version)
    parser.add_argument(
        "--max-dataset-rows",
        type=int,
        default=None,
        help="Subsample rows before covariate-shift split (large OpenML dumps).",
    )
    parser.add_argument(
        "--prior-ratio-clip",
        type=float,
        default=SPTRegYearConfig.prior_ratio_clip,
        help="Clip factor for Stage-A prior ratio stabilization.",
    )
    parser.add_argument(
        "--prior-transport-strength",
        type=float,
        default=SPTRegYearConfig.prior_transport_strength,
        help="Strength of prior transport update (0-1 style blend in transport).",
    )
    args = parser.parse_args()
    cfg = SPTRegYearConfig(
        dataset_path=args.dataset_path or None,
        cache_path=args.cache_path or None,
        allow_download=args.allow_download,
        openml_data_id=args.openml_data_id,
        openml_dataset_name=args.openml_dataset_name or None,
        openml_version=args.openml_version,
        max_dataset_rows=args.max_dataset_rows,
        prior_ratio_clip=float(args.prior_ratio_clip),
        prior_transport_strength=float(args.prior_transport_strength),
    )
    if args.scale_split_factor != 1:
        cfg = spt_year_scale_split_sizes(cfg, args.scale_split_factor)
    main(cfg, summary_json_path=args.summary_json_path)
