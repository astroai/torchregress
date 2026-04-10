"""YearPredictionMSD-style real-data benchmark for Shift-Factored Predictive Transport."""

import argparse
from dataclasses import dataclass
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
from sklearn.datasets import fetch_openml

from torchregress.test_time import (
    FeatureStatNormalizer,
    ShiftFactoredPredictiveTransport,
    ShiftFactoredTransportConfig,
    SignificantSubspaceAligner,
)


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


def _load_dataset_frame(cfg: SPTRegYearConfig) -> tuple[pd.DataFrame, str]:
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

    if not cfg.allow_download:
        raise FileNotFoundError("dataset_path/cache_path missing and allow_download is False")

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

    need = cfg.n_source + cfg.n_target_unlabeled + cfg.n_target_cal + cfg.n_target_test
    if need > x.shape[0]:
        raise ValueError(f"Requested {need} rows but dataset has {x.shape[0]}.")
    if not 0 <= cfg.shift_feature_idx < x.shape[1]:
        raise ValueError("shift_feature_idx is out of bounds for the dataset features")

    rng = np.random.default_rng(cfg.seed)
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
        ShiftFactoredTransportConfig(
            n_support=cfg.n_support,
            alpha=cfg.alpha,
            enable_alignment=False,
            enable_uncertainty_inflation=False,
            random_state=cfg.seed,
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

    spt = ShiftFactoredPredictiveTransport(
        ShiftFactoredTransportConfig(
            n_support=cfg.n_support,
            alpha=cfg.alpha,
            random_state=cfg.seed,
        )
    ).fit_source(
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

    spt_binned = ShiftFactoredPredictiveTransport(
        ShiftFactoredTransportConfig(
            n_support=cfg.n_support,
            alpha=cfg.alpha,
            random_state=cfg.seed,
        )
    ).fit_source(
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
    args = parser.parse_args()
    main(
        SPTRegYearConfig(
            dataset_path=args.dataset_path or None,
            cache_path=args.cache_path or None,
            allow_download=args.allow_download,
        ),
        summary_json_path=args.summary_json_path,
    )
