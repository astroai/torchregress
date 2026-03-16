"""
CLAUDS + COSMOS Spec-z catalog: supervised and semi-supervised photo-z benchmark.

Loads the merged catalog (with spec_z from crossmatch) via Polars, builds color
features from magnitudes, then runs supervised and semi-supervised methods at
multiple labeled fractions. Use pre-built train/cal/test parquets or the full
catalog path (with sampling).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import photoz_transferz_semisupervised_comparison as pzsemi
from comparison_utils import (
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
    write_comparison_summary_json,
)

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

LabelPolicy = Literal["random", "highz_scarce"]
SplitPolicy = Literal["random", "stratified_redshift", "stratified_magnitude"]
FeatureSetPolicy = Literal["optical_colors", "all_bands_missing_ok"]

# Default catalog path (relative to repo root when running from examples/)
DEFAULT_CATALOG_PATH = Path("data/clauds_specz/clauds_specz_catalog.parquet")
MAG_COLS = ["u", "g", "r", "i", "z", "y"]
# All bands (optical + NIR) for "all bands with missing data" / outer-join scenario (paper SED + NIR)
ALL_BANDS = ["u", "g", "r", "i", "z", "y", "Y", "J", "H", "Ks"]
COLOR_PAIRS = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"), ("z", "y")]
DEFAULT_STRATIFY_BINS = 10
# Minimum optical bands required per row when using all_bands_missing_ok (so we have some signal)
MIN_OPTICAL_BANDS_ALL_BANDS_MISSING = 3


def _build_colors_from_magnitudes(df: pd.DataFrame) -> pd.DataFrame:
    """Add color columns and errors from magnitude columns. Drops rows with NaN in required bands."""
    out = df.copy()
    for a, b in COLOR_PAIRS:
        if a not in out.columns or b not in out.columns:
            continue
        col = f"{a}_{b}"
        out[col] = out[a] - out[b]
        a_err = f"{a}_err" if f"{a}_err" in out.columns else None
        b_err = f"{b}_err" if f"{b}_err" in out.columns else None
        if a_err and b_err:
            out[f"{col}_err"] = np.sqrt(
                out[a_err].astype(float) ** 2 + out[b_err].astype(float) ** 2
            )
    return out


def build_all_bands_with_mask_df(
    df: pd.DataFrame,
    all_bands: list[str] | None = None,
    min_optical_bands: int = MIN_OPTICAL_BANDS_ALL_BANDS_MISSING,
    optical_bands: list[str] | None = None,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """
    Build feature matrix with all bands (mags + errs) and per-band observed mask for missing-data training.

    Uses outer-join semantics: keep rows with at least min_optical_bands optical bands; fill missing
    mag/err with fill_value; set obs_* = 1 where observed, 0 where missing. The model can use
    obs_* to propagate missingness into wider redshift intervals.

    Returns DataFrame with columns: u..Ks, u_err..Ks_err, obs_u..obs_Ks, obs_u_err..obs_Ks_err (zeros),
    spec_z, spec_z_err. Rows with fewer than min_optical_bands optical (u,g,r,i,z,y) are dropped.
    """
    bands = all_bands or ALL_BANDS
    optical = optical_bands or MAG_COLS
    optical_in_df = [b for b in optical if b in df.columns]
    if len(optical_in_df) < min_optical_bands:
        raise ValueError(
            f"Need at least {min_optical_bands} optical bands in DataFrame; found {optical_in_df}"
        )
    # Count observed optical bands per row (coerce to float for safety)
    opt_df = df[optical_in_df].apply(pd.to_numeric, errors="coerce")
    observed_optical = opt_df.notna() & np.isfinite(opt_df)
    n_optical_obs = observed_optical.sum(axis=1)
    keep = n_optical_obs >= min_optical_bands
    df = df.loc[keep].copy().reset_index(drop=True)
    n_rows = len(df)
    out: dict[str, np.ndarray] = {}
    for b in bands:
        if b not in df.columns:
            out[b] = np.full(n_rows, fill_value, dtype=np.float32)
            out[f"{b}_err"] = np.full(n_rows, fill_value, dtype=np.float32)
            out[f"obs_{b}"] = np.zeros(n_rows, dtype=np.float32)
            out[f"obs_{b}_err"] = np.zeros(n_rows, dtype=np.float32)
            continue
        vals = pd.to_numeric(df[b], errors="coerce").values
        obs = np.isfinite(vals) & ~np.isnan(vals)
        out[b] = np.where(obs, vals, fill_value).astype(np.float32)
        err_col = f"{b}_err"
        if err_col in df.columns:
            errs = pd.to_numeric(df[err_col], errors="coerce").values
            safe = np.where(np.isfinite(errs), np.clip(errs, 0.0, 1e6), fill_value)
            out[err_col] = np.where(obs, safe, fill_value).astype(np.float32)
        else:
            out[err_col] = np.full(n_rows, fill_value, dtype=np.float32)
        out[f"obs_{b}"] = obs.astype(np.float32)
        out[f"obs_{b}_err"] = np.zeros(n_rows, dtype=np.float32)
    # Placeholder errors for the mag_err features (benchmark expects len(feature_cols)==len(error_cols))
    for b in bands:
        out[f"{b}_err_err"] = np.zeros(n_rows, dtype=np.float32)
    out["spec_z"] = (
        pd.to_numeric(df["spec_z"], errors="coerce").values
        if "spec_z" in df.columns
        else np.full(n_rows, np.nan, dtype=np.float32)
    )
    out["spec_z_err"] = (
        pd.to_numeric(df["spec_z_err"], errors="coerce").fillna(0.001).values
        if "spec_z_err" in df.columns
        else np.full(n_rows, 0.001, dtype=np.float32)
    )
    # Catalog photo-z (for same-sample baseline comparison)
    if "z_phot" in df.columns:
        out["z_phot"] = (
            pd.to_numeric(df["z_phot"], errors="coerce").fillna(0.0).values.astype(np.float32)
        )
    else:
        out["z_phot"] = np.full(n_rows, np.nan, dtype=np.float32)
    if "z_phot_err" in df.columns:
        out["z_phot_err"] = (
            pd.to_numeric(df["z_phot_err"], errors="coerce").fillna(0.01).values.astype(np.float32)
        )
    else:
        out["z_phot_err"] = np.full(n_rows, 0.01, dtype=np.float32)
    # Column order: mags, errs, obs, obs_err, mag_err_err (placeholders), spec_z, spec_z_err, z_phot, z_phot_err
    col_order = (
        list(bands)
        + [f"{b}_err" for b in bands]
        + [f"obs_{b}" for b in bands]
        + [f"obs_{b}_err" for b in bands]
        + [f"{b}_err_err" for b in bands]
        + ["spec_z", "spec_z_err", "z_phot", "z_phot_err"]
    )
    return pd.DataFrame({k: out[k] for k in col_order})


def _stratified_split_indices(
    n_total: int,
    n_train: int,
    n_cal: int,
    n_test: int,
    stratify_values: np.ndarray,
    seed: int,
    n_bins: int = DEFAULT_STRATIFY_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/cal/test indices so each split has similar distribution of stratify_values."""
    need = n_train + n_cal + n_test
    if n_total < need:
        raise ValueError(f"n_total {n_total} < need {need}")
    rng = np.random.default_rng(seed)
    try:
        bins = np.percentile(stratify_values, np.linspace(0, 100, n_bins + 1)[1:-1])
        bin_id = np.searchsorted(bins, stratify_values, side="right")
    except Exception:
        bin_id = np.zeros(n_total, dtype=np.intp)
    train_idx: list[int] = []
    cal_idx: list[int] = []
    test_idx: list[int] = []
    for b in np.unique(bin_id):
        mask = np.where(bin_id == b)[0]
        rng.shuffle(mask)
        n_b = len(mask)
        # Assign proportionally to train/cal/test
        n_train_b = max(0, min(n_train - len(train_idx), int(round(n_b * n_train / need))))
        n_cal_b = max(0, min(n_cal - len(cal_idx), int(round(n_b * n_cal / need))))
        n_test_b = max(0, min(n_test - len(test_idx), n_b - n_train_b - n_cal_b))
        if n_test_b < 0:
            n_test_b = 0
            n_cal_b = min(n_cal - len(cal_idx), n_b - n_train_b)
        if n_cal_b < 0:
            n_cal_b = 0
        train_idx.extend(mask[:n_train_b].tolist())
        cal_idx.extend(mask[n_train_b : n_train_b + n_cal_b].tolist())
        test_idx.extend(mask[n_train_b + n_cal_b : n_train_b + n_cal_b + n_test_b].tolist())
    # Top up to exact sizes from remaining indices
    used = set(train_idx) | set(cal_idx) | set(test_idx)
    remaining = np.array([i for i in range(n_total) if i not in used])
    rng.shuffle(remaining)
    k = 0
    while len(train_idx) < n_train and k < len(remaining):
        train_idx.append(int(remaining[k]))
        k += 1
    while len(cal_idx) < n_cal and k < len(remaining):
        cal_idx.append(int(remaining[k]))
        k += 1
    while len(test_idx) < n_test and k < len(remaining):
        test_idx.append(int(remaining[k]))
        k += 1
    return np.array(train_idx[:n_train]), np.array(cal_idx[:n_cal]), np.array(test_idx[:n_test])


def load_clauds_specz_splits(
    catalog_path: Path,
    n_train: int,
    n_cal: int,
    n_test: int,
    seed: int = 240226,
    repo_root: Path | None = None,
    split_policy: SplitPolicy = "random",
    stratify_n_bins: int = DEFAULT_STRATIFY_BINS,
    report_counts: bool = False,
    feature_set: FeatureSetPolicy = "optical_colors",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load CLAUDS+specz Parquet with Polars (filter spec_z not null, sample), build features, return train/cal/test DataFrames.

    feature_set:
      - optical_colors: build u-g, g-r, ... colors (drop rows with NaN in required bands).
      - all_bands_missing_ok: use all bands (u..Ks + errs + obs_* mask); keep rows with at least
        MIN_OPTICAL_BANDS_ALL_BANDS_MISSING optical bands; missing bands filled with 0, obs_* = 1/0.
        Expect wider redshift intervals when fewer bands observed (if model uses mask).
    split_policy: random, stratified_redshift (by spec_z), or stratified_magnitude (by i or r band).
    """
    if pl is None:
        raise ImportError(
            "Polars is required for loading CLAUDS catalog. Install with: pip install polars"
        )
    catalog_path = Path(catalog_path)
    if not catalog_path.is_absolute() and repo_root is not None:
        catalog_path = repo_root / catalog_path
    if not catalog_path.exists():
        raise FileNotFoundError(f"CLAUDS+specz catalog not found: {catalog_path}")

    need = n_train + n_cal + n_test
    lf = pl.scan_parquet(catalog_path)
    df_with_specz = lf.filter(pl.col("spec_z").is_not_null()).collect()
    n_with_specz = df_with_specz.height
    if report_counts:
        try:
            by_field = df_with_specz.group_by("field").len()
            field_counts = {r["field"]: r["len"] for r in by_field.to_dicts()}
        except Exception:
            field_counts = {}
        print(
            f"Objects with spec_z: {n_with_specz}. Per field: {field_counts}. "
            f"Split: train={n_train}, cal={n_cal}, test={n_test} (policy={split_policy}, feature_set={feature_set})."
        )
    if n_with_specz < need:
        raise ValueError(
            f"Catalog has {n_with_specz} rows with spec_z but need {need}. "
            "Use a larger catalog or smaller n_train/n_cal/n_test."
        )
    df = df_with_specz.to_pandas()

    if feature_set == "all_bands_missing_ok":
        df = build_all_bands_with_mask_df(df, min_optical_bands=MIN_OPTICAL_BANDS_ALL_BANDS_MISSING)
        n_total = len(df)
        if n_total < need:
            raise ValueError(
                f"After all-bands+mask filter: {n_total} rows, need {need}. "
                "Use a larger catalog or smaller n_train/n_cal/n_test."
            )
    else:
        df = _build_colors_from_magnitudes(df)
        required = [f"{a}_{b}" for a, b in COLOR_PAIRS] + [f"{a}_{b}_err" for a, b in COLOR_PAIRS]
        required = [c for c in required if c in df.columns]
        if len(required) < 6:
            raise ValueError(f"Need at least 3 color columns with errors. Found: {required}")
        df = df.dropna(subset=required + ["spec_z"]).reset_index(drop=True)
        if "spec_z_err" not in df.columns or df["spec_z_err"].isna().all():
            df["spec_z_err"] = 0.001
        df["spec_z_err"] = df["spec_z_err"].fillna(0.001)
        n_total = len(df)
        if n_total < need:
            raise ValueError(
                f"After color/dropna: {n_total} rows, need {need}. "
                "Use a larger catalog or smaller n_train/n_cal/n_test."
            )

    if split_policy == "random":
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n_total)[:need]
        df = df.iloc[idx].reset_index(drop=True)
        train_df = df.iloc[:n_train]
        cal_df = df.iloc[n_train : n_train + n_cal]
        test_df = df.iloc[n_train + n_cal :]
    else:
        if split_policy == "stratified_redshift":
            stratify_col = np.asarray(df["spec_z"].values, dtype=float)
        else:
            mag_col = "i" if "i" in df.columns else "r" if "r" in df.columns else "g"
            stratify_col = np.asarray(df[mag_col].values, dtype=float)
        ti, ci, tei = _stratified_split_indices(
            n_total, n_train, n_cal, n_test, stratify_col, seed, n_bins=stratify_n_bins
        )
        train_df = df.iloc[ti].reset_index(drop=True)
        cal_df = df.iloc[ci].reset_index(drop=True)
        test_df = df.iloc[tei].reset_index(drop=True)
    return train_df, cal_df, test_df


def run_comparison(
    catalog_path: Path | None = None,
    train_dataset_path: str | None = None,
    cal_dataset_path: str | None = None,
    test_dataset_path: str | None = None,
    n_train: int = 512,
    n_cal: int = 256,
    n_test: int = 256,
    seed: int = 240226,
    labeled_fractions: tuple[float, ...] = (0.1, 0.25, 0.5),
    label_policy: LabelPolicy = "highz_scarce",
    split_policy: SplitPolicy = "stratified_redshift",
    stratify_n_bins: int = DEFAULT_STRATIFY_BINS,
    report_counts: bool = True,
    repo_root: Path | None = None,
    feature_set: FeatureSetPolicy = "optical_colors",
    batch_size: int = 64,
    epochs: int = 10,
    teacher_epochs: int = 12,
) -> tuple[list[dict], list[str]]:
    """Run supervised and semi-supervised comparison on CLAUDS+specz splits."""
    set_comparison_seed(seed)
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]

    if catalog_path is not None and (
        train_dataset_path is None or cal_dataset_path is None or test_dataset_path is None
    ):
        train_df, cal_df, test_df = load_clauds_specz_splits(
            catalog_path,
            n_train,
            n_cal,
            n_test,
            seed=seed,
            repo_root=repo_root,
            split_policy=split_policy,
            stratify_n_bins=stratify_n_bins,
            report_counts=report_counts,
            feature_set=feature_set,
        )
        split_dir = repo_root / "data" / "clauds_specz" / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        train_path = split_dir / "clauds_specz_train.parquet"
        cal_path = split_dir / "clauds_specz_cal.parquet"
        test_path = split_dir / "clauds_specz_test.parquet"
        train_df.to_parquet(train_path, index=False)
        cal_df.to_parquet(cal_path, index=False)
        test_df.to_parquet(test_path, index=False)
        train_dataset_path = str(train_path)
        cal_dataset_path = str(cal_path)
        test_dataset_path = str(test_path)
    elif train_dataset_path and cal_dataset_path and test_dataset_path:
        pass
    else:
        raise ValueError("Provide either catalog_path or all of train/cal/test_dataset_path.")

    cfg_semi = pzsemi.PhotoZTransferZSemiSupervisedConfig(
        seed=seed,
        n_train=n_train,
        n_cal=n_cal,
        n_test=n_test,
        batch_size=batch_size,
        epochs=epochs,
        teacher_epochs=teacher_epochs,
        labeled_fractions=labeled_fractions,
        label_policy=label_policy,
        train_dataset_path=train_dataset_path,
        cal_dataset_path=cal_dataset_path,
        test_dataset_path=test_dataset_path,
        require_real_data=True,
    )
    rows, notes = pzsemi.run_comparison(cfg_semi)
    return rows, notes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLAUDS + Spec-z: supervised and semi-supervised photo-z comparison."
    )
    parser.add_argument(
        "--catalog-path",
        type=Path,
        default=None,
        help="CLAUDS+specz Parquet (optional if splits set).",
    )
    parser.add_argument("--train-dataset-path", type=str, default=None)
    parser.add_argument("--cal-dataset-path", type=str, default=None)
    parser.add_argument("--test-dataset-path", type=str, default=None)
    parser.add_argument("--n-train", type=int, default=512)
    parser.add_argument("--n-cal", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=256)
    parser.add_argument("--seed", type=int, default=240226)
    parser.add_argument("--labeled-fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument(
        "--label-policy", type=str, choices=["random", "highz_scarce"], default="highz_scarce"
    )
    parser.add_argument(
        "--split-policy",
        type=str,
        choices=["random", "stratified_redshift", "stratified_magnitude"],
        default="stratified_redshift",
        help="How to split train/cal/test from catalog.",
    )
    parser.add_argument(
        "--stratify-n-bins",
        type=int,
        default=DEFAULT_STRATIFY_BINS,
        help="Bins for stratified split.",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        choices=["optical_colors", "all_bands_missing_ok"],
        default="optical_colors",
        help="optical_colors: u-g, g-r, ... (drop missing). all_bands_missing_ok: all bands + obs mask (wider intervals when fewer bands).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training (default 64)."
    )
    parser.add_argument("--epochs", type=int, default=10, help="Student epochs (default 10).")
    parser.add_argument(
        "--teacher-epochs",
        type=int,
        default=12,
        help="Teacher epochs for pseudo-label methods (default 12).",
    )
    parser.add_argument(
        "--no-report-counts", action="store_true", help="Do not print object counts."
    )
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    rows, notes = run_comparison(
        catalog_path=args.catalog_path,
        train_dataset_path=args.train_dataset_path,
        cal_dataset_path=args.cal_dataset_path,
        test_dataset_path=args.test_dataset_path,
        n_train=args.n_train,
        n_cal=args.n_cal,
        n_test=args.n_test,
        seed=args.seed,
        labeled_fractions=tuple(args.labeled_fractions),
        label_policy=args.label_policy,
        split_policy=args.split_policy,
        stratify_n_bins=args.stratify_n_bins,
        report_counts=not args.no_report_counts,
        repo_root=repo_root,
        feature_set=args.feature_set,
        batch_size=args.batch_size,
        epochs=args.epochs,
        teacher_epochs=args.teacher_epochs,
    )

    print_fairness_notes(
        title="CLAUDS + Spec-z Photo-z Comparison",
        seed_policy="fixed seed; train/cal/test from catalog or explicit paths",
        train_budget="Same as TransferZ semi-supervised (teacher + student epochs)",
        metric_policy="NMAD, catastrophic rate, coverage; multiple labeled fractions.",
    )
    print_comparison_summary(
        "CLAUDS + Spec-z Semi-Supervised Summary",
        rows,
        metric_order=["LabeledFraction", "NMAD", "CatastrophicRate", "HighZ_MAE", "train_s"],
    )
    if args.summary_json_path:
        write_comparison_summary_json(
            args.summary_json_path,
            example="examples/photoz_clauds_specz_comparison.py",
            task="CLAUDS+Spec-z semi-supervised photometric redshift comparison",
            config={
                "n_train": args.n_train,
                "n_cal": args.n_cal,
                "n_test": args.n_test,
                "seed": args.seed,
                "labeled_fractions": args.labeled_fractions,
                "label_policy": args.label_policy,
                "split_policy": args.split_policy,
                "stratify_n_bins": args.stratify_n_bins,
                "feature_set": args.feature_set,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "teacher_epochs": args.teacher_epochs,
            },
            rows=rows,
            notes=notes,
        )
        print(f"\nWrote summary JSON: {args.summary_json_path}")


if __name__ == "__main__":
    main()
