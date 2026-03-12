"""Download, normalize, and benchmark an NNC-style photo-z catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tools import photoz_benchmark_suite, photoz_collect_real_data
except ModuleNotFoundError:  # pragma: no cover - script execution path
    import photoz_benchmark_suite  # type: ignore[no-redef]
    import photoz_collect_real_data  # type: ignore[no-redef]

DEFAULT_RAW_DIR = Path("data/nnc_crps/catalogs")
DEFAULT_NORMALIZED_OUTPUT = Path("data/nnc_crps/nnc_photoz_real.csv")
DEFAULT_SUITE_REPORT = Path("reports/example_summaries/photoz_nnc_suite_latest.json")
DEFAULT_MARKDOWN_REPORT = Path("reports/example_summaries/photoz_nnc_suite_latest.md")
DEFAULT_REPORT = Path("reports/example_summaries/photoz_nnc_pipeline_latest.json")
MIN_ROWS_BY_PROFILE = {
    "smoke": 112,
    "audit": 320,
    "full": 1024,
}

BANDS = ("u", "g", "r", "i", "z", "y")
TARGET_CANDIDATES = (
    "spec_z",
    "z_spec",
    "specz_redshift",
    "redshift",
    "redshift_true",
    "true_redshift",
    "z_true",
    "target",
)
TARGET_ERR_CANDIDATES = (
    "spec_z_err",
    "z_spec_err",
    "specz_redshift_err",
    "redshift_err",
    "redshift_true_err",
    "z_true_err",
    "target_err",
)


def _find_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    folded = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in folded:
            return folded[candidate.lower()]
    return None


def _band_mag_candidates(band: str) -> tuple[str, ...]:
    if band == "z":
        return (
            "z_mag",
            "z",
            "mag_z",
            "zmag",
            "lsst_z",
            "z_lsst",
            "z_cmodel_mag",
        )
    return (
        band,
        f"{band}_mag",
        f"mag_{band}",
        f"{band}mag",
        f"lsst_{band}",
        f"{band}_lsst",
        f"{band}_cmodel_mag",
    )


def _band_err_candidates(band: str) -> tuple[str, ...]:
    base = "z_mag" if band == "z" else band
    return (
        f"{base}_err",
        f"{base}_error",
        f"{base}err",
        f"err_{base}",
        f"magerr_{band}",
        f"{band}_magerr",
        f"lsst_{band}_err",
        f"{band}_cmodel_magsigma",
    )


def _band_flux_candidates(band: str) -> tuple[str, ...]:
    return (
        f"flux_{band}",
        f"{band}_flux",
        f"{band}_cmodelflux",
        f"lsst_{band}_flux",
    )


def _band_flux_err_candidates(band: str) -> tuple[str, ...]:
    return (
        f"fluxerr_{band}",
        f"flux_err_{band}",
        f"{band}_flux_err",
        f"{band}_fluxerror",
        f"lsst_{band}_flux_err",
    )


def _load_raw_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=(suffix == ".jsonl"))
    if suffix == ".fits":
        try:
            from astropy.table import Table
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Reading NNC FITS catalogs requires `astropy`. "
                "Install it or supply a CSV/JSON/JSONL catalog."
            ) from exc
        return Table.read(path).to_pandas()
    raise ValueError(
        f"Unsupported raw NNC catalog format `{suffix}` for {path}. "
        "Use CSV, JSON, JSONL, or FITS."
    )


def _count_rows(path: Path) -> int | None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    if suffix in {".json", ".jsonl"}:
        return int(len(_load_raw_table(path)))
    return None


def _select_local_catalog(*, profile: str) -> Path | None:
    min_rows = MIN_ROWS_BY_PROFILE[profile]
    candidates = sorted(path for path in DEFAULT_RAW_DIR.iterdir() if path.is_file())
    valid: list[tuple[int, Path]] = []
    for path in candidates:
        rows = _count_rows(path)
        if rows is not None and rows >= min_rows:
            valid.append((rows, path))
    if not valid:
        return None
    valid.sort(key=lambda item: (item[0], item[1].name))
    return valid[-1][1]


def _flux_to_mag(flux: pd.Series) -> pd.Series:
    flux_f = pd.to_numeric(flux, errors="coerce").astype(float)
    mag = pd.Series(np.nan, index=flux_f.index, dtype=float)
    positive = flux_f > 0
    mag.loc[positive] = 22.5 - 2.5 * np.log10(flux_f.loc[positive])
    return mag


def _flux_err_to_mag_err(flux: pd.Series, flux_err: pd.Series) -> pd.Series:
    flux_f = pd.to_numeric(flux, errors="coerce").astype(float).clip(lower=1e-12)
    flux_err_f = pd.to_numeric(flux_err, errors="coerce").astype(float)
    return (2.5 / np.log(10.0)) * (flux_err_f.abs() / flux_f)


def normalize_nnc_catalog(
    *,
    raw_catalog_path: Path,
    output_path: Path,
    target_col: str | None = None,
    target_err_col: str | None = None,
    default_target_err: float = 0.01,
) -> dict[str, Any]:
    raw = _load_raw_table(raw_catalog_path)
    columns = list(raw.columns)

    resolved_target = target_col or _find_column(columns, TARGET_CANDIDATES)
    if resolved_target is None:
        raise ValueError(
            "Could not infer the target redshift column. " f"Available columns: {columns}"
        )
    resolved_target_err = target_err_col or _find_column(columns, TARGET_ERR_CANDIDATES)

    out = pd.DataFrame()
    object_id = _find_column(columns, ("objid", "object_id", "id", "row_id"))
    out["objid"] = raw[object_id] if object_id is not None else np.arange(len(raw))
    out["spec_z"] = pd.to_numeric(raw[resolved_target], errors="coerce")
    if resolved_target_err is None:
        out["spec_z_err"] = float(default_target_err)
    else:
        out["spec_z_err"] = pd.to_numeric(raw[resolved_target_err], errors="coerce").fillna(
            float(default_target_err)
        )

    used_bands: list[str] = []
    for band in BANDS:
        mag_col = _find_column(columns, _band_mag_candidates(band))
        err_col = _find_column(columns, _band_err_candidates(band))
        flux_col = _find_column(columns, _band_flux_candidates(band))
        flux_err_col = _find_column(columns, _band_flux_err_candidates(band))
        canonical_band = "z_mag" if band == "z" else band
        if mag_col is not None:
            out[canonical_band] = pd.to_numeric(raw[mag_col], errors="coerce")
            if err_col is not None:
                out[f"{canonical_band}_err"] = pd.to_numeric(raw[err_col], errors="coerce")
            else:
                out[f"{canonical_band}_err"] = 0.03
            used_bands.append(band)
            continue
        if flux_col is not None:
            out[canonical_band] = _flux_to_mag(raw[flux_col])
            if flux_err_col is not None:
                out[f"{canonical_band}_err"] = _flux_err_to_mag_err(
                    raw[flux_col],
                    raw[flux_err_col],
                )
            else:
                out[f"{canonical_band}_err"] = 0.03
            used_bands.append(band)

    if len(used_bands) < 3:
        raise ValueError(
            "Need at least three photometric bands to build color features. "
            f"Inferred bands: {used_bands}"
        )

    feature_columns: list[str] = []
    for left, right in zip(used_bands[:-1], used_bands[1:]):
        left_name = "z_mag" if left == "z" else left
        right_name = "z_mag" if right == "z" else right
        feature_name = f"{left}_{right}"
        out[feature_name] = out[left_name] - out[right_name]
        out[f"{feature_name}_err"] = np.sqrt(
            pd.to_numeric(out[f"{left_name}_err"], errors="coerce") ** 2
            + pd.to_numeric(out[f"{right_name}_err"], errors="coerce") ** 2
        )
        feature_columns.append(feature_name)

    required = ["spec_z", "spec_z_err", *feature_columns]
    out = out.dropna(subset=required).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    return {
        "artifact": "photoz_nnc_normalization_report",
        "version": 1,
        "raw_catalog_path": str(raw_catalog_path),
        "normalized_output_path": str(output_path),
        "rows_raw": int(len(raw)),
        "rows_normalized": int(len(out)),
        "target_col": resolved_target,
        "target_err_col": resolved_target_err,
        "used_bands": used_bands,
        "feature_columns": feature_columns,
    }


def run_pipeline(
    *,
    profile: str,
    output_dir: Path,
    raw_catalog_path: Path | None,
    normalized_output_path: Path,
    download_if_missing: bool,
    record_id: int,
    preferred_suffix: str,
    suite_report_path: Path,
    markdown_report_path: Path,
    target_col: str | None = None,
    target_err_col: str | None = None,
    default_target_err: float = 0.01,
) -> dict[str, Any]:
    raw_path = raw_catalog_path
    download_report: dict[str, Any] | None = None
    if raw_path is None:
        DEFAULT_RAW_DIR.mkdir(parents=True, exist_ok=True)
        raw_path = _select_local_catalog(profile=profile)
        if raw_path is None and download_if_missing:
            try:
                download_report = photoz_collect_real_data.collect_nnc_catalog(
                    record_id=record_id,
                    preferred_suffix=preferred_suffix,
                    output_dir=DEFAULT_RAW_DIR,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Unable to acquire a usable NNC catalog. "
                    "No local catalog met the minimum row requirement and the "
                    "download step failed. "
                    "Provide --raw-catalog explicitly or override --record-id/--preferred-suffix."
                ) from exc
            raw_path = Path(download_report["output_path"])
        if raw_path is None:
            min_rows = MIN_ROWS_BY_PROFILE[profile]
            raise FileNotFoundError(
                "No usable raw NNC catalog available. "
                f"Need at least {min_rows} rows for profile `{profile}`. "
                "Supply --raw-catalog or enable --download-if-missing."
            )

    assert raw_path is not None
    normalization_report = normalize_nnc_catalog(
        raw_catalog_path=raw_path,
        output_path=normalized_output_path,
        target_col=target_col,
        target_err_col=target_err_col,
        default_target_err=default_target_err,
    )
    suite_report = photoz_benchmark_suite.run_suite(
        profile=profile,
        output_dir=output_dir,
        real_data_only=True,
        dataset_path=normalized_output_path,
        markdown_report_path=markdown_report_path,
    )
    suite_report_path.parent.mkdir(parents=True, exist_ok=True)
    suite_report_path.write_text(json.dumps(suite_report, indent=2), encoding="utf-8")

    return {
        "artifact": "photoz_nnc_pipeline_report",
        "version": 1,
        "profile": profile,
        "download_report": download_report,
        "normalization_report": normalization_report,
        "suite_report_path": str(suite_report_path),
        "suite_markdown_report_path": str(markdown_report_path),
        "summary_paths": suite_report["summary_paths"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download, normalize, and benchmark an NNC-style photo-z catalog."
    )
    parser.add_argument("--profile", choices=["smoke", "audit", "full"], default="full")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/example_summaries"))
    parser.add_argument("--raw-catalog", type=Path, default=None)
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument(
        "--record-id",
        type=int,
        default=photoz_collect_real_data.DEFAULT_NNC_ZENODO_RECORD,
    )
    parser.add_argument("--preferred-suffix", type=str, default=".csv")
    parser.add_argument("--normalized-output", type=Path, default=DEFAULT_NORMALIZED_OUTPUT)
    parser.add_argument("--suite-report", type=Path, default=DEFAULT_SUITE_REPORT)
    parser.add_argument("--markdown-report", type=Path, default=DEFAULT_MARKDOWN_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--target-col", type=str, default=None)
    parser.add_argument("--target-err-col", type=str, default=None)
    parser.add_argument("--default-target-err", type=float, default=0.01)
    args = parser.parse_args()

    report = run_pipeline(
        profile=args.profile,
        output_dir=args.output_dir,
        raw_catalog_path=args.raw_catalog,
        normalized_output_path=args.normalized_output,
        download_if_missing=args.download_if_missing,
        record_id=args.record_id,
        preferred_suffix=args.preferred_suffix,
        suite_report_path=args.suite_report,
        markdown_report_path=args.markdown_report,
        target_col=args.target_col,
        target_err_col=args.target_err_col,
        default_target_err=args.default_target_err,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote NNC pipeline report: {args.report}")
    print(f"Normalized dataset: {report['normalization_report']['normalized_output_path']}")
    print(f"Suite report: {report['suite_report_path']}")


if __name__ == "__main__":
    main()
