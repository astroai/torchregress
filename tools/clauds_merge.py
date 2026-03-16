"""Merge CLAUDS HSCpipe FITS into a single Parquet using Polars (astropy → PyArrow → Polars, no pandas)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pyarrow as pa

if TYPE_CHECKING:
    from astropy.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "clauds_specz" / "raw"
DEFAULT_MERGED_DIR = REPO_ROOT / "data" / "clauds_specz"
MERGED_FILENAME = "clauds_merged_photometry.parquet"

# Field label and FITS filename (without path) for HSCpipe-only merge.
HSCPIPE_FIELD_FILES = [
    ("E-COSMOS", "COSMOS-HSCpipe-Phosphoros.fits"),
    ("DEEP2-3", "DEEP23-HSCpipe-Phosphoros.fits"),
    ("XMM-LSS", "XMMLSS-HSCpipe-Phosphoros.fits"),
    ("ELAIS-N1", "ELAIS-N1-HSCpipe-Phosphoros.fits"),
]

# Canonical schema: we want these columns in the output. NIR (Y,J,H,Ks) absent in 6-band fields.
CANONICAL_MAG_COLS = ["u", "g", "r", "i", "z", "y", "Y", "J", "H", "Ks"]
CANONICAL_OTHER = ["ra", "dec", "z_phot", "z_phot_err"]

# HSCpipe Phosphoros FITS: flux columns (priority order APER_2, APER_3, CMODEL).
# (canonical_band, flux_col, fluxerr_col); first match present in table is used.
FLUX_BAND_COLUMNS: list[tuple[str, str, str]] = [
    ("g", "FLUX_APER_2_HSC-G", "FLUXERR_APER_2_HSC-G"),
    ("g", "FLUX_APER_3_HSC-G", "FLUXERR_APER_3_HSC-G"),
    ("g", "FLUX_CMODEL_HSC-G", "FLUXERR_CMODEL_HSC-G"),
    ("r", "FLUX_APER_2_HSC-R", "FLUXERR_APER_2_HSC-R"),
    ("r", "FLUX_APER_3_HSC-R", "FLUXERR_APER_3_HSC-R"),
    ("r", "FLUX_CMODEL_HSC-R", "FLUXERR_CMODEL_HSC-R"),
    ("i", "FLUX_APER_2_HSC-I", "FLUXERR_APER_2_HSC-I"),
    ("i", "FLUX_APER_3_HSC-I", "FLUXERR_APER_3_HSC-I"),
    ("i", "FLUX_CMODEL_HSC-I", "FLUXERR_CMODEL_HSC-I"),
    ("z", "FLUX_APER_2_HSC-Z", "FLUXERR_APER_2_HSC-Z"),
    ("z", "FLUX_APER_3_HSC-Z", "FLUXERR_APER_3_HSC-Z"),
    ("z", "FLUX_CMODEL_HSC-Z", "FLUXERR_CMODEL_HSC-Z"),
    ("y", "FLUX_APER_2_HSC-Y", "FLUXERR_APER_2_HSC-Y"),
    ("y", "FLUX_APER_3_HSC-Y", "FLUXERR_APER_3_HSC-Y"),
    ("y", "FLUX_CMODEL_HSC-Y", "FLUXERR_CMODEL_HSC-Y"),
    ("u", "FLUX_APER_2_MegaCam-u", "FLUXERR_APER_2_MegaCam-u"),
    ("u", "FLUX_APER_3_MegaCam-u", "FLUXERR_APER_3_MegaCam-u"),
    ("u", "FLUX_CMODEL_MegaCam-u", "FLUXERR_CMODEL_MegaCam-u"),
    ("u", "FLUX_APER_2_MegaCam-uS", "FLUXERR_APER_2_MegaCam-uS"),
    ("Y", "FLUX_APER_2_VIRCAM-Y", "FLUXERR_APER_2_VIRCAM-Y"),
    ("Y", "FLUX_APER_3_VIRCAM-Y", "FLUXERR_APER_3_VIRCAM-Y"),
    ("Y", "FLUX_CMODEL_VIRCAM-Y", "FLUXERR_CMODEL_VIRCAM-Y"),
    ("J", "FLUX_APER_2_VIRCAM-J", "FLUXERR_APER_2_VIRCAM-J"),
    ("J", "FLUX_APER_3_VIRCAM-J", "FLUXERR_APER_3_VIRCAM-J"),
    ("J", "FLUX_CMODEL_VIRCAM-J", "FLUXERR_CMODEL_VIRCAM-J"),
    ("H", "FLUX_APER_2_VIRCAM-H", "FLUXERR_APER_2_VIRCAM-H"),
    ("H", "FLUX_APER_3_VIRCAM-H", "FLUXERR_APER_3_VIRCAM-H"),
    ("H", "FLUX_CMODEL_VIRCAM-H", "FLUXERR_CMODEL_VIRCAM-H"),
    ("Ks", "FLUX_APER_2_VIRCAM-Ks", "FLUXERR_APER_2_VIRCAM-Ks"),
    ("Ks", "FLUX_APER_3_VIRCAM-Ks", "FLUXERR_APER_3_VIRCAM-Ks"),
    ("Ks", "FLUX_CMODEL_VIRCAM-Ks", "FLUXERR_CMODEL_VIRCAM-Ks"),
]

# Possible FITS column names per canonical (first match wins, case-insensitive).
COLUMN_CANDIDATES: dict[str, list[str]] = {
    "ra": ["ra", "coord_ra", "RA", "ALPHA_J2000", "alpha_j2000"],
    "dec": ["dec", "coord_dec", "DEC", "DELTA_J2000", "delta_j2000"],
    "z_phot": ["z_phot", "z_phot_median", "PHZ", "phot_z", "z_best", "ZPHOT"],
    "z_phot_err": ["z_phot_err", "z_phot_err68", "phot_z_err", "z_err", "Z_CHI"],
}
for band in ["u", "g", "r", "i", "z", "y", "Y", "J", "H", "Ks"]:
    COLUMN_CANDIDATES[band] = [
        band,
        f"{band}_mag",
        f"mag_{band}",
        f"MAG_APER_{band.upper()}",
        f"cmodel_mag_{band}",
        f"{band}_cmodel_mag",
        f"mag_aper_2_{band}",
    ]
    COLUMN_CANDIDATES[f"{band}_err"] = [
        f"{band}_err",
        f"err_{band}",
        f"{band}_magerr",
        f"MAGERR_APER_{band.upper()}",
        f"cmodel_magsigma_{band}",
        f"{band}_cmodel_magsigma",
    ]


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _fits_to_astropy_table(path: Path) -> Table:
    """Load FITS table as astropy Table (1-d columns only). Handles truncated files and big-endian."""
    import numpy as np
    from astropy.io import fits
    from astropy.table import Table

    path = Path(path)
    try:
        tbl = Table.read(path)
    except (ValueError, OSError) as e:
        if "reshape" not in str(e) and "too small" not in str(e).lower():
            raise
        tbl = None
    if tbl is not None:
        names = [n for n in tbl.colnames if len(tbl[n].shape) <= 1]
        return tbl[names]
    # Truncated file: read only complete rows (raw bytes + header dtype)
    with fits.open(path) as hdul:
        hdu = hdul[1]
        naxis1 = int(hdu.header["NAXIS1"])
        data_start = hdu._data_offset
        dtype = hdu.columns.dtype
    file_dtype = dtype.newbyteorder(">") if dtype.byteorder != ">" else dtype
    file_size = path.stat().st_size
    available = file_size - data_start
    n_rows_safe = available // naxis1
    if n_rows_safe <= 0:
        raise ValueError(f"FITS {path} is truncated; no complete rows (need {naxis1} bytes/row).")
    with open(path, "rb") as f:
        f.seek(data_start)
        raw = f.read(n_rows_safe * naxis1)
    arr = np.frombuffer(raw, dtype=file_dtype, count=n_rows_safe)
    if file_dtype.byteorder == ">":
        arr = arr.byteswap().view(dtype)
    tbl = Table(arr)
    names = [n for n in tbl.colnames if len(tbl[n].shape) <= 1]
    return tbl[names]


def _astropy_table_to_pyarrow(tbl: Table) -> pa.Table:
    """Convert astropy Table (1-d columns only) to PyArrow Table (no pandas)."""
    import numpy as np

    data: dict[str, pa.Array] = {}
    for col in tbl.colnames:
        c = tbl[col]
        d = c.data if hasattr(c, "data") else np.asarray(c)
        # PyArrow rejects byte-swapped arrays; ensure native byte order
        if getattr(d.dtype, "byteorder", "") not in ("=", "|", "<", ">"):
            pass
        elif d.dtype.byteorder == ">":
            d = np.asarray(d, dtype=d.dtype.newbyteorder("="))
        mask = getattr(c, "mask", None)
        if mask is not None and np.any(mask):
            d = np.ma.masked_array(d, mask=mask)
        data[col] = pa.array(d)
    return pa.table(data)


def _build_mapping(columns: list[str]) -> dict[str, str]:
    """Build FITS column name -> canonical name mapping."""
    mapping: dict[str, str] = {}
    for canonical, candidates in COLUMN_CANDIDATES.items():
        found = _find_column(columns, candidates)
        if found and found not in mapping:
            mapping[found] = canonical
    return mapping


def _add_mags_from_fluxes(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute magnitudes from FLUX_* / FLUXERR_* when present (HSCpipe Phosphoros).
    mag = -2.5 * log10(flux), mag_err = 1.0857 * fluxerr/flux; invalid/zero flux -> null.
    """
    colset = set(df.columns)
    # Per canonical band, use first (flux_col, fluxerr_col) that exists
    used: set[str] = set()
    for band, flux_col, fluxerr_col in FLUX_BAND_COLUMNS:
        if band in used or flux_col not in colset or fluxerr_col not in colset:
            continue
        used.add(band)
        flux = pl.col(flux_col)
        fluxerr = pl.col(fluxerr_col)
        # mag = -2.5 * log10(flux); null when flux <= 0 or non-finite
        mag_expr = (
            pl.when(flux.gt(0) & flux.is_finite())
            .then(-2.5 * (flux.log() / pl.lit(10.0).log()))
            .otherwise(None)
        )
        err_expr = (
            pl.when(flux.gt(0) & flux.is_finite() & fluxerr.is_finite())
            .then((1.0857362 * fluxerr / flux))  # 2.5/ln(10) ~ 1.086
            .otherwise(None)
        )
        df = df.with_columns(
            mag_expr.alias(band),
            err_expr.alias(f"{band}_err"),
        )
    return df


def _normalize_field_df(df: pl.DataFrame, field: str, mapping: dict[str, str]) -> pl.DataFrame:
    """Rename columns to canonical and add field, object_id. Fill missing bands with null."""
    # Rename only FITS columns that exist (no-op when df was built with canonical names)
    rename_sub = {k: v for k, v in mapping.items() if k in df.columns}
    if rename_sub:
        df = df.rename(rename_sub)
    # Prefer z_phot_err from (Z_HIGH68 - Z_LOW68)/2 when available (HSCpipe)
    if "Z_LOW68" in df.columns and "Z_HIGH68" in df.columns:
        zerr68 = (pl.col("Z_HIGH68") - pl.col("Z_LOW68")) / 2
        if "z_phot_err" in df.columns:
            df = df.with_columns(
                pl.when(zerr68.is_finite())
                .then(zerr68)
                .otherwise(pl.col("z_phot_err"))
                .alias("z_phot_err")
            )
        else:
            df = df.with_columns(zerr68.alias("z_phot_err"))
    all_canonical = (
        ["ra", "dec"]
        + CANONICAL_MAG_COLS
        + [f"{b}_err" for b in CANONICAL_MAG_COLS]
        + ["z_phot", "z_phot_err"]
    )
    for col in all_canonical:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
    existing = [c for c in all_canonical if c in df.columns]
    df = df.select(existing)
    n = df.height
    df = df.with_columns(
        pl.lit(field).alias("field"),
        (pl.lit(field) + "_" + pl.int_range(0, n).cast(pl.Utf8)).alias("object_id"),
    )
    return df


def merge_clauds(
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_MERGED_DIR,
    merged_filename: str = MERGED_FILENAME,
    keep_per_field_parquet: bool = True,
    use_sink_parquet: bool = True,
) -> Path:
    """
    Load each CLAUDS HSCpipe FITS, normalize to canonical schema, write per-field Parquet,
    then lazy concat and write merged Parquet.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_field_paths: list[Path] = []
    merged_fields: set[str] = set()

    for field, filename in HSCPIPE_FIELD_FILES:
        fits_path = raw_dir / filename
        if not fits_path.exists():
            raise FileNotFoundError(
                f"CLAUDS FITS not found: {fits_path}. Run tools/clauds_download.py first."
            )
        if fits_path.stat().st_size == 0:
            continue  # skip empty (incomplete download)
        tbl = _fits_to_astropy_table(fits_path)
        mapping = _build_mapping(tbl.colnames)
        pa_tbl = _astropy_table_to_pyarrow(tbl)
        df = pl.from_arrow(pa_tbl)
        df = _add_mags_from_fluxes(df)
        df = _normalize_field_df(df, field, mapping)
        out_field = output_dir / f"{field.replace('-', '_')}.parquet"
        df.write_parquet(out_field)
        per_field_paths.append(out_field)
        merged_fields.add(field)

    if not per_field_paths:
        raise FileNotFoundError(
            "No non-empty CLAUDS FITS found in "
            f"{raw_dir}. Re-run tools/clauds_download.py (files may have been 0-byte)."
        )
    expected_fields = {f[0] for f in HSCPIPE_FIELD_FILES}
    missing = expected_fields - merged_fields
    if missing:
        raise FileNotFoundError(
            f"Missing fields in merged catalog: {sorted(missing)}. "
            f"Ensure all four CLAUDS FITS are downloaded and complete (see tools/clauds_download.py)."
        )
    # Lazy concat and write merged
    merged_path = output_dir / merged_filename
    if use_sink_parquet and hasattr(pl.LazyFrame, "sink_parquet"):
        lazy = pl.concat(pl.scan_parquet(p) for p in per_field_paths)
        lazy.sink_parquet(merged_path)
    else:
        combined = pl.concat(pl.read_parquet(p) for p in per_field_paths)
        combined.write_parquet(merged_path)

    if not keep_per_field_parquet:
        for p in per_field_paths:
            p.unlink(missing_ok=True)

    return merged_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge CLAUDS HSCpipe FITS into one Parquet (Polars)."
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Directory with CLAUDS FITS."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_MERGED_DIR, help="Directory for Parquet output."
    )
    parser.add_argument(
        "--merged-filename", type=str, default=MERGED_FILENAME, help="Merged Parquet filename."
    )
    parser.add_argument(
        "--no-per-field", action="store_true", help="Do not keep per-field Parquet files."
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Use collect().write_parquet instead of sink_parquet.",
    )
    args = parser.parse_args()
    path = merge_clauds(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        merged_filename=args.merged_filename,
        keep_per_field_parquet=not args.no_per_field,
        use_sink_parquet=not args.no_streaming,
    )
    print(f"Merged catalog: {path}")


if __name__ == "__main__":
    main()
