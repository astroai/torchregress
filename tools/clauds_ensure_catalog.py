"""Ensure all CLAUDS HSCpipe catalogues are downloaded and merged before crossmatch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "clauds_specz" / "raw"
DEFAULT_MERGED_DIR = REPO_ROOT / "data" / "clauds_specz"
MERGED_FILENAME = "clauds_merged_photometry.parquet"
EXPECTED_FIELDS = {"E-COSMOS", "DEEP2-3", "XMM-LSS", "ELAIS-N1"}


def ensure_clauds_catalog(
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_MERGED_DIR,
    merged_filename: str = MERGED_FILENAME,
    overwrite: bool = False,
    use_wget: bool | None = None,
) -> Path:
    """
    Download all four CLAUDS HSCpipe FITS (re-download incomplete), merge to Parquet,
    and verify the merged catalog has all four fields. Raises on failure.
    """
    from tools.clauds_download import (
        BASE_URL,
        HSCPIPE_FILES,
        _expected_fits_size,
        _expected_md5_from_headers,
        _file_md5,
        download_clauds,
    )
    from tools.clauds_merge import merge_clauds

    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    # 1) Download (re-downloads 0-byte and truncated)
    print("Downloading CLAUDS HSCpipe FITS...")
    results = download_clauds(
        output_dir=raw_dir,
        hscpipe=True,
        sextractor=False,
        overwrite=overwrite,
        use_wget=use_wget,
    )
    for name, status in results.items():
        print(f"  {name}: {status}")
    errors = [n for n, s in results.items() if s.startswith("error:")]
    if errors:
        raise RuntimeError(f"Download failed for: {errors}")

    # 2) Verify all four files exist and pass integrity (MD5 from URL headers, else FITS size)
    missing = []
    incomplete = []
    base = BASE_URL.rstrip("/")
    for filename in HSCPIPE_FILES:
        path = raw_dir / filename
        if not path.exists():
            missing.append(filename)
            continue
        size = path.stat().st_size
        if size == 0:
            incomplete.append(f"{filename} (0-byte)")
            continue
        url = f"{base}/{filename}"
        expected_md5 = _expected_md5_from_headers(url)
        if expected_md5:
            if _file_md5(path) != expected_md5:
                incomplete.append(f"{filename} (MD5 mismatch)")
        else:
            expected_size = _expected_fits_size(path)
            if expected_size is None:
                incomplete.append(f"{filename} (FITS header unreadable)")
            elif size < expected_size:
                incomplete.append(f"{filename} ({size} < {expected_size})")
    if missing:
        raise FileNotFoundError(
            f"Missing FITS: {missing}. Run with --overwrite to force re-download."
        )
    if incomplete:
        raise FileNotFoundError(f"Incomplete FITS (re-download with --overwrite): {incomplete}")

    # 3) Merge
    print("Merging CLAUDS FITS...")
    merged_path = merge_clauds(
        raw_dir=raw_dir,
        output_dir=output_dir,
        merged_filename=merged_filename,
        keep_per_field_parquet=True,
        use_sink_parquet=True,
    )
    print(f"  Merged: {merged_path}")

    # 4) Verify merged catalog has all four fields
    import polars as pl

    df = pl.read_parquet(merged_path)
    fields = set(df["field"].unique().to_list())
    if fields != EXPECTED_FIELDS:
        missing_f = EXPECTED_FIELDS - fields
        extra_f = fields - EXPECTED_FIELDS
        msg = []
        if missing_f:
            msg.append(f"missing fields: {sorted(missing_f)}")
        if extra_f:
            msg.append(f"unexpected fields: {sorted(extra_f)}")
        raise RuntimeError(f"Merged catalog validation failed: {'; '.join(msg)}")

    print(f"  Verified: {len(fields)} fields, {df.shape[0]} rows")
    return merged_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ensure all CLAUDS HSCpipe catalogues are downloaded and merged."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory with CLAUDS FITS.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MERGED_DIR,
        help="Directory for merged Parquet.",
    )
    parser.add_argument(
        "--merged-filename",
        type=str,
        default=MERGED_FILENAME,
        help="Merged Parquet filename.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing FITS (re-download all).",
    )
    parser.add_argument(
        "--no-wget",
        action="store_true",
        help="Use Python urllib instead of wget.",
    )
    args = parser.parse_args()
    try:
        ensure_clauds_catalog(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            merged_filename=args.merged_filename,
            overwrite=args.overwrite,
            use_wget=not args.no_wget,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
