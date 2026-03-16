"""Crossmatch CLAUDS merged catalog with COSMOS Spec-z Compilation (E-COSMOS only)."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MERGED_PATH = REPO_ROOT / "data" / "clauds_specz" / "clauds_merged_photometry.parquet"
DEFAULT_SPECZ_PATH = (
    REPO_ROOT
    / "data"
    / "clauds_specz"
    / "specz_compilation"
    / "specz_compilation_COSMOS_DR1.1_unique.fits"
)
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "clauds_specz" / "clauds_specz_catalog.parquet"
MATCH_RADIUS_ARCSEC = 1.0
SECURE_QUALITY_FLAGS = (3, 4, 9)


def _load_specz_fits(
    path: Path, quality_flags: tuple[int, ...] = SECURE_QUALITY_FLAGS
) -> pl.DataFrame:
    from astropy.table import Table

    t = Table.read(path)
    pdf = t.to_pandas()
    df = pl.from_pandas(pdf)

    lower = {c.lower(): c for c in df.columns}
    # COSMOS speczcompilation uses ra_corrected, dec_corrected, specz, flag
    ra_col = (
        lower.get("ra")
        or lower.get("ra_corrected")
        or lower.get("coord_ra")
        or next((c for c in df.columns if "ra" in c.lower()), None)
    )
    dec_col = (
        lower.get("dec")
        or lower.get("dec_corrected")
        or lower.get("coord_dec")
        or next((c for c in df.columns if "dec" in c.lower()), None)
    )
    z_col = (
        lower.get("z")
        or lower.get("specz")
        or lower.get("redshift")
        or lower.get("z_spec")
        or next((c for c in df.columns if "redshift" in c.lower() or c.lower() == "z"), None)
    )
    q_col = (
        lower.get("quality")
        or lower.get("quality_flag")
        or lower.get("flag")
        or next((c for c in df.columns if "qual" in c.lower() or "flag" in c.lower()), None)
    )

    if ra_col is None or dec_col is None or z_col is None:
        raise ValueError(
            f"Spec-z FITS must have RA/Dec/redshift columns. Found: {list(df.columns)}"
        )

    df = df.rename({ra_col: "ra_spec", dec_col: "dec_spec", z_col: "z_spec"})
    if q_col is not None:
        df = df.filter(pl.col(q_col).is_in(list(quality_flags)))
    return df.select(["ra_spec", "dec_spec", "z_spec"])


def crossmatch_clauds_specz(
    *,
    merged_parquet: Path = DEFAULT_MERGED_PATH,
    specz_fits: Path = DEFAULT_SPECZ_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    match_radius_arcsec: float = MATCH_RADIUS_ARCSEC,
    quality_flags: tuple[int, ...] = SECURE_QUALITY_FLAGS,
    use_sink_parquet: bool = True,
) -> Path:
    """
    Load spec-z FITS, filter by quality. Load E-COSMOS from merged Parquet.
    Positional match (1"), attach spec_z columns. Concat with other fields and write.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    merged_parquet = Path(merged_parquet)
    specz_fits = Path(specz_fits)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not specz_fits.exists():
        # Write merged catalog with null spec_z columns so pipeline can proceed; re-run with
        # --specz-fits once the FITS is available (e.g. from specz_compilation_download.py).
        import warnings

        warnings.warn(
            f"Spec-z FITS not found: {specz_fits}. Writing catalog with null spec_z; "
            "re-run with --specz-fits after obtaining the file (e.g. tools/specz_compilation_download.py).",
            UserWarning,
            stacklevel=1,
        )
        full = pl.scan_parquet(merged_parquet).collect()
        full = full.with_columns(
            pl.lit(None).cast(pl.Float64).alias("spec_z"),
            pl.lit(None).cast(pl.Float64).alias("spec_z_err"),
            pl.lit(None).cast(pl.Int64).alias("spec_z_quality"),
        )
        if use_sink_parquet and hasattr(pl.LazyFrame, "sink_parquet"):
            full.lazy().sink_parquet(output_path)
        else:
            full.write_parquet(output_path)
        return output_path

    spec_df = _load_specz_fits(specz_fits, quality_flags=quality_flags)
    # Spec-z RA/Dec may be in deg (COSMOS usually is)
    spec_ra = spec_df["ra_spec"].to_list()
    spec_dec = spec_df["dec_spec"].to_list()
    spec_z = spec_df["z_spec"].to_list()
    spec_coord = SkyCoord(ra=spec_ra * u.deg, dec=spec_dec * u.deg)

    # Load only E-COSMOS from merged catalog
    eco = pl.scan_parquet(merged_parquet).filter(pl.col("field") == "E-COSMOS").collect()
    phot_ra = eco["ra"].to_list()
    phot_dec = eco["dec"].to_list()
    # RA/Dec in merged may be rad or deg; HSC often uses rad. Convert to deg if needed.
    if phot_ra and (max(abs(x) for x in phot_ra) < 1.0 or min(phot_ra) < 0):
        phot_ra = [math.degrees(x) for x in phot_ra]
        phot_dec = [math.degrees(x) for x in phot_dec]
    phot_coord = SkyCoord(ra=phot_ra * u.deg, dec=phot_dec * u.deg)

    idx, sep2d, _ = phot_coord.match_to_catalog_sky(spec_coord, nthneighbor=1)
    sep_arcsec = sep2d.arcsec

    # For each phot row: if sep <= radius, assign spec_z from spec_df row idx[i]
    n = len(phot_ra)
    spec_z_matched: list[float | None] = [None] * n
    spec_z_err_matched: list[float | None] = [None] * n
    spec_z_quality_matched: list[int | None] = [None] * n
    for i in range(n):
        if sep_arcsec[i] <= match_radius_arcsec:
            j = int(idx[i])
            spec_z_matched[i] = spec_z[j]
            spec_z_err_matched[i] = 0.001  # nominal if not in FITS
            spec_z_quality_matched[i] = SECURE_QUALITY_FLAGS[0]

    eco = eco.with_columns(
        pl.Series("spec_z", spec_z_matched),
        pl.Series("spec_z_err", spec_z_err_matched),
        pl.Series("spec_z_quality", spec_z_quality_matched),
    )

    # Other fields: no spec_z
    other = pl.scan_parquet(merged_parquet).filter(pl.col("field") != "E-COSMOS").collect()
    other = other.with_columns(
        pl.lit(None).cast(pl.Float64).alias("spec_z"),
        pl.lit(None).cast(pl.Float64).alias("spec_z_err"),
        pl.lit(None).cast(pl.Int64).alias("spec_z_quality"),
    )

    if use_sink_parquet and hasattr(pl.LazyFrame, "sink_parquet"):
        pl.concat([eco.lazy(), other.lazy()]).sink_parquet(output_path)
    else:
        pl.concat([eco, other]).write_parquet(output_path)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crossmatch CLAUDS merged catalog with COSMOS Spec-z Compilation (E-COSMOS)."
    )
    parser.add_argument(
        "--merged-parquet", type=Path, default=DEFAULT_MERGED_PATH, help="Merged CLAUDS Parquet."
    )
    parser.add_argument(
        "--specz-fits",
        type=Path,
        default=DEFAULT_SPECZ_PATH,
        help="Spec-z compilation unique FITS.",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output Parquet path."
    )
    parser.add_argument(
        "--radius-arcsec", type=float, default=MATCH_RADIUS_ARCSEC, help="Match radius in arcsec."
    )
    parser.add_argument(
        "--no-streaming", action="store_true", help="Use collect().write_parquet instead of sink."
    )
    args = parser.parse_args()
    path = crossmatch_clauds_specz(
        merged_parquet=args.merged_parquet,
        specz_fits=args.specz_fits,
        output_path=args.output,
        match_radius_arcsec=args.radius_arcsec,
        use_sink_parquet=not args.no_streaming,
    )
    print(f"Catalog with spec-z: {path}")


if __name__ == "__main__":
    main()
