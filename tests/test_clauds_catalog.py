"""Tests for CLAUDS merge, spec-z crossmatch, and final catalog validation."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("polars")
pytest.importorskip("astropy")

import numpy as np
import polars as pl  # type: ignore
from astropy.table import Table  # type: ignore

from tools import clauds_catalog_validate, clauds_merge, clauds_specz_crossmatch


def _write_mini_clauds_fits(path: Path, field_key: str, n_rows: int = 30) -> None:
    """Write a minimal CLAUDS-like FITS with columns merge can map (ra, dec, mags, z_phot)."""
    rng = np.random.default_rng(42)
    # Use column names that clauds_merge COLUMN_CANDIDATES will find
    ra = 148.0 + rng.uniform(-0.5, 0.5, n_rows)
    dec = 2.0 + rng.uniform(-0.5, 0.5, n_rows)
    u = 22.0 + rng.uniform(0, 2, n_rows)
    g = 21.0 + rng.uniform(0, 2, n_rows)
    r = 20.0 + rng.uniform(0, 2, n_rows)
    i = 19.5 + rng.uniform(0, 2, n_rows)
    z = 19.0 + rng.uniform(0, 2, n_rows)
    y = 18.5 + rng.uniform(0, 2, n_rows)
    z_phot = 0.5 + rng.uniform(0, 1.5, n_rows)
    z_phot_err = 0.05 + rng.uniform(0, 0.05, n_rows)
    t = Table(
        {
            "ra": ra,
            "dec": dec,
            "u": u,
            "g": g,
            "r": r,
            "i": i,
            "z": z,
            "y": y,
            "z_phot": z_phot,
            "z_phot_err": z_phot_err,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    t.write(path, format="fits", overwrite=True)


def _write_mini_specz_fits(
    path: Path,
    ra_deg: list[float],
    dec_deg: list[float],
    z: list[float],
    quality: list[int] | None = None,
) -> None:
    """Write minimal spec-z FITS (RA, Dec, z, quality) for crossmatch."""
    n = len(ra_deg)
    if quality is None:
        quality = [3] * n
    t = Table(
        {
            "ra": ra_deg,
            "dec": dec_deg,
            "z": z,
            "quality": quality,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    t.write(path, format="fits", overwrite=True)


def test_merge_produces_canonical_schema_and_unique_object_id(tmp_path: Path) -> None:
    """Merge of minimal CLAUDS FITS yields parquet with required schema and unique object_id."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "out"
    # Merge expects these exact filenames (HSCPIPE_FIELD_FILES)
    for field, fname in [
        ("E-COSMOS", "COSMOS-HSCpipe-Phosphoros.fits"),
        ("DEEP2-3", "DEEP23-HSCpipe-Phosphoros.fits"),
        ("XMM-LSS", "XMMLSS-HSCpipe-Phosphoros.fits"),
        ("ELAIS-N1", "ELAIS-N1-HSCpipe-Phosphoros.fits"),
    ]:
        _write_mini_clauds_fits(raw_dir / fname, field, n_rows=25)

    merged_path = clauds_merge.merge_clauds(
        raw_dir=raw_dir,
        output_dir=out_dir,
        keep_per_field_parquet=True,
        use_sink_parquet=True,
    )
    assert merged_path.exists()
    df = pl.read_parquet(merged_path)
    assert df.height == 100  # 4 * 25
    assert "object_id" in df.columns
    assert df["object_id"].n_unique() == 100
    assert "field" in df.columns
    assert set(df["field"].unique().to_list()) == {
        "E-COSMOS",
        "DEEP2-3",
        "XMM-LSS",
        "ELAIS-N1",
    }
    for col in ["ra", "dec", "u", "g", "r", "i", "z", "y", "z_phot", "z_phot_err"]:
        assert col in df.columns


def test_crossmatch_attaches_spec_z_only_for_e_cosmos_within_radius(
    tmp_path: Path,
) -> None:
    """Crossmatch leaves spec_z null for non-E-COSMOS; E-COSMOS gets spec_z only within 1\"."""
    # Build minimal merged parquet: E-COSMOS with a few positions, DEEP2-3 without
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    merged_path = out_dir / "merged.parquet"
    specz_path = tmp_path / "specz" / "specz_unique.fits"
    catalog_path = out_dir / "catalog.parquet"

    # E-COSMOS: 20 rows; first 5 have (ra, dec) we will match with spec-z
    ra_eco = [150.0 + i * 0.001 for i in range(20)]
    dec_eco = [2.0 + i * 0.001 for i in range(20)]
    # DEEP2-3: 10 rows, no overlap with spec-z
    ra_d23 = [200.0 + i * 0.01 for i in range(10)]
    dec_d23 = [5.0 + i * 0.01 for i in range(10)]

    def _make_df(ra_list: list[float], dec_list: list[float], field: str) -> pl.DataFrame:
        n = len(ra_list)
        return pl.DataFrame(
            {
                "ra": ra_list,
                "dec": dec_list,
                "u": [22.0] * n,
                "g": [21.0] * n,
                "r": [20.0] * n,
                "i": [19.5] * n,
                "z": [19.0] * n,
                "y": [18.5] * n,
                "z_phot": [0.5] * n,
                "z_phot_err": [0.05] * n,
                "field": [field] * n,
                "object_id": [f"{field}_{i}" for i in range(n)],
            }
        )

    eco = _make_df(ra_eco, dec_eco, "E-COSMOS")
    d23 = _make_df(ra_d23, dec_d23, "DEEP2-3")
    # Add missing canonical columns as null so merge schema is satisfied
    for band in ["Y", "J", "H", "Ks"]:
        eco = eco.with_columns(pl.lit(None).cast(pl.Float64).alias(band))
        d23 = d23.with_columns(pl.lit(None).cast(pl.Float64).alias(band))
    for b in ["u", "g", "r", "i", "z", "y", "Y", "J", "H", "Ks"]:
        eco = eco.with_columns(pl.lit(None).cast(pl.Float64).alias(f"{b}_err"))
        d23 = d23.with_columns(pl.lit(None).cast(pl.Float64).alias(f"{b}_err"))
    pl.concat([eco, d23]).write_parquet(merged_path)

    # Spec-z: 5 sources at (ra, dec) matching first 5 E-COSMOS (within 1")
    _write_mini_specz_fits(
        specz_path,
        ra_deg=ra_eco[:5],
        dec_deg=dec_eco[:5],
        z=[0.1 * i for i in range(5)],
        quality=[3] * 5,
    )

    clauds_specz_crossmatch.crossmatch_clauds_specz(
        merged_parquet=merged_path,
        specz_fits=specz_path,
        output_path=catalog_path,
        match_radius_arcsec=1.0,
        use_sink_parquet=True,
    )
    df = pl.read_parquet(catalog_path)
    # Non-E-COSMOS must have null spec_z
    non_eco = df.filter(pl.col("field") != "E-COSMOS")
    assert non_eco.filter(pl.col("spec_z").is_not_nan()).height == 0
    # E-COSMOS: exactly 5 matched (we put 5 spec-z at same positions)
    eco = df.filter(pl.col("field") == "E-COSMOS")
    n_matched = eco.filter(pl.col("spec_z").is_not_nan()).height
    assert n_matched == 5
    assert eco.height == 20


def test_validate_accepts_valid_catalog(tmp_path: Path) -> None:
    """Validation passes for a catalog with correct spec_z/field consistency."""
    path = tmp_path / "valid_catalog.parquet"
    df = pl.DataFrame(
        {
            "object_id": ["E-COSMOS_0", "E-COSMOS_1", "DEEP2-3_0"],
            "field": ["E-COSMOS", "E-COSMOS", "DEEP2-3"],
            "ra": [150.0, 150.01, 200.0],
            "dec": [2.0, 2.01, 5.0],
            "spec_z": [0.5, 1.0, None],
            "spec_z_err": [0.01, 0.01, None],
            "spec_z_quality": [3, 3, None],
            "u": [22.0, 21.5, 23.0],
            "g": [21.0, 20.5, 22.0],
        }
    )
    df.write_parquet(path)
    result = clauds_catalog_validate.validate_clauds_specz_catalog(path)
    assert result["ok"] is True
    assert result["stats"]["n_rows"] == 3
    assert result["stats"]["n_E-COSMOS_with_spec_z"] == 2
    assert result["stats"]["n_non_E-COSMOS_with_spec_z"] == 0


def test_validate_rejects_spec_z_on_non_ecosmos(tmp_path: Path) -> None:
    """Validation fails when a non-E-COSMOS row has non-null spec_z."""
    path = tmp_path / "invalid_catalog.parquet"
    df = pl.DataFrame(
        {
            "object_id": ["DEEP2-3_0"],
            "field": ["DEEP2-3"],
            "ra": [200.0],
            "dec": [5.0],
            "spec_z": [0.5],  # invalid: DEEP2-3 must not have spec_z
            "spec_z_err": [0.01],
            "spec_z_quality": [3],
            "u": [23.0],
            "g": [22.0],
        }
    )
    df.write_parquet(path)
    result = clauds_catalog_validate.validate_clauds_specz_catalog(path)
    assert result["ok"] is False
    assert any("non-E-COSMOS" in e for e in result["errors"])


def test_validate_rejects_duplicate_object_id(tmp_path: Path) -> None:
    """Validation fails when object_id is not unique."""
    path = tmp_path / "dup_id.parquet"
    df = pl.DataFrame(
        {
            "object_id": ["E-COSMOS_0", "E-COSMOS_0"],
            "field": ["E-COSMOS", "E-COSMOS"],
            "ra": [150.0, 150.01],
            "dec": [2.0, 2.01],
            "spec_z": [None, None],
            "spec_z_err": [None, None],
            "spec_z_quality": [None, None],
            "u": [22.0, 21.0],
            "g": [21.0, 20.0],
        }
    )
    df.write_parquet(path)
    result = clauds_catalog_validate.validate_clauds_specz_catalog(path)
    assert result["ok"] is False
    assert any("unique" in e.lower() for e in result["errors"])


def test_build_all_bands_with_mask_df() -> None:
    """build_all_bands_with_mask_df produces mags, errs, obs mask; drops rows with too few optical bands."""
    import sys

    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    import photoz_clauds_specz_comparison as clauds_ex  # type: ignore

    build_all_bands_with_mask_df = clauds_ex.build_all_bands_with_mask_df
    ALL_BANDS = clauds_ex.ALL_BANDS

    rng = np.random.default_rng(77)
    n = 50
    df = __import__("pandas").DataFrame(
        {
            "u": 22.0 + rng.uniform(0, 1, n),
            "g": 21.0 + rng.uniform(0, 1, n),
            "r": 20.0 + rng.uniform(0, 1, n),
            "i": 19.5 + rng.uniform(0, 1, n),
            "z": 19.0 + rng.uniform(0, 1, n),
            "y": 18.5 + rng.uniform(0, 1, n),
            "u_err": 0.1,
            "g_err": 0.1,
            "r_err": 0.1,
            "i_err": 0.1,
            "z_err": 0.1,
            "y_err": 0.1,
            "Y": [np.nan] * n,
            "J": [np.nan] * n,
            "H": [np.nan] * n,
            "Ks": [np.nan] * n,
            "Y_err": [np.nan] * n,
            "J_err": [np.nan] * n,
            "H_err": [np.nan] * n,
            "Ks_err": [np.nan] * n,
            "spec_z": 0.5 + rng.uniform(0, 1, n),
            "spec_z_err": 0.05,
        }
    )
    out = build_all_bands_with_mask_df(df, min_optical_bands=3)
    assert len(out) == 50
    assert list(out.columns[:10]) == ALL_BANDS
    assert "obs_u" in out.columns and "obs_Y" in out.columns
    assert out["obs_Y"].sum() == 0  # NIR was all null
    assert out["obs_u"].sum() == 50  # optical all present
    assert (out["Y"] == 0).all()  # filled missing with 0
    assert "spec_z" in out.columns and "spec_z_err" in out.columns

    # Drop optical so some rows have < 3 optical bands and are filtered out
    df_few = df.copy()
    df_few.loc[:24, ["u", "g", "r", "i"]] = np.nan  # first 25 rows only have z,y (2 optical)
    out_few = build_all_bands_with_mask_df(df_few, min_optical_bands=3)
    assert len(out_few) == 25  # only rows 25..49 have >= 3 optical


def test_stratified_split_indices_shapes_and_disjoint() -> None:
    """Stratified split returns disjoint train/cal/test of exact sizes."""
    import sys

    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    import photoz_clauds_specz_comparison as clauds_ex

    _stratified_split_indices = clauds_ex._stratified_split_indices

    n_total = 500
    stratify = np.linspace(0.0, 2.0, n_total)  # redshift-like
    ti, ci, tei = _stratified_split_indices(
        n_total, n_train=300, n_cal=100, n_test=100, stratify_values=stratify, seed=42
    )
    assert len(ti) == 300
    assert len(ci) == 100
    assert len(tei) == 100
    assert len(set(ti) | set(ci) | set(tei)) == 500
    assert len(set(ti) & set(ci)) == 0
    assert len(set(ti) & set(tei)) == 0
    assert len(set(ci) & set(tei)) == 0


def test_validate_missing_file() -> None:
    """Validation returns ok=False for missing path."""
    result = clauds_catalog_validate.validate_clauds_specz_catalog(
        Path("/nonexistent/clauds_specz_catalog.parquet")
    )
    assert result["ok"] is False
    assert "not found" in result["errors"][0].lower()


def _make_merged_parquet_with_mag_errors(
    path: Path,
    n_eco: int = 200,
    n_other: int = 50,
) -> None:
    """Write merged parquet with E-COSMOS and DEEP2-3, magnitude columns and errors for colors."""
    rng = np.random.default_rng(99)
    ra_eco = 150.0 + rng.uniform(-0.1, 0.1, n_eco)
    dec_eco = 2.0 + rng.uniform(-0.1, 0.1, n_eco)
    ra_other = 200.0 + rng.uniform(-0.1, 0.1, n_other)
    dec_other = 5.0 + rng.uniform(-0.1, 0.1, n_other)

    def _rows(ra_list: list[float], dec_list: list[float], field: str) -> pl.DataFrame:
        n = len(ra_list)
        return pl.DataFrame(
            {
                "ra": ra_list,
                "dec": dec_list,
                "u": 22.0 + rng.uniform(0, 1, n),
                "g": 21.0 + rng.uniform(0, 1, n),
                "r": 20.0 + rng.uniform(0, 1, n),
                "i": 19.5 + rng.uniform(0, 1, n),
                "z": 19.0 + rng.uniform(0, 1, n),
                "y": 18.5 + rng.uniform(0, 1, n),
                "z_phot": 0.5 + rng.uniform(0, 1, n),
                "z_phot_err": 0.05 + rng.uniform(0, 0.02, n),
                "field": [field] * n,
                "object_id": [f"{field}_{i}" for i in range(n)],
            }
        )

    eco = _rows(ra_eco.tolist(), dec_eco.tolist(), "E-COSMOS")
    other = _rows(ra_other.tolist(), dec_other.tolist(), "DEEP2-3")
    for band in ["u", "g", "r", "i", "z", "y"]:
        eco = eco.with_columns(pl.lit(0.1).alias(f"{band}_err"))
        other = other.with_columns(pl.lit(0.1).alias(f"{band}_err"))
    for band in ["Y", "J", "H", "Ks"]:
        eco = eco.with_columns(pl.lit(None).cast(pl.Float64).alias(band))
        other = other.with_columns(pl.lit(None).cast(pl.Float64).alias(band))
    for band in ["Y", "J", "H", "Ks"]:
        eco = eco.with_columns(pl.lit(None).cast(pl.Float64).alias(f"{band}_err"))
        other = other.with_columns(pl.lit(None).cast(pl.Float64).alias(f"{band}_err"))
    pl.concat([eco, other]).write_parquet(path)


def test_pipeline_validates_after_build(tmp_path: Path) -> None:
    """When pipeline builds the catalog, it runs validation and fails if invalid."""

    catalog_path = tmp_path / "clauds_specz_catalog.parquet"
    merged_path = tmp_path / "merged.parquet"
    specz_path = tmp_path / "specz" / "specz_unique.fits"
    specz_path.parent.mkdir(parents=True, exist_ok=True)

    _make_merged_parquet_with_mag_errors(merged_path, n_eco=200, n_other=50)
    df_merged = pl.read_parquet(merged_path)
    eco = df_merged.filter(pl.col("field") == "E-COSMOS")
    ra = eco["ra"].to_list()
    dec = eco["dec"].to_list()
    _write_mini_specz_fits(
        specz_path,
        ra_deg=ra,
        dec_deg=dec,
        z=[0.3 + 0.01 * i for i in range(len(ra))],
        quality=[3] * len(ra),
    )

    clauds_specz_crossmatch.crossmatch_clauds_specz(
        merged_parquet=merged_path,
        specz_fits=specz_path,
        output_path=catalog_path,
        match_radius_arcsec=1.0,
    )
    validation = clauds_catalog_validate.validate_clauds_specz_catalog(catalog_path)
    assert validation["ok"], validation["errors"]
    assert validation["stats"]["n_E-COSMOS_with_spec_z"] == 200
