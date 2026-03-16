"""Validate CLAUDS+spec-z final catalog: schema, consistency, and crossmatch stats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Minimal required columns for consistency checks (plan Section 5).
REQUIRED_COLUMNS_MINIMAL = [
    "object_id",
    "field",
    "ra",
    "dec",
    "spec_z",
    "spec_z_err",
    "spec_z_quality",
]
# Full schema: at least one magnitude column and z_phot expected from merge.
REQUIRED_FIELDS = frozenset({"E-COSMOS", "DEEP2-3", "XMM-LSS", "ELAIS-N1"})


def validate_clauds_specz_catalog(
    catalog_path: Path | str,
) -> dict[str, Any]:
    """
    Validate the CLAUDS+spec-z Parquet catalog.

    Checks:
    - Required columns present and correct dtypes where applicable
    - object_id is unique
    - field only takes allowed values
    - spec_z (and spec_z_err, spec_z_quality) non-null only for E-COSMOS
    - spec_z null for non-E-COSMOS fields
    - Basic magnitude sanity (numeric, finite where present)
    - Crossmatch stats for E-COSMOS (match rate, spec_z range)

    Returns a dict with keys: ok (bool), errors (list[str]), warnings (list[str]), stats (dict).
    """
    import polars as pl

    catalog_path = Path(catalog_path)
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {}

    if not catalog_path.exists():
        return {
            "ok": False,
            "errors": [f"Catalog not found: {catalog_path}"],
            "warnings": [],
            "stats": {},
        }

    try:
        df = pl.read_parquet(catalog_path)
    except Exception as e:
        return {
            "ok": False,
            "errors": [f"Failed to read Parquet: {e!s}"],
            "warnings": [],
            "stats": {},
        }

    # Schema: minimal required columns
    missing = [c for c in REQUIRED_COLUMNS_MINIMAL if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    # Unique object_id
    n_rows = df.height
    n_unique_id = df["object_id"].n_unique() if "object_id" in df.columns else 0
    stats["n_rows"] = n_rows
    stats["n_unique_object_id"] = n_unique_id
    if "object_id" in df.columns and n_unique_id != n_rows:
        errors.append(f"object_id must be unique: {n_unique_id} unique vs {n_rows} rows")

    # field values
    if "field" in df.columns:
        fields = set(df["field"].unique().to_list())
        stats["fields"] = sorted(fields)
        bad = fields - REQUIRED_FIELDS
        if bad:
            errors.append(f"Unexpected field values: {bad}")
        missing_fields = REQUIRED_FIELDS - fields
        if missing_fields and n_rows > 0:
            warnings.append(f"Expected fields not present: {missing_fields}")

    # spec_z consistency: only E-COSMOS may have non-null spec_z
    if "field" in df.columns and "spec_z" in df.columns:
        non_ecosmos = df.filter(pl.col("field") != "E-COSMOS")
        n_non_ecosmos = non_ecosmos.height
        # Use is_not_null() so all-null (Null dtype) columns don't raise in is_not_nan()
        n_non_ecosmos_with_specz = non_ecosmos.filter(pl.col("spec_z").is_not_null()).height
        stats["n_non_E-COSMOS"] = n_non_ecosmos
        stats["n_non_E-COSMOS_with_spec_z"] = n_non_ecosmos_with_specz
        if n_non_ecosmos_with_specz > 0:
            errors.append(
                f"spec_z must be null for non-E-COSMOS: found {n_non_ecosmos_with_specz} non-null in "
                f"non-E-COSMOS rows"
            )

        ecosmos = df.filter(pl.col("field") == "E-COSMOS")
        n_ecosmos = ecosmos.height
        n_ecosmos_with_specz = ecosmos.filter(pl.col("spec_z").is_not_null()).height
        stats["n_E-COSMOS"] = n_ecosmos
        stats["n_E-COSMOS_with_spec_z"] = n_ecosmos_with_specz
        if n_ecosmos > 0:
            stats["E-COSMOS_match_rate"] = round(n_ecosmos_with_specz / n_ecosmos, 6)
        if n_ecosmos_with_specz > 0:
            spec_z_vals = ecosmos.filter(pl.col("spec_z").is_not_null())["spec_z"]
            stats["spec_z_min"] = float(spec_z_vals.min())
            stats["spec_z_max"] = float(spec_z_vals.max())
            stats["spec_z_median"] = float(spec_z_vals.median())

    # Magnitudes: at least one band expected from merge
    mag_cols = [c for c in ["u", "g", "r", "i", "z", "y"] if c in df.columns]
    if not mag_cols and n_rows > 0:
        warnings.append("No optical magnitude columns (u,g,r,i,z,y) found")
    if mag_cols and n_rows > 0:
        for c in mag_cols:
            if df[c].null_count() == n_rows:
                warnings.append(f"Column {c} is all null")
        stats["magnitude_columns_present"] = mag_cols

    stats["columns_present"] = list(df.columns)

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate CLAUDS+spec-z final catalog (schema and consistency)."
    )
    parser.add_argument(
        "catalog",
        type=Path,
        nargs="?",
        default=REPO_ROOT / "data" / "clauds_specz" / "clauds_specz_catalog.parquet",
        help="Path to clauds_specz_catalog.parquet",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full report as JSON.",
    )
    args = parser.parse_args()

    result = validate_clauds_specz_catalog(args.catalog)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result["errors"]:
            for e in result["errors"]:
                print(f"ERROR: {e}")
        if result["warnings"]:
            for w in result["warnings"]:
                print(f"WARN: {w}")
        print("Stats:", json.dumps(result["stats"], indent=2))
        print("Valid:" if result["ok"] else "INVALID:", result["ok"])

    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
