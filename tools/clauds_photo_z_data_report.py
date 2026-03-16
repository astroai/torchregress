"""Report on CLAUDS+spec-z catalog for photo-z: labels, completeness, and split recommendations."""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = REPO_ROOT / "data" / "clauds_specz" / "clauds_specz_catalog.parquet"
MAG_COLS = ["u", "g", "r", "i", "z", "y", "Y", "J", "H", "Ks"]
NIR_COLS = ["Y", "J", "H", "Ks"]

# Feature sets from the CLAUDS photo-z paper (Desprez et al. 2023, A&A, arXiv:2301.13750).
# Phosphoros SED-fitting uses Ugrizy (optical) and optionally YJHKs (NIR) where available.
# Use these to compare "paper SED features" vs "all magnitude features" in benchmarks.
PAPER_SED_OPTICAL_BANDS = ["u", "g", "r", "i", "z", "y"]  # Ugrizy only
PAPER_SED_OPTICAL_PLUS_NIR_BANDS = [
    "u",
    "g",
    "r",
    "i",
    "z",
    "y",
    "Y",
    "J",
    "H",
    "Ks",
]  # Ugrizy + YJHKs


def run_report(catalog_path: Path) -> None:
    import polars as pl

    p = pl.read_parquet(catalog_path)
    n_total = p.height

    print("=" * 60)
    print("CLAUDS+spec-z catalog: photo-z data report")
    print("=" * 60)
    print(f"Catalog: {catalog_path}")
    print(f"Total rows: {n_total:,}")
    print()

    # --- Spec-z labels ---
    labeled = p.filter(pl.col("spec_z").is_not_null())
    n_specz = labeled.height
    print("--- Spec-z labels ---")
    print(
        f"Rows with spec_z (labeled): {n_specz:,} of {n_total:,} ({100 * n_specz / n_total:.2f}%)"
    )
    print()
    print("Per field:")
    t = (
        p.group_by("field")
        .agg(
            pl.len().alias("n"),
            pl.col("spec_z").drop_nulls().len().alias("n_specz"),
        )
        .with_columns((100 * pl.col("n_specz") / pl.col("n")).round(2).alias("pct_specz"))
    )
    print(t)
    print()

    if n_specz > 0:
        specz = labeled["spec_z"]
        print("Spec-z distribution (labeled only):")
        print(f"  min={specz.min():.3f}  max={specz.max():.3f}")
        print(f"  mean={specz.mean():.3f}  median={specz.median():.3f}")
        print(
            f"  percentiles: 5%={specz.quantile(0.05):.3f}  25%={specz.quantile(0.25):.3f}  "
            f"50%={specz.quantile(0.5):.3f}  75%={specz.quantile(0.75):.3f}  95%={specz.quantile(0.95):.3f}"
        )
    print()

    # --- Magnitude completeness ---
    print("--- Magnitude completeness ---")
    mag_exist = [c for c in MAG_COLS if c in p.columns]
    if not mag_exist:
        print("  No magnitude columns in catalog.")
    elif mag_exist and p[mag_exist[0]].null_count() == n_total:
        print("  All magnitude columns are null (current merge does not populate mags;")
        print("   HSC FITS use FLUX_* columns; add flux->mag or flux features for photo-z.).")
        print("  Rows with missing ANY band: all (100%) until magnitudes are populated.")
        print("  Rows with missing ANY NIR (Y,J,H,Ks): all (100%) until magnitudes are populated.")
    else:
        for c in mag_exist:
            n_ok = p.filter(pl.col(c).is_not_null() & pl.col(c).is_finite()).height
            print(f"  {c}: {n_ok:,} non-null ({100 * n_ok / n_total:.1f}%)")
        expr_any = pl.any_horizontal(pl.col(c).is_null() for c in mag_exist)
        n_any_missing = p.filter(expr_any).height
        print(f"  Rows missing ANY band: {n_any_missing:,} ({100 * n_any_missing / n_total:.1f}%)")
        nir_exist = [c for c in NIR_COLS if c in p.columns]
        if nir_exist:
            expr_nir = pl.any_horizontal(pl.col(c).is_null() for c in nir_exist)
            n_nir_missing = p.filter(expr_nir).height
            print(
                f"  Rows missing ANY NIR (Y,J,H,Ks): {n_nir_missing:,} ({100 * n_nir_missing / n_total:.1f}%)"
            )
    print()

    # --- Features available for photo-z ---
    print("--- Features currently available ---")
    print("  Always: ra, dec, z_phot, z_phot_err, field, object_id")
    print("  When matched: spec_z, spec_z_err, spec_z_quality")
    mag_filled = mag_exist and p[mag_exist[0]].null_count() < n_total
    if mag_filled:
        print(
            "  Magnitudes: u,g,r,i,z,y (and Y,J,H,Ks where available) from FLUX_APER_* conversion."
        )
    else:
        print(
            "  Magnitudes: columns present but unpopulated (use flux->mag in merge or flux-based features)."
        )
    print()

    # --- Paper SED vs all features (for comparison) ---
    print("--- Feature sets: paper SED-fitting vs all (for photo-z comparison) ---")
    print("  Reference: Desprez et al. 2023, A&A, arXiv:2301.13750")
    print("  'Combining the CLAUDS & HSC-SSP surveys: U+grizy(+YJHKs) photometry and")
    print("   photometric redshifts for 18M galaxies...' — Phosphoros template SED-fitting.")
    print("  Paper uses:")
    print("    - Ugrizy (optical): u, g, r, i, z, y  → σ_NMAD ≲ 0.04, η ≲ 10% at m_i ~ 25")
    print("    - Ugrizy + YJHKs (NIR where available): η ≲ 6% at m_i ~ 25")
    print("  Feature sets for benchmarks (use same magnitude columns ± errors):")
    print("    - Paper SED (optical only): " + ", ".join(PAPER_SED_OPTICAL_BANDS))
    print("    - Paper SED (optical + NIR): " + ", ".join(PAPER_SED_OPTICAL_PLUS_NIR_BANDS))
    print("    - All magnitudes in catalog: same bands as above (no extra bands in merge).")
    print("  Compare: train with (1) Ugrizy only vs (2) Ugrizy + NIR (where available)")
    print("  to mirror paper SED setup and see gain from NIR.")
    print()
    print("3) All bands with missing data (outer join of all catalogues)")
    print(
        "   - Use the full catalog: all fields merged, all band columns (u..y, Y,J,H,Ks + errors)."
    )
    print(
        "   - Do not drop rows with missing bands: NIR is null for most fields (only 2 have NIR);"
    )
    print("     some optical can be null. Build a per-sample observed-band mask.")
    print(
        "   - Train semi-supervised: labeled (spec_z) + unlabeled (no spec_z), both with possible"
    )
    print("     missing bands. Use loss that respects sample mask; pass observed-band mask so the")
    print(
        "     model can propagate missingness into uncertainty (wider intervals when fewer bands)."
    )
    print("   - Expectation: predictions with fewer observed bands should get wider redshift")
    print("     intervals than when all bands are present, if the pipeline is working correctly.")
    print()

    # --- Split recommendations ---
    n_ecosmos = p.filter(pl.col("field") == "E-COSMOS").height
    print("--- Split recommendations for supervised and semi-supervised ---")
    print()
    print("1) Supervised (labeled only)")
    print("   - Use E-COSMOS rows with non-null spec_z only (n={:,}).".format(n_specz))
    print(
        "   - Train/val/test: e.g. 80/10/10 stratified by spec_z bins (e.g. 5 bins: 0–0.3, 0.3–0.6, 0.6–1, 1–1.5, 1.5+) to keep redshift distribution."
    )
    print(
        "   - Optional: stratify by magnitude (if/when mags available) to avoid bias toward bright objects."
    )
    print()
    print("2) Semi-supervised (labeled + unlabeled)")
    print("   - Labeled: same E-COSMOS spec_z subset (train/val/test as above).")
    print("   - Unlabeled options:")
    print(
        "     a) E-COSMOS only: all E-COSMOS rows ({:,}); same field as labels.".format(n_ecosmos)
    )
    print(
        "     b) All fields: full catalog ({:,}); DEEP2-3, XMM-LSS, ELAIS-N1 are unlabeled.".format(
            n_total
        )
    )
    print(
        "   - SSL strategy: train supervised loss on labeled split; add consistency/prior loss on unlabeled (e.g. pseudo-labels from teacher, or mean-teacher)."
    )
    print("   - Recommendation: start with (a) for same-domain SSL; add (b) for scale if needed.")
    print(
        "   - c) All bands with missing data: use outer join (full catalog), keep all band columns;"
    )
    print(
        "     do not drop rows with missing NIR. Pass per-sample observed-band mask; expect wider"
    )
    print("     redshift intervals when fewer bands are observed.")
    print()
    print("3) Splits to implement")
    print(
        "   - Create indices or parquet subsets: train_labeled, val_labeled, test_labeled (from E-COSMOS spec_z), plus unlabeled_E-COSMOS and optionally unlabeled_all."
    )
    print(
        "   - Hold-out test: fix test set (e.g. 10% of labeled) and do not use for model selection; report final metrics on test only."
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report on CLAUDS+spec-z catalog for photo-z (labels, completeness, splits)."
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
        help="Path to clauds_specz_catalog.parquet.",
    )
    args = parser.parse_args()
    run_report(args.catalog)


if __name__ == "__main__":
    main()
