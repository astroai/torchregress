#!/usr/bin/env python3
"""Launch the full supervised-gap hyperparameter grid (default SupervisedGapTuningConfig).

Year track uses paper-scale splits and 32/32 epochs unless overridden. Higgs runs only if
``--higgs-parquet`` points to an existing file (unless ``--force-higgs`` is set, which errors
if missing). Intended for long overnight runs; tuning resumes from partial ``sweep.csv``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "examples" / "benchmarks") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "examples" / "benchmarks"))

import self_agreement_supervised_gap_tuning as tuning  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for sweep.csv / sweep.json / sweep.png",
    )
    p.add_argument(
        "--year-cache",
        type=Path,
        required=True,
        help="Materialized OpenML Year CSV (offline; use tools/materialize_openml_year.py).",
    )
    p.add_argument(
        "--higgs-parquet",
        type=Path,
        default=REPO_ROOT
        / "docs/research/sage_reg_results/2026-04-09/higgs_public/extracted"
        / "FAIR_Universe_HiggsML_data.parquet",
    )
    p.add_argument(
        "--skip-higgs",
        action="store_true",
        help="Do not run the Higgs-public sweep even if parquet exists.",
    )
    p.add_argument(
        "--force-higgs",
        action="store_true",
        help="Require Higgs parquet to exist (exit non-zero if missing).",
    )
    p.add_argument("--year-n-labeled", type=int, default=4096)
    p.add_argument("--year-n-unlabeled", type=int, default=131_072)
    p.add_argument("--year-n-test", type=int, default=32_768)
    p.add_argument("--year-teacher-epochs", type=int, default=32)
    p.add_argument("--year-student-epochs", type=int, default=32)
    p.add_argument("--year-lr-schedule", type=str, default="cosine")
    p.add_argument("--year-lr-min", type=float, default=1e-5)
    p.add_argument(
        "--higgs-teacher-epochs",
        type=int,
        default=32,
        help="Must match multiseed confirmation when comparing tuning rows to confirms.",
    )
    p.add_argument("--higgs-student-epochs", type=int, default=32)
    p.add_argument("--seed", type=int, default=260410)
    args = p.parse_args()

    year_cache = args.year_cache.resolve()
    if not year_cache.is_file():
        raise SystemExit(f"Year cache not found: {year_cache}")

    higgs_path = args.higgs_parquet.resolve()
    include_higgs = not args.skip_higgs
    higgs_dataset: str | None = None
    if include_higgs:
        if not higgs_path.is_file():
            if args.force_higgs:
                raise SystemExit(f"Higgs parquet not found: {higgs_path}")
            print(f"Skipping Higgs sweep (parquet missing): {higgs_path}", flush=True)
            include_higgs = False
        else:
            higgs_dataset = str(higgs_path)

    base = tuning.SupervisedGapTuningConfig()
    cfg = replace(
        base,
        seed=args.seed,
        out_dir=str(args.out_dir.resolve()),
        year_cache_path=str(year_cache),
        year_allow_download=False,
        year_n_labeled=int(args.year_n_labeled),
        year_n_unlabeled=int(args.year_n_unlabeled),
        year_n_test=int(args.year_n_test),
        year_teacher_epochs=int(args.year_teacher_epochs),
        year_student_epochs=int(args.year_student_epochs),
        year_lr_schedule=str(args.year_lr_schedule),
        year_lr_min=float(args.year_lr_min),
        include_higgs=include_higgs,
        higgs_dataset_path=higgs_dataset,
        higgs_teacher_epochs=int(args.higgs_teacher_epochs),
        higgs_student_epochs=int(args.higgs_student_epochs),
        log_progress=True,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Full gap tuning: year grid={3 * 3 * 3 * 3 * 2 * 2 * 2} SAGE configs per benchmark, "
        f"include_higgs={include_higgs}, out_dir={cfg.out_dir}",
        flush=True,
    )
    tuning.main(cfg)


if __name__ == "__main__":
    main()
