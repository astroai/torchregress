"""Run multi-seed confirmation for the best SAGE-Reg supervised-gap settings."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import self_agreement_supervised_gap_confirm as confirm  # noqa: E402


@dataclass(frozen=True)
class SupervisedGapMultiSeedConfig:
    tuning_csv_path: str
    out_dir: str = str(
        Path("docs/research/sage_reg_results") / str(date.today()) / "supervised_gap_multiseed"
    )
    seeds: tuple[int, ...] = (260410, 260411, 260412)
    year_dataset_path: str | None = None
    year_cache_path: str | None = None
    year_allow_download: bool | None = None
    higgs_dataset_path: str | None = None
    include_year: bool = True
    include_higgs: bool = True
    year_teacher_epochs: int | None = None
    year_student_epochs: int | None = None
    higgs_teacher_epochs: int | None = None
    higgs_student_epochs: int | None = None
    higgs_split_scale_factor: int = 1
    higgs_parquet_max_sample_rows: int | None = None
    higgs_parquet_full_read_row_limit: int | None = None
    year_n_labeled: int | None = None
    year_n_unlabeled: int | None = None
    year_n_test: int | None = None


def _metric_names(benchmark: str) -> tuple[str, str]:
    if benchmark == "year":
        return "NLL", "Cov90"
    if benchmark == "higgs_public":
        return "NLL_OOD", "Cov90_OOD"
    raise ValueError(f"unsupported benchmark: {benchmark}")


def _summarize_seed_run(
    benchmark: str,
    seed: int,
    rows: list[dict[str, Any]],
) -> dict[str, object]:
    objective_metric, extra_metric = _metric_names(benchmark)
    by_method = {str(row["Method"]): row for row in rows}
    supervised = by_method["SupervisedOnly"]
    confidence = by_method["ConfidenceWeightedPseudoLabel"]
    sage = by_method["SAGE-Reg"]
    row: dict[str, object] = {
        "Benchmark": benchmark,
        "Seed": seed,
        "ObjectiveMetric": objective_metric,
        "ExtraMetric": extra_metric,
        "SupervisedObjective": float(supervised[objective_metric]),
        "SAGEObjective": float(sage[objective_metric]),
        "ConfidenceObjective": float(confidence[objective_metric]),
        "SAGEMinusSupervised": float(sage[objective_metric]) - float(supervised[objective_metric]),
        "ConfidenceMinusSupervised": float(confidence[objective_metric])
        - float(supervised[objective_metric]),
        "SupervisedExtra": float(supervised[extra_metric]),
        "SAGEExtra": float(sage[extra_metric]),
        "ConfidenceExtra": float(confidence[extra_metric]),
        "SAGEMeanWeight": float(sage["MeanWeight"]),
        "SAGEMeanDisagreement": float(sage["MeanDisagreement"]),
    }
    if "MeanTeacher" in by_method:
        mt = by_method["MeanTeacher"]
        row["MeanTeacherObjective"] = float(mt[objective_metric])
        row["MeanTeacherMinusSupervised"] = float(mt[objective_metric]) - float(
            supervised[objective_metric]
        )
        row["MeanTeacherExtra"] = float(mt[extra_metric])
    if benchmark == "higgs_public":
        row["SupervisedNLL_ID"] = float(supervised["NLL_ID"])
        row["SAGENLL_ID"] = float(sage["NLL_ID"])
        row["SupervisedRMSE_OOD"] = float(supervised["RMSE_OOD"])
        row["SAGERMSE_OOD"] = float(sage["RMSE_OOD"])
    return row


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["Benchmark"]), []).append(row)

    aggregates: list[dict[str, object]] = []
    for benchmark, values in sorted(grouped.items()):
        sage_gaps = [float(v["SAGEMinusSupervised"]) for v in values]
        conf_gaps = [float(v["ConfidenceMinusSupervised"]) for v in values]
        sage_weights = [float(v["SAGEMeanWeight"]) for v in values]
        sage_disagreements = [float(v["SAGEMeanDisagreement"]) for v in values]
        aggregates.append(
            {
                "Benchmark": benchmark,
                "Seeds": len(values),
                "SAGEMinusSupervisedMean": mean(sage_gaps),
                "SAGEMinusSupervisedMedian": median(sage_gaps),
                "SAGEMinusSupervisedStd": pstdev(sage_gaps) if len(sage_gaps) > 1 else 0.0,
                "ConfidenceMinusSupervisedMean": mean(conf_gaps),
                "ConfidenceMinusSupervisedMedian": median(conf_gaps),
                "ConfidenceMinusSupervisedStd": pstdev(conf_gaps) if len(conf_gaps) > 1 else 0.0,
                "SAGEMeanWeightMean": mean(sage_weights),
                "SAGEMeanWeightStd": pstdev(sage_weights) if len(sage_weights) > 1 else 0.0,
                "SAGEMeanDisagreementMean": mean(sage_disagreements),
                "SAGEMeanDisagreementStd": (
                    pstdev(sage_disagreements) if len(sage_disagreements) > 1 else 0.0
                ),
            }
        )
    return aggregates


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("rows must not be empty")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def main(cfg: SupervisedGapMultiSeedConfig) -> dict[str, Any]:
    tuning_rows = confirm._read_rows(cfg.tuning_csv_path)
    tuning_config = confirm._load_tuning_config(cfg.tuning_csv_path)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_rows: list[dict[str, object]] = []
    benchmark_runs: list[dict[str, Any]] = []

    for seed in cfg.seeds:
        seed_dir = out_dir / f"seed_{seed}"
        seed_cfg = confirm.SupervisedGapConfirmConfig(
            tuning_csv_path=cfg.tuning_csv_path,
            out_dir=str(seed_dir),
            seed=seed,
            year_dataset_path=cfg.year_dataset_path,
            year_cache_path=cfg.year_cache_path,
            year_allow_download=cfg.year_allow_download,
            higgs_dataset_path=cfg.higgs_dataset_path,
            include_year=cfg.include_year,
            include_higgs=cfg.include_higgs,
            year_n_labeled=cfg.year_n_labeled,
            year_n_unlabeled=cfg.year_n_unlabeled,
            year_n_test=cfg.year_n_test,
            year_teacher_epochs=cfg.year_teacher_epochs,
            year_student_epochs=cfg.year_student_epochs,
            higgs_teacher_epochs=cfg.higgs_teacher_epochs,
            higgs_student_epochs=cfg.higgs_student_epochs,
            higgs_split_scale_factor=cfg.higgs_split_scale_factor,
            higgs_parquet_max_sample_rows=cfg.higgs_parquet_max_sample_rows,
            higgs_parquet_full_read_row_limit=cfg.higgs_parquet_full_read_row_limit,
        )
        if cfg.include_year:
            run = confirm._confirm_year(
                seed_cfg,
                confirm._best_row(tuning_rows, "year"),
                seed_dir,
                tuning_config,
            )
            benchmark_runs.append(run)
            seed_rows.append(_summarize_seed_run("year", seed, run["rows"]))
        if cfg.include_higgs:
            run = confirm._confirm_higgs(
                seed_cfg,
                confirm._best_row(tuning_rows, "higgs_public"),
                seed_dir,
                tuning_config,
            )
            benchmark_runs.append(run)
            seed_rows.append(_summarize_seed_run("higgs_public", seed, run["rows"]))

    aggregate_rows = _aggregate(seed_rows)
    seeds_csv = out_dir / "multiseed_rows.csv"
    aggregate_csv = out_dir / "multiseed_summary.csv"
    _write_csv(seeds_csv, seed_rows)
    _write_csv(aggregate_csv, aggregate_rows)

    payload = {
        "tuning_csv_path": cfg.tuning_csv_path,
        "tuning_config": tuning_config,
        "seeds": list(cfg.seeds),
        "seed_rows": seed_rows,
        "aggregate_rows": aggregate_rows,
        "runs": benchmark_runs,
    }
    summary_path = out_dir / "multiseed_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote per-seed CSV: {seeds_csv}")
    print(f"Wrote aggregate CSV: {aggregate_csv}")
    print(f"Wrote multiseed summary: {summary_path}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multi-seed confirmation for the best SAGE-Reg supervised-gap settings."
    )
    parser.add_argument("--tuning-csv", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=SupervisedGapMultiSeedConfig.out_dir)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(SupervisedGapMultiSeedConfig.seeds),
    )
    parser.add_argument("--year-dataset-path", type=str, default="")
    parser.add_argument("--year-cache-path", type=str, default="")
    parser.add_argument("--no-year-download", action="store_true")
    parser.add_argument(
        "--higgs-dataset-path",
        type=str,
        default="",
        help=(
            "Path to FAIR Higgs parquet (not the literal /path/to/... placeholder). "
            "See papers/neurips_sage_reg/reproducibility.md section ‘Higgs parquet’."
        ),
    )
    parser.add_argument("--skip-year", action="store_true")
    parser.add_argument("--skip-higgs", action="store_true")
    parser.add_argument("--year-teacher-epochs", type=int, default=None)
    parser.add_argument("--year-student-epochs", type=int, default=None)
    parser.add_argument("--higgs-teacher-epochs", type=int, default=None)
    parser.add_argument("--higgs-student-epochs", type=int, default=None)
    parser.add_argument(
        "--higgs-split-scale-factor",
        type=int,
        default=1,
        help="Multiply Higgs split sizes (same as supervised_gap_confirm).",
    )
    parser.add_argument(
        "--higgs-parquet-max-sample-rows",
        type=int,
        default=None,
        help="Optional parquet reservoir cap for Higgs parquet loads.",
    )
    parser.add_argument(
        "--higgs-parquet-full-read-row-limit",
        type=int,
        default=None,
        help="Optional override for full-parquet read safety threshold.",
    )
    parser.add_argument(
        "--year-n-labeled",
        type=int,
        default=None,
        help="Override Year labeled pool size for all seeds (semi-sup / label-budget axis).",
    )
    parser.add_argument(
        "--year-n-unlabeled",
        type=int,
        default=None,
        help="Override Year unlabeled pool size for all seeds.",
    )
    parser.add_argument(
        "--year-n-test",
        type=int,
        default=None,
        help="Override Year test set size for all seeds.",
    )
    args = parser.parse_args()
    main(
        SupervisedGapMultiSeedConfig(
            tuning_csv_path=args.tuning_csv,
            out_dir=args.out_dir,
            seeds=tuple(args.seeds),
            year_dataset_path=args.year_dataset_path or None,
            year_cache_path=args.year_cache_path or None,
            year_allow_download=False if args.no_year_download else None,
            higgs_dataset_path=args.higgs_dataset_path or None,
            include_year=not args.skip_year,
            include_higgs=not args.skip_higgs,
            year_n_labeled=args.year_n_labeled,
            year_n_unlabeled=args.year_n_unlabeled,
            year_n_test=args.year_n_test,
            year_teacher_epochs=args.year_teacher_epochs,
            year_student_epochs=args.year_student_epochs,
            higgs_teacher_epochs=args.higgs_teacher_epochs,
            higgs_student_epochs=args.higgs_student_epochs,
            higgs_split_scale_factor=args.higgs_split_scale_factor,
            higgs_parquet_max_sample_rows=args.higgs_parquet_max_sample_rows,
            higgs_parquet_full_read_row_limit=args.higgs_parquet_full_read_row_limit,
        )
    )
