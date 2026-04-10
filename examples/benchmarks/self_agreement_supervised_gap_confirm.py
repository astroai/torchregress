"""Confirm the best SAGE-Reg supervised-gap settings from a tuning sweep."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import self_agreement_higgs_ood as higgs_benchmark  # noqa: E402
import self_agreement_realdata_year as year_benchmark  # noqa: E402


@dataclass(frozen=True)
class SupervisedGapConfirmConfig:
    tuning_csv_path: str
    out_dir: str = str(
        Path("docs/research/sage_reg_results") / str(date.today()) / "supervised_gap_confirm"
    )
    seed: int | None = None
    year_dataset_path: str | None = None
    year_cache_path: str | None = None
    year_allow_download: bool | None = None
    higgs_dataset_path: str | None = None
    include_year: bool = True
    include_higgs: bool = True
    year_n_labeled: int | None = None
    year_n_unlabeled: int | None = None
    year_n_test: int | None = None
    year_hidden: int | None = None
    year_batch_size: int | None = None
    year_teacher_epochs: int | None = None
    year_student_epochs: int | None = None
    higgs_n_train: int | None = None
    higgs_n_unlabeled_id: int | None = None
    higgs_n_unlabeled_ood: int | None = None
    higgs_n_id_test: int | None = None
    higgs_n_ood_test: int | None = None
    higgs_hidden: int | None = None
    higgs_batch_size: int | None = None
    higgs_teacher_epochs: int | None = None
    higgs_student_epochs: int | None = None


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"tuning CSV not found: {resolved}. Run the tuning sweep first, "
            "or point --tuning-csv at an existing completed sweep file."
        )
    with resolved.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _best_row(rows: list[dict[str, str]], benchmark: str) -> dict[str, str]:
    matches = [row for row in rows if row.get("Benchmark") == benchmark]
    if not matches:
        raise ValueError(f"benchmark {benchmark!r} not found in tuning CSV")
    return min(matches, key=lambda row: float(row["SAGEMinusSupervised"]))


def _parse_optional_threshold(value: str) -> float | None:
    parsed = float(value)
    return None if parsed < 0.0 else parsed


def _load_tuning_config(path: str | Path) -> dict[str, Any]:
    csv_path = Path(path)
    candidates = [csv_path.with_suffix(".json"), csv_path.parent / "sweep.json"]
    for candidate in candidates:
        if candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            config = payload.get("config", {})
            if isinstance(config, dict):
                return config
    return {}


def _resolve_value(
    explicit: Any,
    tuning_config: dict[str, Any],
    key: str,
    fallback: Any,
) -> Any:
    if explicit is not None:
        return explicit
    if key in tuning_config:
        return tuning_config[key]
    return fallback


def _confirm_year(
    cfg: SupervisedGapConfirmConfig,
    row: dict[str, str],
    out_dir: Path,
    tuning_config: dict[str, Any],
) -> dict[str, Any]:
    bench_cfg = year_benchmark.YearRealDataConfig(
        seed=int(
            _resolve_value(cfg.seed, tuning_config, "seed", year_benchmark.YearRealDataConfig.seed)
        ),
        dataset_path=_resolve_value(
            cfg.year_dataset_path,
            tuning_config,
            "year_dataset_path",
            year_benchmark.YearRealDataConfig.dataset_path,
        ),
        cache_path=_resolve_value(
            cfg.year_cache_path,
            tuning_config,
            "year_cache_path",
            year_benchmark.YearRealDataConfig.cache_path,
        ),
        allow_download=bool(
            _resolve_value(
                cfg.year_allow_download,
                tuning_config,
                "year_allow_download",
                year_benchmark.YearRealDataConfig.allow_download,
            )
        ),
        n_labeled=int(
            _resolve_value(
                cfg.year_n_labeled,
                tuning_config,
                "year_n_labeled",
                year_benchmark.YearRealDataConfig.n_labeled,
            )
        ),
        n_unlabeled=int(
            _resolve_value(
                cfg.year_n_unlabeled,
                tuning_config,
                "year_n_unlabeled",
                year_benchmark.YearRealDataConfig.n_unlabeled,
            )
        ),
        n_test=int(
            _resolve_value(
                cfg.year_n_test,
                tuning_config,
                "year_n_test",
                year_benchmark.YearRealDataConfig.n_test,
            )
        ),
        hidden=int(
            _resolve_value(
                cfg.year_hidden,
                tuning_config,
                "year_hidden",
                year_benchmark.YearRealDataConfig.hidden,
            )
        ),
        batch_size=int(
            _resolve_value(
                cfg.year_batch_size,
                tuning_config,
                "year_batch_size",
                year_benchmark.YearRealDataConfig.batch_size,
            )
        ),
        teacher_epochs=int(
            _resolve_value(
                cfg.year_teacher_epochs,
                tuning_config,
                "year_teacher_epochs",
                year_benchmark.YearRealDataConfig.teacher_epochs,
            )
        ),
        student_epochs=int(
            _resolve_value(
                cfg.year_student_epochs,
                tuning_config,
                "year_student_epochs",
                year_benchmark.YearRealDataConfig.student_epochs,
            )
        ),
        tau=float(row["tau"]),
        unlabeled_noise=float(row["unlabeled_noise"]),
        feature_drop_prob=float(row["feature_drop_prob"]),
        pseudo_weight=float(row["pseudo_weight"]),
        agreement_weight=float(row["agreement_weight"]),
        weight_power=float(row["weight_power"]),
        hard_weight_threshold=_parse_optional_threshold(row["hard_weight_threshold"]),
        unlabeled_fractions=(1.0,),
    )
    result_rows = year_benchmark.main(
        bench_cfg,
        output_csv=str(out_dir / "year_confirm.csv"),
        performance_figure_path=str(out_dir / "year_confirm_perf.png"),
        calibration_figure_path=str(out_dir / "year_confirm_calib.png"),
        summary_json_path=str(out_dir / "year_confirm_summary.json"),
    )
    return {
        "benchmark": "year",
        "objective_metric": row.get("ObjectiveMetric"),
        "extra_metric": row.get("ExtraMetric"),
        "selected_row": row,
        "config": asdict(bench_cfg),
        "rows": result_rows,
    }


def _confirm_higgs(
    cfg: SupervisedGapConfirmConfig,
    row: dict[str, str],
    out_dir: Path,
    tuning_config: dict[str, Any],
) -> dict[str, Any]:
    dataset_path = _resolve_value(
        cfg.higgs_dataset_path,
        tuning_config,
        "higgs_dataset_path",
        None,
    )
    if dataset_path is None:
        raise ValueError("higgs_dataset_path is required to confirm the Higgs benchmark")
    bench_cfg = higgs_benchmark.HiggsOODConfig(
        seed=int(
            _resolve_value(cfg.seed, tuning_config, "seed", higgs_benchmark.HiggsOODConfig.seed)
        ),
        dataset_path=dataset_path,
        target_column="labels",
        ood_score_column="PRI_met",
        drop_columns=("weights", "detailed_labels"),
        n_train=int(
            _resolve_value(
                cfg.higgs_n_train,
                tuning_config,
                "higgs_n_train",
                higgs_benchmark.HiggsOODConfig.n_train,
            )
        ),
        n_unlabeled_id=int(
            _resolve_value(
                cfg.higgs_n_unlabeled_id,
                tuning_config,
                "higgs_n_unlabeled_id",
                higgs_benchmark.HiggsOODConfig.n_unlabeled_id,
            )
        ),
        n_unlabeled_ood=int(
            _resolve_value(
                cfg.higgs_n_unlabeled_ood,
                tuning_config,
                "higgs_n_unlabeled_ood",
                higgs_benchmark.HiggsOODConfig.n_unlabeled_ood,
            )
        ),
        n_id_test=int(
            _resolve_value(
                cfg.higgs_n_id_test,
                tuning_config,
                "higgs_n_id_test",
                higgs_benchmark.HiggsOODConfig.n_id_test,
            )
        ),
        n_ood_test=int(
            _resolve_value(
                cfg.higgs_n_ood_test,
                tuning_config,
                "higgs_n_ood_test",
                higgs_benchmark.HiggsOODConfig.n_ood_test,
            )
        ),
        hidden=int(
            _resolve_value(
                cfg.higgs_hidden,
                tuning_config,
                "higgs_hidden",
                higgs_benchmark.HiggsOODConfig.hidden,
            )
        ),
        batch_size=int(
            _resolve_value(
                cfg.higgs_batch_size,
                tuning_config,
                "higgs_batch_size",
                higgs_benchmark.HiggsOODConfig.batch_size,
            )
        ),
        teacher_epochs=int(
            _resolve_value(
                cfg.higgs_teacher_epochs,
                tuning_config,
                "higgs_teacher_epochs",
                higgs_benchmark.HiggsOODConfig.teacher_epochs,
            )
        ),
        student_epochs=int(
            _resolve_value(
                cfg.higgs_student_epochs,
                tuning_config,
                "higgs_student_epochs",
                higgs_benchmark.HiggsOODConfig.student_epochs,
            )
        ),
        tau=float(row["tau"]),
        unlabeled_noise=float(row["unlabeled_noise"]),
        feature_drop_prob=float(row["feature_drop_prob"]),
        pseudo_weight=float(row["pseudo_weight"]),
        agreement_weight=float(row["agreement_weight"]),
        weight_power=float(row["weight_power"]),
        hard_weight_threshold=_parse_optional_threshold(row["hard_weight_threshold"]),
    )
    result_rows = higgs_benchmark.main(
        bench_cfg,
        output_csv=str(out_dir / "higgs_confirm.csv"),
        performance_figure_path=str(out_dir / "higgs_confirm_perf.png"),
        calibration_figure_path=str(out_dir / "higgs_confirm_calib.png"),
        summary_json_path=str(out_dir / "higgs_confirm_summary.json"),
    )
    return {
        "benchmark": "higgs_public",
        "objective_metric": row.get("ObjectiveMetric"),
        "extra_metric": row.get("ExtraMetric"),
        "selected_row": row,
        "config": asdict(bench_cfg),
        "rows": result_rows,
    }


def main(cfg: SupervisedGapConfirmConfig) -> dict[str, Any]:
    rows = _read_rows(cfg.tuning_csv_path)
    tuning_config = _load_tuning_config(cfg.tuning_csv_path)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "tuning_csv_path": cfg.tuning_csv_path,
        "tuning_config": tuning_config,
        "runs": [],
    }
    if cfg.include_year:
        payload["runs"].append(_confirm_year(cfg, _best_row(rows, "year"), out_dir, tuning_config))
    if cfg.include_higgs:
        payload["runs"].append(
            _confirm_higgs(cfg, _best_row(rows, "higgs_public"), out_dir, tuning_config)
        )

    summary_path = out_dir / "selected_configs.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote selected config summary: {summary_path}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Confirm the best SAGE-Reg settings from a supervised-gap tuning sweep."
    )
    parser.add_argument("--tuning-csv", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=SupervisedGapConfirmConfig.out_dir)
    parser.add_argument("--year-dataset-path", type=str, default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--year-cache-path", type=str, default="")
    parser.add_argument("--no-year-download", action="store_true")
    parser.add_argument("--higgs-dataset-path", type=str, default="")
    parser.add_argument("--skip-year", action="store_true")
    parser.add_argument("--skip-higgs", action="store_true")
    parser.add_argument("--year-n-labeled", type=int, default=None)
    parser.add_argument("--year-n-unlabeled", type=int, default=None)
    parser.add_argument("--year-n-test", type=int, default=None)
    parser.add_argument("--year-hidden", type=int, default=None)
    parser.add_argument("--year-batch-size", type=int, default=None)
    parser.add_argument("--year-teacher-epochs", type=int, default=None)
    parser.add_argument("--year-student-epochs", type=int, default=None)
    parser.add_argument("--higgs-n-train", type=int, default=None)
    parser.add_argument(
        "--higgs-n-unlabeled-id",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--higgs-n-unlabeled-ood",
        type=int,
        default=None,
    )
    parser.add_argument("--higgs-n-id-test", type=int, default=None)
    parser.add_argument("--higgs-n-ood-test", type=int, default=None)
    parser.add_argument("--higgs-hidden", type=int, default=None)
    parser.add_argument("--higgs-batch-size", type=int, default=None)
    parser.add_argument("--higgs-teacher-epochs", type=int, default=None)
    parser.add_argument("--higgs-student-epochs", type=int, default=None)
    args = parser.parse_args()
    main(
        SupervisedGapConfirmConfig(
            tuning_csv_path=args.tuning_csv,
            out_dir=args.out_dir,
            seed=args.seed,
            year_dataset_path=args.year_dataset_path or None,
            year_cache_path=args.year_cache_path or None,
            year_allow_download=False if args.no_year_download else None,
            higgs_dataset_path=args.higgs_dataset_path or None,
            include_year=not args.skip_year,
            include_higgs=not args.skip_higgs,
            year_n_labeled=args.year_n_labeled,
            year_n_unlabeled=args.year_n_unlabeled,
            year_n_test=args.year_n_test,
            year_hidden=args.year_hidden,
            year_batch_size=args.year_batch_size,
            year_teacher_epochs=args.year_teacher_epochs,
            year_student_epochs=args.year_student_epochs,
            higgs_n_train=args.higgs_n_train,
            higgs_n_unlabeled_id=args.higgs_n_unlabeled_id,
            higgs_n_unlabeled_ood=args.higgs_n_unlabeled_ood,
            higgs_n_id_test=args.higgs_n_id_test,
            higgs_n_ood_test=args.higgs_n_ood_test,
            higgs_hidden=args.higgs_hidden,
            higgs_batch_size=args.higgs_batch_size,
            higgs_teacher_epochs=args.higgs_teacher_epochs,
            higgs_student_epochs=args.higgs_student_epochs,
        )
    )
