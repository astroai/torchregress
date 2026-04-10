"""Tune SAGE-Reg against the supervised-only gap on real-data benchmarks."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from datetime import date
from itertools import product
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import matplotlib.pyplot as plt  # noqa: E402
import self_agreement_higgs_ood as higgs_benchmark  # noqa: E402
import self_agreement_realdata_year as year_benchmark  # noqa: E402
from comparison_utils import (  # noqa: E402
    print_comparison_summary,
    print_fairness_notes,
    write_comparison_summary_json,
)


@dataclass(frozen=True)
class SupervisedGapTuningConfig:
    seed: int = 260410
    out_dir: str = str(Path("docs/research/sage_reg_results") / str(date.today()) / "supervised_gap_tuning")
    year_dataset_path: str | None = None
    year_cache_path: str | None = None
    year_allow_download: bool = True
    higgs_dataset_path: str | None = None
    include_year: bool = True
    include_higgs: bool = True
    tau_values: tuple[float, ...] = (0.12, 0.18, 0.28)
    unlabeled_noise_values: tuple[float, ...] = (0.02, 0.05, 0.10)
    feature_drop_prob_values: tuple[float, ...] = (0.0, 0.1, 0.2)
    pseudo_weight_values: tuple[float, ...] = (0.4, 0.8)
    agreement_weight_values: tuple[float, ...] = (0.25, 0.5)
    weight_power_values: tuple[float, ...] = (1.0, 2.0)
    hard_weight_threshold_values: tuple[float | None, ...] = (None, 0.85)
    year_n_labeled: int = 4_096
    year_n_unlabeled: int = 65_536
    year_n_test: int = 32_768
    year_teacher_epochs: int = 16
    year_student_epochs: int = 16
    year_batch_size: int = 512
    year_hidden: int = 128
    year_unlabeled_fraction: float = 1.0
    higgs_n_train: int = 4_096
    higgs_n_unlabeled_id: int = 16_384
    higgs_n_unlabeled_ood: int = 16_384
    higgs_n_id_test: int = 8_192
    higgs_n_ood_test: int = 8_192
    higgs_teacher_epochs: int = 16
    higgs_student_epochs: int = 16
    higgs_batch_size: int = 512
    higgs_hidden: int = 128
    log_progress: bool = True


def _write_csv(path: str | Path, rows: list[dict[str, object]]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("rows must not be empty")
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def _maybe_float(value: str) -> object:
    try:
        return float(value)
    except ValueError:
        return value


def _read_csv_rows(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [{key: _maybe_float(value) for key, value in row.items()} for row in reader]


def _append_csv_row(path: str | Path, row: dict[str, object]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with out_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return out_path


def _row_key(
    benchmark: str,
    *,
    tau: float,
    unlabeled_noise: float,
    feature_drop_prob: float,
    pseudo_weight: float,
    agreement_weight: float,
    weight_power: float,
    hard_weight_threshold: float | None,
) -> tuple[object, ...]:
    return (
        benchmark,
        float(tau),
        float(unlabeled_noise),
        float(feature_drop_prob),
        float(pseudo_weight),
        float(agreement_weight),
        float(weight_power),
        None if hard_weight_threshold is None else float(hard_weight_threshold),
    )


def _completed_keys(rows: list[dict[str, object]]) -> set[tuple[object, ...]]:
    keys: set[tuple[object, ...]] = set()
    for row in rows:
        threshold = float(row["hard_weight_threshold"])
        keys.add(
            _row_key(
                str(row["Benchmark"]),
                tau=float(row["tau"]),
                unlabeled_noise=float(row["unlabeled_noise"]),
                feature_drop_prob=float(row["feature_drop_prob"]),
                pseudo_weight=float(row["pseudo_weight"]),
                agreement_weight=float(row["agreement_weight"]),
                weight_power=float(row["weight_power"]),
                hard_weight_threshold=None if threshold < 0.0 else threshold,
            )
        )
    return keys


def _maybe_log_progress(
    cfg: SupervisedGapTuningConfig,
    *,
    benchmark: str,
    index: int,
    total: int,
    tau: float,
    unlabeled_noise: float,
    feature_drop_prob: float,
    pseudo_weight: float,
    agreement_weight: float,
    weight_power: float,
    hard_weight_threshold: float | None,
) -> None:
    if not cfg.log_progress:
        return
    threshold_s = "none" if hard_weight_threshold is None else f"{hard_weight_threshold:.2f}"
    print(
        f"[{benchmark} {index}/{total}] "
        f"tau={tau:.2f} noise={unlabeled_noise:.2f} drop={feature_drop_prob:.2f} "
        f"pseudo={pseudo_weight:.2f} agree={agreement_weight:.2f} "
        f"power={weight_power:.2f} threshold={threshold_s}"
    )


def _plot_summary(path: str | Path, rows: list[dict[str, object]]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["Benchmark"]), []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for benchmark, values in grouped.items():
        x = range(len(values))
        axes[0].plot(
            x,
            [float(v["SAGEMinusSupervised"]) for v in values],
            marker="o",
            label=benchmark,
        )
        axes[1].plot(
            x,
            [float(v["ConfidenceMinusSupervised"]) for v in values],
            marker="o",
            label=benchmark,
        )
    axes[0].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axes[0].set_title("SAGE minus supervised objective gap")
    axes[0].set_ylabel("Lower is better")
    axes[1].set_title("Confidence minus supervised objective gap")
    axes[1].set_ylabel("Lower is better")
    axes[0].set_xlabel("Sweep config index")
    axes[1].set_xlabel("Sweep config index")
    axes[1].legend(loc="best", fontsize=8)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _summarize_run(
    *,
    benchmark: str,
    rows: list[dict[str, object]],
    objective_metric: str,
    extra_metric: str,
    tau: float,
    unlabeled_noise: float,
    feature_drop_prob: float,
    pseudo_weight: float,
    agreement_weight: float,
    weight_power: float,
    hard_weight_threshold: float | None,
) -> dict[str, object]:
    by_method = {str(row["Method"]): row for row in rows}
    supervised = by_method["SupervisedOnly"]
    confidence = by_method["ConfidenceWeightedPseudoLabel"]
    sage = by_method["SAGE-Reg"]
    return {
        "Benchmark": benchmark,
        "ObjectiveMetric": objective_metric,
        "ExtraMetric": extra_metric,
        "tau": tau,
        "unlabeled_noise": unlabeled_noise,
        "feature_drop_prob": feature_drop_prob,
        "pseudo_weight": pseudo_weight,
        "agreement_weight": agreement_weight,
        "weight_power": weight_power,
        "hard_weight_threshold": -1.0 if hard_weight_threshold is None else hard_weight_threshold,
        "SupervisedObjective": float(supervised[objective_metric]),
        "ConfidenceObjective": float(confidence[objective_metric]),
        "SAGEObjective": float(sage[objective_metric]),
        "SAGEMinusSupervised": float(sage[objective_metric]) - float(supervised[objective_metric]),
        "ConfidenceMinusSupervised": float(confidence[objective_metric])
        - float(supervised[objective_metric]),
        "SupervisedExtra": float(supervised[extra_metric]),
        "SAGEExtra": float(sage[extra_metric]),
        "ConfidenceExtra": float(confidence[extra_metric]),
        "SAGEMeanWeight": float(sage["MeanWeight"]),
        "SAGEMeanDisagreement": float(sage["MeanDisagreement"]),
    }


def _summarize_metrics(
    *,
    benchmark: str,
    objective_metric: str,
    extra_metric: str,
    tau: float,
    unlabeled_noise: float,
    feature_drop_prob: float,
    pseudo_weight: float,
    agreement_weight: float,
    weight_power: float,
    hard_weight_threshold: float | None,
    supervised: dict[str, float],
    confidence: dict[str, float],
    sage: dict[str, float],
) -> dict[str, object]:
    return {
        "Benchmark": benchmark,
        "ObjectiveMetric": objective_metric,
        "ExtraMetric": extra_metric,
        "tau": tau,
        "unlabeled_noise": unlabeled_noise,
        "feature_drop_prob": feature_drop_prob,
        "pseudo_weight": pseudo_weight,
        "agreement_weight": agreement_weight,
        "weight_power": weight_power,
        "hard_weight_threshold": -1.0 if hard_weight_threshold is None else hard_weight_threshold,
        "SupervisedObjective": float(supervised[objective_metric]),
        "ConfidenceObjective": float(confidence[objective_metric]),
        "SAGEObjective": float(sage[objective_metric]),
        "SAGEMinusSupervised": float(sage[objective_metric]) - float(supervised[objective_metric]),
        "ConfidenceMinusSupervised": float(confidence[objective_metric])
        - float(supervised[objective_metric]),
        "SupervisedExtra": float(supervised[extra_metric]),
        "SAGEExtra": float(sage[extra_metric]),
        "ConfidenceExtra": float(confidence[extra_metric]),
        "SAGEMeanWeight": float(sage["MeanWeight"]),
        "SAGEMeanDisagreement": float(sage["MeanDisagreement"]),
    }


def _best_confidence_row(
    cache: dict[float, dict[str, float]],
    *,
    objective_metric: str,
) -> tuple[float, dict[str, float]]:
    best_pseudo_weight: float | None = None
    best_metrics: dict[str, float] | None = None
    for pseudo_weight, metrics in cache.items():
        if best_metrics is None or float(metrics[objective_metric]) < float(best_metrics[objective_metric]):
            best_pseudo_weight = pseudo_weight
            best_metrics = metrics
    if best_pseudo_weight is None or best_metrics is None:
        raise ValueError("confidence cache must not be empty")
    return best_pseudo_weight, best_metrics


def _prepare_year_context(cfg: SupervisedGapTuningConfig) -> dict[str, Any]:
    base_cfg = year_benchmark.YearRealDataConfig(
        seed=cfg.seed,
        dataset_path=cfg.year_dataset_path,
        cache_path=cfg.year_cache_path,
        allow_download=cfg.year_allow_download,
        n_labeled=cfg.year_n_labeled,
        n_unlabeled=cfg.year_n_unlabeled,
        n_test=cfg.year_n_test,
        hidden=cfg.year_hidden,
        teacher_epochs=cfg.year_teacher_epochs,
        student_epochs=cfg.year_student_epochs,
        batch_size=cfg.year_batch_size,
        unlabeled_fractions=(cfg.year_unlabeled_fraction,),
    )
    split = year_benchmark._make_split(base_cfg)
    x_unlabeled, _ = year_benchmark._subsample_pair(
        split.x_unlabeled, split.y_unlabeled_true, cfg.year_unlabeled_fraction
    )
    teacher = year_benchmark._train_supervised_teacher(
        base_cfg,
        split.x_labeled,
        split.y_labeled,
        input_dim=split.n_features,
    )
    supervised_metrics = {
        **year_benchmark._evaluate_model(teacher, split.x_test, split.y_test),
        "MeanWeight": 0.0,
        "MeanDisagreement": 0.0,
    }
    confidence_cache: dict[float, dict[str, float]] = {}
    for pseudo_weight in cfg.pseudo_weight_values:
        conf_cfg = year_benchmark.YearRealDataConfig(
            **{**base_cfg.__dict__, "pseudo_weight": pseudo_weight}
        )
        model, meta = year_benchmark._train_confidence_weighted_student(
            conf_cfg,
            teacher,
            split.x_labeled,
            split.y_labeled,
            x_unlabeled,
        )
        confidence_cache[float(pseudo_weight)] = {
            **year_benchmark._evaluate_model(model, split.x_test, split.y_test),
            "MeanWeight": float(meta["mean_weight"]),
            "MeanDisagreement": float(meta["mean_disagreement"]),
        }
    return {
        "base_cfg": base_cfg,
        "split": split,
        "x_unlabeled": x_unlabeled,
        "teacher": teacher,
        "supervised_metrics": supervised_metrics,
        "confidence_cache": confidence_cache,
    }


def _prepare_higgs_context(cfg: SupervisedGapTuningConfig) -> dict[str, Any]:
    base_cfg = higgs_benchmark.HiggsOODConfig(
        seed=cfg.seed,
        dataset_path=cfg.higgs_dataset_path,
        target_column="labels",
        ood_score_column="PRI_met",
        drop_columns=("weights", "detailed_labels"),
        n_train=cfg.higgs_n_train,
        n_unlabeled_id=cfg.higgs_n_unlabeled_id,
        n_unlabeled_ood=cfg.higgs_n_unlabeled_ood,
        n_id_test=cfg.higgs_n_id_test,
        n_ood_test=cfg.higgs_n_ood_test,
        hidden=cfg.higgs_hidden,
        teacher_epochs=cfg.higgs_teacher_epochs,
        student_epochs=cfg.higgs_student_epochs,
        batch_size=cfg.higgs_batch_size,
    )
    split = higgs_benchmark.make_split(base_cfg)
    teacher = higgs_benchmark._train_supervised_teacher(base_cfg, split)
    id_metrics = higgs_benchmark._evaluate_regime(teacher, split.x_id_test, split.y_id_test)
    ood_metrics = higgs_benchmark._evaluate_regime(teacher, split.x_ood_test, split.y_ood_test)
    supervised_metrics = {
        **{f"{k}_ID": float(v) for k, v in id_metrics.items() if k != "MeanStd"},
        **{f"{k}_OOD": float(v) for k, v in ood_metrics.items() if k != "MeanStd"},
        "OODUncGap": float(ood_metrics["MeanStd"] - id_metrics["MeanStd"]),
        "MeanWeight": 0.0,
        "MeanDisagreement": 0.0,
    }
    confidence_cache: dict[float, dict[str, float]] = {}
    for pseudo_weight in cfg.pseudo_weight_values:
        conf_cfg = higgs_benchmark.HiggsOODConfig(
            **{**base_cfg.__dict__, "pseudo_weight": pseudo_weight}
        )
        model, meta = higgs_benchmark._train_confidence_student(conf_cfg, split, teacher)
        id_metrics = higgs_benchmark._evaluate_regime(model, split.x_id_test, split.y_id_test)
        ood_metrics = higgs_benchmark._evaluate_regime(model, split.x_ood_test, split.y_ood_test)
        confidence_cache[float(pseudo_weight)] = {
            **{f"{k}_ID": float(v) for k, v in id_metrics.items() if k != "MeanStd"},
            **{f"{k}_OOD": float(v) for k, v in ood_metrics.items() if k != "MeanStd"},
            "OODUncGap": float(ood_metrics["MeanStd"] - id_metrics["MeanStd"]),
            "MeanWeight": float(meta["mean_weight"]),
            "MeanDisagreement": float(meta["mean_disagreement"]),
        }
    return {
        "base_cfg": base_cfg,
        "split": split,
        "teacher": teacher,
        "supervised_metrics": supervised_metrics,
        "confidence_cache": confidence_cache,
    }


def _run_year_sweep(
    cfg: SupervisedGapTuningConfig,
    *,
    rows: list[dict[str, object]],
    completed: set[tuple[object, ...]],
    output_csv: str | None,
) -> None:
    context = _prepare_year_context(cfg)
    best_conf_pseudo_weight, best_confidence = _best_confidence_row(
        context["confidence_cache"],
        objective_metric="NLL",
    )
    combos = list(
        product(
            cfg.tau_values,
            cfg.unlabeled_noise_values,
            cfg.feature_drop_prob_values,
            cfg.agreement_weight_values,
            cfg.weight_power_values,
            cfg.hard_weight_threshold_values,
        )
    )
    total = len(combos)
    for index, (
        tau,
        unlabeled_noise,
        feature_drop_prob,
        agreement_weight,
        weight_power,
        hard_weight_threshold,
    ) in enumerate(combos, start=1):
        key = _row_key(
            "year",
            tau=tau,
            unlabeled_noise=unlabeled_noise,
            feature_drop_prob=feature_drop_prob,
            pseudo_weight=best_conf_pseudo_weight,
            agreement_weight=agreement_weight,
            weight_power=weight_power,
            hard_weight_threshold=hard_weight_threshold,
        )
        if key in completed:
            continue
        _maybe_log_progress(
            cfg,
            benchmark="year",
            index=index,
            total=total,
            tau=tau,
            unlabeled_noise=unlabeled_noise,
            feature_drop_prob=feature_drop_prob,
            pseudo_weight=best_conf_pseudo_weight,
            agreement_weight=agreement_weight,
            weight_power=weight_power,
            hard_weight_threshold=hard_weight_threshold,
        )
        bench_cfg = year_benchmark.YearRealDataConfig(
            **{
                **context["base_cfg"].__dict__,
                "unlabeled_noise": unlabeled_noise,
                "feature_drop_prob": feature_drop_prob,
                "tau": tau,
                "agreement_weight": agreement_weight,
                "weight_power": weight_power,
                "hard_weight_threshold": hard_weight_threshold,
            }
        )
        model, meta = year_benchmark._train_sage_student(
            bench_cfg,
            context["teacher"],
            context["split"].x_labeled,
            context["split"].y_labeled,
            context["x_unlabeled"],
        )
        sage_metrics = {
            **year_benchmark._evaluate_model(model, context["split"].x_test, context["split"].y_test),
            "MeanWeight": float(meta["mean_weight"]),
            "MeanDisagreement": float(meta["mean_disagreement"]),
        }
        row = _summarize_metrics(
            benchmark="year",
            objective_metric="NLL",
            extra_metric="Cov90",
            tau=tau,
            unlabeled_noise=unlabeled_noise,
            feature_drop_prob=feature_drop_prob,
            pseudo_weight=best_conf_pseudo_weight,
            agreement_weight=agreement_weight,
            weight_power=weight_power,
            hard_weight_threshold=hard_weight_threshold,
            supervised=context["supervised_metrics"],
            confidence=best_confidence,
            sage=sage_metrics,
        )
        rows.append(row)
        completed.add(key)
        if output_csv is not None:
            _append_csv_row(output_csv, row)


def _run_higgs_sweep(
    cfg: SupervisedGapTuningConfig,
    *,
    rows: list[dict[str, object]],
    completed: set[tuple[object, ...]],
    output_csv: str | None,
) -> None:
    if not cfg.higgs_dataset_path:
        return
    context = _prepare_higgs_context(cfg)
    best_conf_pseudo_weight, best_confidence = _best_confidence_row(
        context["confidence_cache"],
        objective_metric="NLL_OOD",
    )
    combos = list(
        product(
            cfg.tau_values,
            cfg.unlabeled_noise_values,
            cfg.feature_drop_prob_values,
            cfg.agreement_weight_values,
            cfg.weight_power_values,
            cfg.hard_weight_threshold_values,
        )
    )
    total = len(combos)
    for index, (
        tau,
        unlabeled_noise,
        feature_drop_prob,
        agreement_weight,
        weight_power,
        hard_weight_threshold,
    ) in enumerate(combos, start=1):
        key = _row_key(
            "higgs_public",
            tau=tau,
            unlabeled_noise=unlabeled_noise,
            feature_drop_prob=feature_drop_prob,
            pseudo_weight=best_conf_pseudo_weight,
            agreement_weight=agreement_weight,
            weight_power=weight_power,
            hard_weight_threshold=hard_weight_threshold,
        )
        if key in completed:
            continue
        _maybe_log_progress(
            cfg,
            benchmark="higgs_public",
            index=index,
            total=total,
            tau=tau,
            unlabeled_noise=unlabeled_noise,
            feature_drop_prob=feature_drop_prob,
            pseudo_weight=best_conf_pseudo_weight,
            agreement_weight=agreement_weight,
            weight_power=weight_power,
            hard_weight_threshold=hard_weight_threshold,
        )
        bench_cfg = higgs_benchmark.HiggsOODConfig(
            **{
                **context["base_cfg"].__dict__,
                "unlabeled_noise": unlabeled_noise,
                "feature_drop_prob": feature_drop_prob,
                "tau": tau,
                "agreement_weight": agreement_weight,
                "weight_power": weight_power,
                "hard_weight_threshold": hard_weight_threshold,
            }
        )
        model, meta = higgs_benchmark._train_sage_student(
            bench_cfg,
            context["split"],
            context["teacher"],
        )
        id_metrics = higgs_benchmark._evaluate_regime(
            model, context["split"].x_id_test, context["split"].y_id_test
        )
        ood_metrics = higgs_benchmark._evaluate_regime(
            model, context["split"].x_ood_test, context["split"].y_ood_test
        )
        sage_metrics = {
            **{f"{k}_ID": float(v) for k, v in id_metrics.items() if k != "MeanStd"},
            **{f"{k}_OOD": float(v) for k, v in ood_metrics.items() if k != "MeanStd"},
            "OODUncGap": float(ood_metrics["MeanStd"] - id_metrics["MeanStd"]),
            "MeanWeight": float(meta["mean_weight"]),
            "MeanDisagreement": float(meta["mean_disagreement"]),
        }
        row = _summarize_metrics(
            benchmark="higgs_public",
            objective_metric="NLL_OOD",
            extra_metric="Cov90_OOD",
            tau=tau,
            unlabeled_noise=unlabeled_noise,
            feature_drop_prob=feature_drop_prob,
            pseudo_weight=best_conf_pseudo_weight,
            agreement_weight=agreement_weight,
            weight_power=weight_power,
            hard_weight_threshold=hard_weight_threshold,
            supervised=context["supervised_metrics"],
            confidence=best_confidence,
            sage=sage_metrics,
        )
        rows.append(row)
        completed.add(key)
        if output_csv is not None:
            _append_csv_row(output_csv, row)


def run_tuning(
    cfg: SupervisedGapTuningConfig,
    *,
    existing_rows: list[dict[str, object]] | None = None,
    output_csv: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [] if existing_rows is None else list(existing_rows)
    completed = _completed_keys(rows)
    if cfg.include_year:
        _run_year_sweep(cfg, rows=rows, completed=completed, output_csv=output_csv)
    if cfg.include_higgs:
        _run_higgs_sweep(cfg, rows=rows, completed=completed, output_csv=output_csv)
    return rows


def _best_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best: list[dict[str, object]] = []
    seen: dict[str, dict[str, object]] = {}
    for row in rows:
        benchmark = str(row["Benchmark"])
        current = seen.get(benchmark)
        if current is None or float(row["SAGEMinusSupervised"]) < float(current["SAGEMinusSupervised"]):
            seen[benchmark] = row
    for key in sorted(seen):
        best.append({"Method": key, **seen[key]})
    return best


def main(
    cfg: SupervisedGapTuningConfig | None = None,
    *,
    output_csv: str | None = None,
    figure_path: str | None = None,
    summary_json_path: str | None = None,
) -> list[dict[str, object]]:
    resolved = SupervisedGapTuningConfig() if cfg is None else cfg
    out_dir = Path(resolved.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_output_csv = output_csv or str(out_dir / "sweep.csv")
    resolved_figure_path = figure_path or str(out_dir / "sweep.png")
    resolved_summary_json_path = summary_json_path or str(out_dir / "sweep.json")
    existing_rows: list[dict[str, object]] = []
    if Path(resolved_output_csv).exists():
        existing_rows = _read_csv_rows(resolved_output_csv)
        if resolved.log_progress:
            print(
                "Resuming from existing CSV with "
                f"{len(existing_rows)} completed rows: {resolved_output_csv}"
            )
    rows = run_tuning(resolved, existing_rows=existing_rows, output_csv=resolved_output_csv)
    print_fairness_notes(
        title="SAGE-Reg Supervised-Gap Tuning",
        seed_policy=f"single fixed seed ({resolved.seed}) across all sweeps",
        train_budget="shared teacher/student budgets within each benchmark sweep",
        metric_policy="objective gap versus SupervisedOnly on year (NLL) and Higgs-public (OOD NLL)",
    )
    print_comparison_summary(
        "Best SAGE-Reg configurations by benchmark",
        _best_rows(rows),
        metric_order=[
            "tau",
            "unlabeled_noise",
            "feature_drop_prob",
            "pseudo_weight",
            "agreement_weight",
            "weight_power",
            "hard_weight_threshold",
            "SAGEMinusSupervised",
            "ConfidenceMinusSupervised",
            "SAGEObjective",
            "SupervisedObjective",
            "SAGEMeanWeight",
            "SAGEMeanDisagreement",
        ],
    )
    if not Path(resolved_output_csv).exists():
        out = _write_csv(resolved_output_csv, rows)
        print(f"\nWrote CSV: {out}")
    else:
        print(f"\nWrote/updated CSV incrementally: {resolved_output_csv}")
    if resolved_figure_path:
        out = _plot_summary(resolved_figure_path, rows)
        print(f"Wrote tuning figure: {out}")
    if resolved_summary_json_path:
        out = write_comparison_summary_json(
            resolved_summary_json_path,
            example="examples/benchmarks/self_agreement_supervised_gap_tuning.py",
            task="tuning SAGE-Reg against the supervised-only gap on year and Higgs public data",
            config=resolved,
            rows=rows,
            notes=[
                "This sweep is explicitly about closing the gap to SupervisedOnly, not about proving broad SOTA gains.",
                "year uses NLL as the objective gap metric; Higgs public uses OOD NLL.",
                "Higgs public remains an OOD trust-weighting stress test rather than a clean regression benchmark.",
            ],
        )
        print(f"Wrote summary JSON: {out}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune SAGE-Reg against the supervised-only gap.")
    parser.add_argument("--year-dataset-path", type=str, default="")
    parser.add_argument("--year-cache-path", type=str, default="")
    parser.add_argument("--no-year-download", action="store_true")
    parser.add_argument("--higgs-dataset-path", type=str, default="")
    parser.add_argument("--out-dir", type=str, default=SupervisedGapTuningConfig.out_dir)
    parser.add_argument("--skip-year", action="store_true")
    parser.add_argument("--skip-higgs", action="store_true")
    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument("--figure-path", type=str, default="")
    parser.add_argument("--summary-json-path", type=str, default="")
    parser.add_argument("--year-teacher-epochs", type=int, default=SupervisedGapTuningConfig.year_teacher_epochs)
    parser.add_argument("--year-student-epochs", type=int, default=SupervisedGapTuningConfig.year_student_epochs)
    parser.add_argument("--higgs-teacher-epochs", type=int, default=SupervisedGapTuningConfig.higgs_teacher_epochs)
    parser.add_argument("--higgs-student-epochs", type=int, default=SupervisedGapTuningConfig.higgs_student_epochs)
    args = parser.parse_args()

    main(
        SupervisedGapTuningConfig(
            out_dir=args.out_dir,
            year_dataset_path=args.year_dataset_path or None,
            year_cache_path=args.year_cache_path or None,
            year_allow_download=not args.no_year_download,
            higgs_dataset_path=args.higgs_dataset_path or None,
            include_year=not args.skip_year,
            include_higgs=not args.skip_higgs,
            year_teacher_epochs=args.year_teacher_epochs,
            year_student_epochs=args.year_student_epochs,
            higgs_teacher_epochs=args.higgs_teacher_epochs,
            higgs_student_epochs=args.higgs_student_epochs,
        ),
        output_csv=args.output_csv or None,
        figure_path=args.figure_path or None,
        summary_json_path=args.summary_json_path or None,
    )
