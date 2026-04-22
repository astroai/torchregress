#!/usr/bin/env python3
"""Sweep YearPredictionMSD (or a local CSV) over labeled fraction of the train pool.

Uses the same SSL benchmark family as ``self_agreement_realdata_year`` but varies
how much of the **full train pool** is labeled vs unlabeled. Optional CatBoost
``RMSEWithUncertainty`` rows are included for SOTA-style tabular comparisons when
``catboost`` is installed.

Designed for large batch runs: set ``--batch-size`` and ``--dataloader-num-workers``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import self_agreement_realdata_year as year_mod  # noqa: E402


def _train_val_idx(n: int, seed: int, val_frac: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    val = perm[:n_val]
    train = perm[n_val:]
    if train.size == 0:
        train, val = perm[1:], perm[:1]
    return train, val


def _gaussian_nll(y: np.ndarray, mu: np.ndarray, var: np.ndarray) -> float:
    var = np.maximum(var, 1e-8)
    return float(0.5 * np.mean(np.log(2.0 * np.pi * var) + (y - mu) ** 2 / var))


def _maybe_catboost_metrics(
    split: year_mod.YearSplit,
    *,
    seed: int,
    iterations: int,
) -> dict[str, float] | None:
    if iterations <= 0:
        return None
    try:
        from catboost import CatBoostRegressor  # noqa: PLC0415
    except ImportError:
        return None

    x = split.x_labeled.cpu().numpy()
    y = split.y_labeled.cpu().numpy().reshape(-1)
    xt = split.x_test.cpu().numpy()
    yt = split.y_test.cpu().numpy().reshape(-1)
    tr_idx, va_idx = _train_val_idx(x.shape[0], seed)
    x_tr, y_tr = x[tr_idx], y[tr_idx]
    x_va, y_va = x[va_idx], y[va_idx]

    reg = CatBoostRegressor(
        loss_function="RMSEWithUncertainty",
        iterations=iterations,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        early_stopping_rounds=100,
    )
    reg.fit(x_tr, y_tr, eval_set=(x_va, y_va), use_best_model=True)
    pred = reg.predict(xt, prediction_type="RMSEWithUncertainty")
    mu = pred[:, 0].astype(np.float64)
    sig2 = pred[:, 1].astype(np.float64)
    return {
        "CatBoost_RMSE": float(np.sqrt(np.mean((yt.astype(np.float64) - mu) ** 2))),
        "CatBoost_NLL": _gaussian_nll(yt.astype(np.float64), mu, sig2),
        "CatBoost_best_iteration": float(reg.get_best_iteration() or iterations),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows to write")
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run_sweep(
    base_cfg: year_mod.YearRealDataConfig,
    *,
    seeds: list[int],
    label_percents: list[float],
    shift_modes: list[str],
    min_unlabeled: int,
    catboost_iterations: int,
    out_csv: Path,
    summary_json: Path | None,
    shard_id: int | None = None,
    shard_count: int | None = None,
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    total_outer = max(1, len(seeds) * len(shift_modes) * len(label_percents))
    if shard_count is not None:
        if shard_id is None or not (0 <= shard_id < shard_count):
            raise ValueError("shard_id is required and must satisfy 0 <= shard_id < shard_count")
        cells_in_shard = (total_outer + shard_count - 1 - shard_id) // shard_count
        print(
            "[sage_year_label_fraction_sweep] run_sweep "
            f"seeds={len(seeds)} shifts={len(shift_modes)} pcts={len(label_percents)} "
            f"outer_steps={total_outer} shard={shard_id}/{shard_count} (~{cells_in_shard} cells)",
            flush=True,
        )
    else:
        print(
            "[sage_year_label_fraction_sweep] run_sweep "
            f"seeds={len(seeds)} shifts={len(shift_modes)} pcts={len(label_percents)} "
            f"outer_steps={total_outer}",
            flush=True,
        )
    global_idx = 0
    shard_step = 0
    for seed in seeds:
        cfg = year_mod.YearRealDataConfig(**{**asdict(base_cfg), "seed": int(seed)})
        for shift in shift_modes:
            for pct in label_percents:
                if shard_count is not None:
                    if global_idx % shard_count != shard_id:
                        global_idx += 1
                        continue
                shard_step += 1
                global_idx += 1
                print(
                    f"[sage_year_label_fraction_sweep] ({shard_step}/{cells_in_shard if shard_count else total_outer}) "
                    f"[flat_idx={global_idx - 1}] seed={seed} shift={shift!r} label_pct={pct}",
                    flush=True,
                )
                split = year_mod.make_year_split_label_pool_fraction(
                    cfg,
                    label_pool_percent=float(pct),
                    shift_mode=str(shift),
                    min_unlabeled=int(min_unlabeled),
                )
                n_lab = int(split.x_labeled.shape[0])
                n_ul = int(split.x_unlabeled.shape[0])
                bench_rows = year_mod.run_benchmark_on_split(cfg, split)
                cat_extra = _maybe_catboost_metrics(
                    split,
                    seed=int(seed),
                    iterations=int(catboost_iterations),
                )
                for row in bench_rows:
                    extra: dict[str, Any] = {
                        "Seed": int(seed),
                        "ShiftMode": str(shift),
                        "LabelPoolPercent_requested": float(pct),
                        "LabelPoolPercent_effective": float(n_lab) / float(n_lab + n_ul) * 100.0,
                        "N_train_pool": int(n_lab + n_ul),
                        "N_labeled": n_lab,
                        "N_unlabeled": n_ul,
                    }
                    if cat_extra:
                        extra.update(cat_extra)
                    rows_out.append({**extra, **row})
                print(
                    f"[sage_year_label_fraction_sweep]   finished bench rows={len(bench_rows)} "
                    f"(cumulative csv rows={len(rows_out)})",
                    flush=True,
                )

    _write_csv(out_csv, rows_out)
    if summary_json is not None:
        payload: dict[str, Any] = {
            "example": "examples/benchmarks/sage_year_label_fraction_sweep.py",
            "out_csv": str(out_csv),
            "base_config": asdict(base_cfg),
            "seeds": seeds,
            "label_percents": label_percents,
            "shift_modes": shift_modes,
            "min_unlabeled": min_unlabeled,
            "catboost_iterations": catboost_iterations,
            "n_rows": len(rows_out),
        }
        if shard_count is not None and shard_id is not None:
            payload["shard_id"] = int(shard_id)
            payload["shard_count"] = int(shard_count)
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return rows_out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--summary-json", type=Path, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[260408])
    p.add_argument(
        "--label-percents",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0],
    )
    p.add_argument(
        "--shift-modes",
        type=str,
        nargs="+",
        default=["none"],
        help="none | covariate | label (aliases for *_high_labeled modes).",
    )
    p.add_argument("--min-unlabeled", type=int, default=2048)
    p.add_argument("--dataset-path", type=str, default="")
    p.add_argument("--cache-path", type=str, default="")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--max-dataset-rows", type=int, default=None)
    p.add_argument("--n-test", type=int, default=year_mod.YearRealDataConfig.n_test)
    p.add_argument("--batch-size", type=int, default=year_mod.YearRealDataConfig.batch_size)
    p.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=year_mod.YearRealDataConfig.dataloader_num_workers,
    )
    p.add_argument("--teacher-epochs", type=int, default=year_mod.YearRealDataConfig.teacher_epochs)
    p.add_argument("--student-epochs", type=int, default=year_mod.YearRealDataConfig.student_epochs)
    p.add_argument("--hidden", type=int, default=year_mod.YearRealDataConfig.hidden)
    p.add_argument("--lr", type=float, default=year_mod.YearRealDataConfig.lr)
    p.add_argument("--weight-decay", type=float, default=year_mod.YearRealDataConfig.weight_decay)
    p.add_argument(
        "--unlabeled-fractions",
        type=float,
        nargs="+",
        default=list(year_mod.YearRealDataConfig.unlabeled_fractions),
    )
    p.add_argument(
        "--catboost-iterations",
        type=int,
        default=0,
        help="If >0, append CatBoost RMSEWithUncertainty metrics per row when catboost is installed.",
    )
    p.add_argument("--openml-data-id", type=int, default=None)
    p.add_argument("--openml-dataset-name", type=str, default="")
    p.add_argument("--openml-version", type=int, default=year_mod.YearRealDataConfig.openml_version)
    p.add_argument(
        "--shard-id",
        type=int,
        default=None,
        help="0-based shard index; must be used with --shard-count. Cells run where flat_idx %% shard_count == shard_id.",
    )
    p.add_argument(
        "--shard-count",
        type=int,
        default=None,
        help="Number of shards (e.g. 40 for CANFAR jobs). Must be used with --shard-id.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if (args.shard_id is None) ^ (args.shard_count is None):
        raise SystemExit("Provide both --shard-id and --shard-count, or neither.")
    if args.shard_count is not None and args.shard_count < 1:
        raise SystemExit("--shard-count must be >= 1")
    if args.shard_id is not None and not (0 <= args.shard_id < args.shard_count):
        raise SystemExit("--shard-id must satisfy 0 <= shard-id < shard-count")
    base_cfg = year_mod.YearRealDataConfig(
        dataset_path=args.dataset_path or None,
        cache_path=args.cache_path or None,
        allow_download=not args.no_download,
        max_dataset_rows=args.max_dataset_rows,
        n_test=args.n_test,
        n_labeled=year_mod.YearRealDataConfig.n_labeled,
        n_unlabeled=year_mod.YearRealDataConfig.n_unlabeled,
        batch_size=args.batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        teacher_epochs=args.teacher_epochs,
        student_epochs=args.student_epochs,
        hidden=args.hidden,
        lr=args.lr,
        weight_decay=args.weight_decay,
        unlabeled_fractions=tuple(args.unlabeled_fractions),
        openml_data_id=args.openml_data_id,
        openml_dataset_name=args.openml_dataset_name or None,
        openml_version=int(args.openml_version),
    )
    rows = run_sweep(
        base_cfg,
        seeds=list(args.seeds),
        label_percents=list(args.label_percents),
        shift_modes=list(args.shift_modes),
        min_unlabeled=int(args.min_unlabeled),
        catboost_iterations=int(args.catboost_iterations),
        out_csv=args.out_csv,
        summary_json=args.summary_json,
        shard_id=args.shard_id,
        shard_count=args.shard_count,
    )
    print(
        f"[sage_year_label_fraction_sweep] main() complete: {len(rows)} rows -> {args.out_csv}",
        flush=True,
    )
    if args.summary_json:
        print(f"Wrote {args.summary_json}", flush=True)


if __name__ == "__main__":
    main()
