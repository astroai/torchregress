#!/usr/bin/env python3
"""Export YearPredictionMSD (or a local CSV) split as NumPy for external / official baselines.

Writes ``*.npy`` tensors and ``meta.json`` describing normalization so you can run
`pm25/semi-supervised-regression` (RankUp) or other code **on the exact same split**
as ``examples/benchmarks/self_agreement_realdata_year.py``, then merge metrics with
``tools/merge_ssl_official_metrics.py``.

Targets ``X_*`` / ``y_*`` are **already normalized** (same as in-repo training). If an
official implementation expects raw targets, denormalize using ``meta.json`` fields.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "examples" / "benchmarks"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import self_agreement_realdata_year as year_mod  # noqa: E402


def _write_np(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def _tensor_to_numpy_f32(t) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def export_split(
    cfg: year_mod.YearRealDataConfig,
    *,
    out_dir: Path,
    split_mode: str,
    label_pool_percent: float | None,
    shift_mode: str | None,
    min_unlabeled: int | None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[export_year_ssl_split_for_external] split_mode={split_mode!r} out_dir={out_dir}",
        flush=True,
    )
    mode = split_mode.strip().lower()
    if mode == "default":
        split = year_mod._make_split(cfg)
    elif mode == "label_pool_fraction":
        if label_pool_percent is None or shift_mode is None or min_unlabeled is None:
            raise ValueError(
                "label_pool_fraction requires --label-pool-percent, --shift-mode, --min-unlabeled"
            )
        split = year_mod.make_year_split_label_pool_fraction(
            cfg,
            label_pool_percent=float(label_pool_percent),
            shift_mode=str(shift_mode),
            min_unlabeled=int(min_unlabeled),
        )
    else:
        raise ValueError(f"unknown split_mode: {split_mode!r}")

    xl, yl = split.x_labeled, split.y_labeled
    xu, yu = split.x_unlabeled, split.y_unlabeled_true
    xt, yt = split.x_test, split.y_test

    _write_np(out_dir / "x_labeled.npy", _tensor_to_numpy_f32(xl))
    _write_np(out_dir / "y_labeled.npy", _tensor_to_numpy_f32(yl))
    _write_np(out_dir / "x_unlabeled.npy", _tensor_to_numpy_f32(xu))
    _write_np(out_dir / "y_unlabeled_true.npy", _tensor_to_numpy_f32(yu))
    _write_np(out_dir / "x_test.npy", _tensor_to_numpy_f32(xt))
    _write_np(out_dir / "y_test.npy", _tensor_to_numpy_f32(yt))

    meta = {
        "protocol": "torchregress-year-ssl-export-v1",
        "split_mode": mode,
        "label_pool_percent": label_pool_percent,
        "shift_mode": shift_mode,
        "min_unlabeled": min_unlabeled,
        "dataset_name": split.dataset_name,
        "n_features": int(split.n_features),
        "shapes": {
            "labeled": [int(xl.shape[0]), int(xl.shape[1])],
            "unlabeled": [int(xu.shape[0]), int(xu.shape[1])],
            "test": [int(xt.shape[0]), int(xt.shape[1])],
        },
        "normalization": {
            "x": "mean/std from labeled+unlabeled train pool (see self_agreement_realdata_year._year_split_from_indices)",
            "y": "mean/std from labeled targets only (same module)",
        },
        "torchregress_config": asdict(cfg),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"[export_year_ssl_split_for_external] wrote {out_dir} (npys + meta.json)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--split-mode", type=str, default="default", help="default | label_pool_fraction"
    )
    p.add_argument("--label-pool-percent", type=float, default=None)
    p.add_argument("--shift-mode", type=str, default="none")
    p.add_argument("--min-unlabeled", type=int, default=None)
    p.add_argument("--dataset-path", type=str, default="")
    p.add_argument("--cache-path", type=str, default="")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--seed", type=int, default=year_mod.YearRealDataConfig.seed)
    p.add_argument("--n-labeled", type=int, default=year_mod.YearRealDataConfig.n_labeled)
    p.add_argument("--n-unlabeled", type=int, default=year_mod.YearRealDataConfig.n_unlabeled)
    p.add_argument("--n-test", type=int, default=year_mod.YearRealDataConfig.n_test)
    p.add_argument("--max-dataset-rows", type=int, default=None)
    args = p.parse_args()

    cfg = year_mod.YearRealDataConfig(
        dataset_path=args.dataset_path or None,
        cache_path=args.cache_path or None,
        allow_download=not args.no_download,
        seed=int(args.seed),
        n_labeled=args.n_labeled,
        n_unlabeled=args.n_unlabeled,
        n_test=args.n_test,
        max_dataset_rows=args.max_dataset_rows,
    )
    export_split(
        cfg,
        out_dir=args.out_dir,
        split_mode=str(args.split_mode),
        label_pool_percent=args.label_pool_percent,
        shift_mode=args.shift_mode,
        min_unlabeled=args.min_unlabeled,
    )


if __name__ == "__main__":
    main()
