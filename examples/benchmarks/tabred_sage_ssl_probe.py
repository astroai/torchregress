"""TabReD (Rubachev et al.) × SAGE-Reg SSL probe — structured outputs for paper triage.

Loads **preprocessed TabReD** tensors produced by `yandex-research/tabred` preprocessing
(``save_dataset``: ``X_*.npy``, ``Y.npy``, ``split-<name>/{train,val,test}_idx.npy``).

Wide ``X_*`` blocks are assembled with **Polars** (``pl.concat(..., how="horizontal")``), matching
upstream TabReD tooling. Requires **polars** (install ``torchregress[tabred]`` or ``uv pip install polars``).

Default local layout (materialized by ``tools/fetch_tabred_data.py`` or the morning shell script)::

    data/tabred/
      cooking-time/
      delivery-eta/
      maps-routing/
      .vendor/yandex-tabred/   # upstream clone (gitignored under data/)

Morning run (from repo root) — clones TabReD, patches ``DATA_DIR``, Kaggle-downloads, preprocesses,
then runs the probe (needs ``~/.kaggle/kaggle.json`` and ``uv pip install -e '.[tabred]'`` for fetch + probe)::

    ./scripts/morning_tabred_bundle.sh

Fetch only (no SSL probe)::

    TABRED_FETCH_ONLY=1 ./scripts/morning_tabred_bundle.sh

Or probe only if data already exists::

    SKIP_TABRED_FETCH=1 TABRED_DATA_ROOT=data/tabred ./scripts/morning_tabred_bundle.sh

Direct probe (no fetch)::

    uv run python examples/benchmarks/tabred_sage_ssl_probe.py \\
      --tabred-data-root data/tabred \\
      --out-dir docs/research/sage_reg_results/$(date -u +%Y-%m-%d)/tabred_sage_bundle

Outputs (under ``--out-dir``)::

    bundle_summary.json   # all datasets + gap vs supervised + audit notes
    results_long.csv      # spreadsheet-friendly
    <dataset>/summary.json, rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import numpy as np
import polars as pl
import torch

BENCH_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = BENCH_DIR.parent
REPO_ROOT = BENCH_DIR.parent.parent
DEFAULT_TABRED_DATA_ROOT = REPO_ROOT / "data" / "tabred"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from comparison_utils import (  # noqa: E402
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
)
from self_agreement_realdata_year import (  # noqa: E402
    YearRealDataConfig,
    YearSplit,
    _run_fraction,
    _write_csv,
)

DEFAULT_DATASETS = ("cooking-time", "delivery-eta", "maps-routing")


@dataclass(frozen=True)
class TabRedProbeConfig:
    """Serializable knobs for bundle_summary.json."""

    tabred_data_root: str
    datasets: tuple[str, ...]
    split_name: str
    seed: int
    n_labeled: int
    n_unlabeled: int
    max_train_pool: int | None
    max_test_rows: int | None
    include_x_meta: bool
    teacher_epochs: int
    student_epochs: int
    hidden: int
    batch_size: int
    lr: float
    unlabeled_fractions: tuple[float, ...]
    quick: bool


def _load_tabred_arrays(
    dataset_dir: Path,
    *,
    include_x_meta: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X float32 [N,D], Y float32 [N,1]) via Polars (wide horizontal concat)."""
    blocks: list[pl.DataFrame] = []
    for key in ("X_num", "X_bin", "X_cat"):
        path = dataset_dir / f"{key}.npy"
        if path.exists():
            arr = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
            df = pl.DataFrame(arr)
            rename = {c: f"{key}_{c}" for c in df.columns}
            blocks.append(df.rename(rename))
    if include_x_meta:
        meta_p = dataset_dir / "X_meta.npy"
        if meta_p.exists():
            arr = np.asarray(np.load(meta_p, mmap_mode="r"), dtype=np.float32)
            df = pl.DataFrame(arr)
            blocks.append(df.rename({c: f"X_meta_{c}" for c in df.columns}))
    if not blocks:
        raise FileNotFoundError(
            f"No X_num/X_bin/X_cat in {dataset_dir} (TabReD preprocessing outputs missing?)"
        )
    xf = pl.concat(blocks, how="horizontal")
    x = xf.fill_nan(0.0).fill_null(0.0).to_numpy().astype(np.float32, copy=False)
    y_path = dataset_dir / "Y.npy"
    if not y_path.exists():
        raise FileNotFoundError(f"Missing Y.npy in {dataset_dir}")
    y_raw = np.asarray(np.load(y_path, mmap_mode="r"), dtype=np.float32).reshape(-1, 1)
    y = (
        pl.DataFrame({"y": y_raw.reshape(-1)})
        .fill_nan(0.0)
        .fill_null(0.0)
        .to_numpy()
        .astype(np.float32, copy=False)
        .reshape(-1, 1)
    )
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"X rows {x.shape[0]} != Y rows {y.shape[0]} in {dataset_dir}")
    return x, y


def _load_split_indices(dataset_dir: Path, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    split_dir = dataset_dir / f"split-{split_name}"
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing {split_dir} (try split name 'default')")
    train_idx = np.load(split_dir / "train_idx.npy")
    test_idx = np.load(split_dir / "test_idx.npy")
    return train_idx.reshape(-1), test_idx.reshape(-1)


def _subsample_idx(idxs: np.ndarray, max_n: int | None, seed: int) -> np.ndarray:
    if max_n is None or idxs.shape[0] <= max_n:
        return idxs
    rng = np.random.default_rng(seed)
    pick = rng.choice(idxs.shape[0], size=max_n, replace=False)
    return idxs[pick]


def _make_year_split_from_tabred(
    dataset_name: str,
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    seed: int,
    n_labeled: int,
    n_unlabeled: int,
    max_train_pool: int | None,
    max_test_rows: int | None,
) -> YearSplit:
    train_idx = _subsample_idx(train_idx, max_train_pool, seed + 11)
    test_idx = _subsample_idx(test_idx, max_test_rows, seed + 13)

    need = n_labeled + n_unlabeled
    if train_idx.shape[0] < need:
        raise ValueError(
            f"{dataset_name}: train pool has {train_idx.shape[0]} rows after caps; "
            f"need n_labeled + n_unlabeled = {need}. "
            "Raise --max-train-pool or lower labeled/unlabeled counts."
        )

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(train_idx.shape[0], generator=g).numpy()
    t_perm = train_idx[perm]
    labeled_pos = t_perm[:n_labeled]
    unlabeled_pos = t_perm[n_labeled : n_labeled + n_unlabeled]

    x_all = torch.tensor(x, dtype=torch.float32)
    y_all = torch.tensor(y, dtype=torch.float32)

    x_labeled = x_all[labeled_pos]
    y_labeled = y_all[labeled_pos]
    x_unlabeled = x_all[unlabeled_pos]
    y_unlabeled_true = y_all[unlabeled_pos]
    x_test = x_all[test_idx]
    y_test = y_all[test_idx]

    x_train_pool = torch.cat([x_labeled, x_unlabeled], dim=0)
    x_mean = x_train_pool.mean(dim=0, keepdim=True)
    x_std = x_train_pool.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_mean = y_labeled.mean(dim=0, keepdim=True)
    y_std = y_labeled.std(dim=0, keepdim=True).clamp_min(1e-6)

    return YearSplit(
        x_labeled=(x_labeled - x_mean) / x_std,
        y_labeled=(y_labeled - y_mean) / y_std,
        x_unlabeled=(x_unlabeled - x_mean) / x_std,
        y_unlabeled_true=(y_unlabeled_true - y_mean) / y_std,
        x_test=(x_test - x_mean) / x_std,
        y_test=(y_test - y_mean) / y_std,
        dataset_name=dataset_name,
        n_features=int(x.shape[1]),
    )


def _year_cfg_from_probe(pc: TabRedProbeConfig) -> YearRealDataConfig:
    return YearRealDataConfig(
        seed=pc.seed,
        n_labeled=pc.n_labeled,
        n_unlabeled=pc.n_unlabeled,
        n_test=1,  # unused when using custom YearSplit
        teacher_epochs=pc.teacher_epochs,
        student_epochs=pc.student_epochs,
        hidden=pc.hidden,
        batch_size=pc.batch_size,
        lr=pc.lr,
        unlabeled_fractions=pc.unlabeled_fractions,
        allow_download=False,
    )


def _gaps_vs_supervised(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_ds: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        ds = str(row.get("Dataset", ""))
        method = str(row.get("Method", ""))
        by_ds.setdefault(ds, {})[method] = row
    out: list[dict[str, Any]] = []
    for ds, methods in by_ds.items():
        sup = methods.get("SupervisedOnly")
        if not sup:
            continue
        for name, row in methods.items():
            if name == "SupervisedOnly":
                continue
            out.append(
                {
                    "dataset": ds,
                    "method": name,
                    "delta_RMSE": float(row["RMSE"]) - float(sup["RMSE"]),
                    "delta_NLL": float(row["NLL"]) - float(sup["NLL"]),
                    "delta_CRPS": float(row["CRPS"]) - float(sup["CRPS"]),
                    "delta_CoverageGap90": float(row["CoverageGap90"])
                    - float(sup["CoverageGap90"]),
                }
            )
    return out


def _utility_audit_notes(rows: list[dict[str, Any]], gaps: list[dict[str, Any]]) -> list[str]:
    notes: list[str] = []
    if not rows:
        return ["No rows produced — check data paths and split name."]
    sage_gaps = [g for g in gaps if g["method"] == "SAGE-Reg"]
    if sage_gaps:
        worse_rmse = sum(1 for g in sage_gaps if g["delta_RMSE"] > 0)
        if worse_rmse == len(sage_gaps):
            notes.append(
                "SAGE-Reg is worse than SupervisedOnly on RMSE for every dataset — "
                "consider τ / agreement_weight / batch-relative gating or a tuning pass."
            )
        elif worse_rmse == 0:
            notes.append("SAGE-Reg improves RMSE vs SupervisedOnly on all datasets in this bundle.")
    notes.append(
        "Metrics are on **normalized target** (y standardized using labeled train stats) — "
        "compare ranks across datasets, not raw numbers vs Year/Higgs."
    )
    notes.append(
        "For strict TabReD comparability, prefer `split-default` (time-based) from upstream preprocessing."
    )
    return notes


def run_bundle(
    pc: TabRedProbeConfig, out_dir: Path
) -> tuple[list[dict[str, Any]], Path, list[str]]:
    root = Path(pc.tabred_data_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_comparison_seed(pc.seed)

    all_rows: list[dict[str, Any]] = []
    cfg = _year_cfg_from_probe(pc)

    for name in pc.datasets:
        dpath = root / name
        if not dpath.is_dir():
            print(f"[skip] missing dataset directory: {dpath}", flush=True)
            continue
        info = dpath / "info.json"
        if not info.exists():
            print(f"[skip] {dpath} has no info.json (not TabReD preprocessed?)", flush=True)
            continue

        x, y = _load_tabred_arrays(dpath, include_x_meta=pc.include_x_meta)
        train_idx, test_idx = _load_split_indices(dpath, pc.split_name)
        split = _make_year_split_from_tabred(
            name,
            x,
            y,
            train_idx,
            test_idx,
            seed=pc.seed,
            n_labeled=pc.n_labeled,
            n_unlabeled=pc.n_unlabeled,
            max_train_pool=pc.max_train_pool,
            max_test_rows=pc.max_test_rows,
        )

        ds_rows: list[dict[str, Any]] = []
        for frac in pc.unlabeled_fractions:
            part = _run_fraction(cfg, split, float(frac))
            for row in part:
                row["TabReDSplit"] = pc.split_name
                row["N_train_pool_used"] = pc.n_labeled + pc.n_unlabeled
                row["N_test_used"] = int(split.x_test.shape[0])
            ds_rows.extend(part)
        all_rows.extend(ds_rows)

        ds_out = out_dir / name.replace("/", "_")
        ds_out.mkdir(parents=True, exist_ok=True)
        _write_csv(ds_out / "rows.csv", cast(list[dict[str, object]], ds_rows))
        with (ds_out / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": name,
                    "split": pc.split_name,
                    "n_features": split.n_features,
                    "rows": ds_rows,
                },
                f,
                indent=2,
            )

    gaps = _gaps_vs_supervised(all_rows)
    notes = _utility_audit_notes(all_rows, gaps)
    bundle_payload = {
        "artifact": "tabred_sage_bundle_summary",
        "version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(pc),
        "rows": all_rows,
        "gaps_vs_supervised": gaps,
        "utility_audit_notes": notes,
    }
    bundle_path = out_dir / "bundle_summary.json"
    bundle_path.write_text(json.dumps(bundle_payload, indent=2), encoding="utf-8")

    if all_rows:
        with (out_dir / "results_long.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)

    return all_rows, bundle_path, notes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TabReD × SAGE-Reg SSL probe with bundle JSON + CSV for triage."
    )
    parser.add_argument(
        "--tabred-data-root",
        type=str,
        default=os.environ.get("TABRED_DATA_ROOT", str(DEFAULT_TABRED_DATA_ROOT)),
        help="Parent of cooking-time/, delivery-eta/, maps-routing/ (default: <repo>/data/tabred).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for bundle_summary.json and per-dataset artifacts.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Subdirectory names under tabred-data-root.",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="default",
        help="Uses split-<name>/ from TabReD (time-based default split is usually 'default').",
    )
    parser.add_argument("--seed", type=int, default=260411)
    parser.add_argument("--n-labeled", type=int, default=4096)
    parser.add_argument("--n-unlabeled", type=int, default=65536)
    parser.add_argument(
        "--max-train-pool",
        type=int,
        default=None,
        help="Subsample TabReD train indices to this many rows before labeled/unlabeled split.",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=None,
        help="Subsample test indices (cap evaluation cost on huge tests).",
    )
    parser.add_argument(
        "--include-x-meta",
        action="store_true",
        help="Concatenate X_meta.npy if present (timestamps/ids — usually omit).",
    )
    parser.add_argument("--teacher-epochs", type=int, default=24)
    parser.add_argument("--student-epochs", type=int, default=24)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument(
        "--unlabeled-fractions",
        type=float,
        nargs="+",
        default=[1.0],
        help="Fractions of unlabeled pool (same protocol as Year benchmark).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small budgets: 2048+8192 train pool, 4096 test cap, 8/8 epochs, hidden 64.",
    )
    args = parser.parse_args()

    tabred_root = Path(args.tabred_data_root).expanduser().resolve()
    if not tabred_root.is_dir():
        parser.error(
            f"TabReD data root is not a directory: {tabred_root}\n"
            "Run: uv run python tools/fetch_tabred_data.py --out-dir data/tabred\n"
            "or ./scripts/morning_tabred_bundle.sh (needs Kaggle credentials)."
        )

    if args.quick:
        n_lab, n_unl, max_pool, max_test, hid, te, se = 2048, 8192, 50_000, 4096, 64, 8, 8
    else:
        n_lab = args.n_labeled
        n_unl = args.n_unlabeled
        max_pool = args.max_train_pool
        max_test = args.max_test_rows
        hid = args.hidden
        te = args.teacher_epochs
        se = args.student_epochs

    pc = TabRedProbeConfig(
        tabred_data_root=str(tabred_root),
        datasets=tuple(args.datasets),
        split_name=args.split_name,
        seed=args.seed,
        n_labeled=n_lab,
        n_unlabeled=n_unl,
        max_train_pool=max_pool,
        max_test_rows=max_test,
        include_x_meta=bool(args.include_x_meta),
        teacher_epochs=te,
        student_epochs=se,
        hidden=hid,
        batch_size=args.batch_size,
        lr=args.lr,
        unlabeled_fractions=tuple(args.unlabeled_fractions),
        quick=bool(args.quick),
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    rows, bundle_path, audit_notes = run_bundle(pc, out_dir)

    print_fairness_notes(
        title="TabReD SAGE-Reg probe",
        seed_policy=f"seed={pc.seed}; TabReD split={pc.split_name}",
        train_budget=(
            f"Gaussian MLP hidden={pc.hidden}, teacher_epochs={pc.teacher_epochs}, "
            f"student_epochs={pc.student_epochs}, nl={pc.n_labeled}, nu={pc.n_unlabeled}"
        ),
        metric_policy="RMSE/NLL/CRPS/Cov90/CalibMAE on normalized target (labeled-train y stats)",
    )
    print_comparison_summary(
        "TabReD bundle (all datasets × methods × fractions)",
        rows,
        metric_order=[
            "Dataset",
            "UnlabeledFraction",
            "RMSE",
            "NLL",
            "CRPS",
            "Cov90",
            "CalibMAE",
            "MeanWeight",
            "MeanDisagreement",
            "train_s",
        ],
    )
    print(f"\nWrote bundle: {bundle_path}")
    for line in audit_notes:
        print(f"[audit] {line}")


if __name__ == "__main__":
    main()
