#!/usr/bin/env python3
"""One-shot SAGE-Reg NeurIPS evidence: Year cache, tabular runs, sweeps, optional extras.

Covers optional TabReD, CatBoost, and Higgs when dependencies or assets exist.

Usage (repo root)::

    uv run python scripts/run_neurips_sage_reg_full.py
    uv run python scripts/run_neurips_sage_reg_full.py --quick

Defaults need no flags. Large raw assets (Higgs parquet) are skipped with logs if missing.

TabReD is optional in two modes:

- If ``~/.kaggle/kaggle.json`` exists, we run ``tools/fetch_tabred_data.py`` (with
  ``--skip-if-present``) and then the SSL probe.
- If Kaggle credentials are missing but ``<tabred-data-root>/<dataset>/info.json``
  exists for all default TabReD tasks, we **skip fetch** and still run the probe
  against the local materialized tensors.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Canonical caches / inputs (override with env or --run-root only changes output tree)
YEAR_CACHE_DEFAULT = REPO_ROOT / "data" / "paper" / "openml_year.csv"
OPENML_DIAMONDS_CACHE_DEFAULT = (
    REPO_ROOT / "data" / "paper" / "openml_large_tabular_diamonds.parquet"
)
TUNING_CSV_DEFAULT = (
    REPO_ROOT / "docs/research/sage_reg_results/2026-04-10/supervised_gap_tuning_v3/sweep.csv"
)
HIGGS_PARQUET_DEFAULT = (
    REPO_ROOT
    / "docs/research/sage_reg_results/2026-04-09/higgs_public/extracted"
    / "FAIR_Universe_HiggsML_data.parquet"
)
TABRED_ROOT_DEFAULT = REPO_ROOT / "data" / "tabred"
TABRED_DEFAULT_DATASETS = ("cooking-time", "delivery-eta", "maps-routing")
SHIFTS_OUT_ROOT_DEFAULT = REPO_ROOT / "data" / "shifts"
SHIFTS_DATASET_DEFAULT = "solar"

# Ten fixed seeds for variance reporting (extends prior six-seed protocol).
FULL_SEEDS = tuple(260410 + i for i in range(10))
QUICK_SEEDS = (260410, 260411)

# Keys for --only-phases / --skip-phases (must match manifest["phases"] / docs).
NEURIPS_PHASE_KEYS: frozenset[str] = frozenset(
    {
        "year_direct",
        "multiseed",
        "openml_diamonds_multiseed",
        "year_labeled_sweep",
        "multiseed_year_nl2048",
        "catboost",
        "tabred",
        "synthetic",
        "backbone",
        "ablations",
        "shifts",
        "image_rebuttal",
        "aggregate",
    }
)
NEURIPS_YEAR_CACHE_PHASES: frozenset[str] = frozenset(
    {
        "year_direct",
        "multiseed",
        "openml_diamonds_multiseed",
        "year_labeled_sweep",
        "multiseed_year_nl2048",
        "catboost",
        "ablations",
    }
)
NEURIPS_OPENML_CACHE_PHASES: frozenset[str] = frozenset({"openml_diamonds_multiseed"})


def _parse_phase_csv(value: str | None) -> frozenset[str] | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return frozenset(x.strip() for x in text.split(",") if x.strip())


def _active_phases(only: frozenset[str] | None, skip: frozenset[str]) -> frozenset[str]:
    """Phases that are eligible to run after applying only/skip."""
    if only is not None:
        return frozenset(p for p in only if p not in skip)
    return frozenset(p for p in NEURIPS_PHASE_KEYS if p not in skip)


def _phase_selected(phase: str, only: frozenset[str] | None, skip: frozenset[str]) -> bool:
    if only is not None and phase not in only:
        return False
    return phase not in skip


def _uv_run(parts: list[str], *, cwd: Path | None = None, check: bool = True) -> int:
    cmd = [sys.executable, *parts]
    print("==", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    r = subprocess.run(cmd, cwd=str(cwd or REPO_ROOT), env=env, check=False)
    status = int(r.returncode)
    print(f"==> subprocess exit {status}", flush=True)
    if check and status != 0:
        print(f"==> FAILED (raising): exit {status}\n    cmd: {cmd}", file=sys.stderr, flush=True)
        raise subprocess.CalledProcessError(status, cmd, None, None)
    return status


def _ensure_year_cache(cache: Path, *, allow_download: bool) -> None:
    if cache.is_file():
        return
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not allow_download:
        raise SystemExit(f"Year cache missing and --no-year-download: {cache}")
    _uv_run(
        [
            str(REPO_ROOT / "tools/materialize_openml_year.py"),
            "--cache-path",
            str(cache),
        ]
    )


def _ensure_openml_diamonds_cache(cache: Path, *, allow_download: bool) -> None:
    if cache.is_file():
        return
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not allow_download:
        print(f"Skip OpenML diamonds multiseed: cache missing {cache}", flush=True)
        return
    _uv_run(
        [
            str(REPO_ROOT / "tools/materialize_openml_large_tabular.py"),
            "--cache-path",
            str(cache),
            "--data-id",
            "42225",
        ]
    )


def _parquet_num_rows(path: Path) -> int:
    import pyarrow.parquet as pq  # noqa: PLC0415

    return int(pq.read_metadata(str(path)).num_rows)


def _scale_year_split_sizes_for_row_budget(
    *,
    n_labeled: int,
    n_unlabeled: int,
    n_test: int,
    n_rows: int,
) -> tuple[int, int, int]:
    """Shrink (nu, nt) proportionally if nl+nu+nt exceeds a fixed-row dataset."""
    nl = int(n_labeled)
    nu = int(n_unlabeled)
    nt = int(n_test)
    need = nl + nu + nt
    if need <= int(n_rows):
        return nl, nu, nt
    extra = int(n_rows) - nl
    if extra <= 0:
        raise ValueError(f"n_labeled={nl} exceeds dataset rows={n_rows}")
    denom = nu + nt
    if denom <= 0:
        raise ValueError("n_unlabeled and n_test must be positive for proportional shrink")
    nu2 = int(extra * (nu / denom))
    nt2 = extra - nu2
    if nu2 < 1 or nt2 < 1:
        raise ValueError(
            "Could not fit splits into "
            f"n_rows={n_rows} with nl={nl}, nu={nu}, nt={nt} -> {nu2}, {nt2}"
        )
    return nl, nu2, nt2


def _tabred_local_markers_present(tabred_root: Path) -> bool:
    """Return True if local TabReD preprocessing outputs look complete (info.json per dataset)."""
    root = tabred_root.expanduser().resolve()
    return all((root / name / "info.json").is_file() for name in TABRED_DEFAULT_DATASETS)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help=(
            "Output root (default: "
            "docs/research/sage_reg_results/<UTC-date>/neurips_sage_reg_full)."
        ),
    )
    parser.add_argument("--quick", action="store_true", help="Small budgets for CI smoke")
    parser.add_argument("--no-year-download", action="store_true")
    parser.add_argument(
        "--no-openml-download",
        action="store_true",
        help="Do not materialize missing OpenML large-tabular caches (e.g., diamonds parquet).",
    )
    parser.add_argument("--year-cache", type=Path, default=YEAR_CACHE_DEFAULT)
    parser.add_argument(
        "--openml-diamonds-cache",
        type=Path,
        default=OPENML_DIAMONDS_CACHE_DEFAULT,
        help="Pinned OpenML diamonds (id 42225) table cache path (.parquet or .csv).",
    )
    parser.add_argument("--tuning-csv", type=Path, default=TUNING_CSV_DEFAULT)
    parser.add_argument("--higgs-parquet", type=Path, default=HIGGS_PARQUET_DEFAULT)
    parser.add_argument("--skip-catboost", action="store_true")
    parser.add_argument("--skip-tabred", action="store_true")
    parser.add_argument(
        "--tabred-data-root",
        type=Path,
        default=TABRED_ROOT_DEFAULT,
        help="Root containing TabReD dataset folders (default: data/tabred).",
    )
    parser.add_argument("--skip-higgs", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--skip-backbone", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument(
        "--include-image-rebuttal",
        action="store_true",
        help="Run optional synthetic image-regression rebuttal benchmark.",
    )
    parser.add_argument(
        "--skip-shifts",
        action="store_true",
        help="Skip Shifts dataset placeholder materialization under data/shifts/.",
    )
    parser.add_argument(
        "--shifts-out-root",
        type=Path,
        default=SHIFTS_OUT_ROOT_DEFAULT,
        help="Root for Shifts placeholder README (default: data/shifts).",
    )
    parser.add_argument(
        "--shifts-dataset",
        type=str,
        default=SHIFTS_DATASET_DEFAULT,
        help="Symbolic Shifts dataset key for placeholder layout (default: solar).",
    )
    parser.add_argument(
        "--only-phases",
        default=None,
        help=(
            "Comma-separated phase names to run exclusively (omit for full pipeline). "
            "Valid keys: " + ", ".join(sorted(NEURIPS_PHASE_KEYS))
        ),
    )
    parser.add_argument(
        "--skip-phases",
        default="",
        help="Comma-separated phase names to skip (applied together with --only-phases).",
    )
    args = parser.parse_args()

    only = _parse_phase_csv(args.only_phases)
    skip = _parse_phase_csv(args.skip_phases) or frozenset()
    unknown = ((only or frozenset()) | skip) - NEURIPS_PHASE_KEYS
    if unknown:
        raise SystemExit(
            "Unknown --only-phases/--skip-phases name(s): "
            + ", ".join(sorted(unknown))
            + "\nValid: "
            + ", ".join(sorted(NEURIPS_PHASE_KEYS))
        )

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run_root = args.run_root
    if run_root is None:
        run_root = REPO_ROOT / "docs/research/sage_reg_results" / date_str / "neurips_sage_reg_full"
    run_root = run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    print(
        "[run_neurips_sage_reg_full] starting",
        f"resolved_run_root={run_root}",
        f"quick={args.quick}",
        f"year_cache={args.year_cache}",
        f"skip_higgs={args.skip_higgs}",
        f"only_phases={sorted(only) if only else None}",
        f"skip_phases={sorted(skip)}",
        flush=True,
    )

    year_cache = args.year_cache.resolve()
    diamonds_cache = args.openml_diamonds_cache.resolve()
    allow_y = not args.no_year_download
    allow_openml = not args.no_openml_download
    active_prefetch = _active_phases(only, skip)
    if NEURIPS_YEAR_CACHE_PHASES & active_prefetch:
        _ensure_year_cache(year_cache, allow_download=allow_y)
    if NEURIPS_OPENML_CACHE_PHASES & active_prefetch:
        _ensure_openml_diamonds_cache(diamonds_cache, allow_download=allow_openml)

    tabred_root = args.tabred_data_root.resolve()

    quick = args.quick
    seeds = list(QUICK_SEEDS if quick else FULL_SEEDS)
    nl, nu, nt = (512, 4096, 1024) if quick else (4096, 131_072, 32_768)
    yteach, ystu = (1, 1) if quick else (32, 32)
    ufrac = ["0.25", "0.5", "1.0"] if not quick else ["1.0"]
    n_labeled_sweep = [2048, 4096, 8192] if quick else [1024, 2048, 4096, 8192, 16384, 32768]
    sweep_nu, sweep_nt = (8192, 2048) if quick else (131_072, 32_768)
    if quick:
        n_labeled_sweep = [2048]
    cat_iters = 800 if quick else 4000

    manifest: dict[str, Any] = {
        "artifact": "neurips_sage_reg_full_manifest",
        "version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "year_cache": str(year_cache),
        "quick": quick,
        "defaults": {"shifts": not args.skip_shifts},
        "phase_filter": {
            "only_phases": sorted(only) if only else None,
            "skip_phases": sorted(skip),
        },
        "phases": {},
    }

    sage_dir = run_root / "sage"
    sage_dir.mkdir(parents=True, exist_ok=True)

    # 1) Year direct
    if _phase_selected("year_direct", only, skip):
        manifest["phases"]["year_direct"] = str(sage_dir / "year_direct")
        _uv_run(
            [
                str(REPO_ROOT / "examples/benchmarks/self_agreement_realdata_year.py"),
                "--cache-path",
                str(year_cache),
                "--no-download",
                "--n-labeled",
                str(nl),
                "--n-unlabeled",
                str(nu),
                "--n-test",
                str(nt),
                "--teacher-epochs",
                str(yteach),
                "--student-epochs",
                str(ystu),
                "--unlabeled-fractions",
                *ufrac,
                "--output-csv",
                str(sage_dir / "year_direct/metrics.csv"),
                "--performance-figure-path",
                str(sage_dir / "year_direct/performance.png"),
                "--calibration-figure-path",
                str(sage_dir / "year_direct/calibration.png"),
                "--summary-json-path",
                str(sage_dir / "year_direct/summary.json"),
            ]
        )
    else:
        manifest["phases"]["year_direct"] = "skipped_not_selected"

    # 2) Multiseed (tuned CSV)
    if _phase_selected("multiseed", only, skip) and args.tuning_csv.is_file():
        multi_cmd = [
            str(REPO_ROOT / "examples/benchmarks/self_agreement_supervised_gap_multiseed.py"),
            "--tuning-csv",
            str(args.tuning_csv),
            "--year-cache-path",
            str(year_cache),
            "--no-year-download",
            "--year-n-labeled",
            str(nl),
            "--year-n-unlabeled",
            str(nu),
            "--year-n-test",
            str(nt),
            "--year-teacher-epochs",
            str(yteach),
            "--year-student-epochs",
            str(ystu),
            "--out-dir",
            str(sage_dir / "multiseed"),
            "--seeds",
            *[str(s) for s in seeds],
        ]
        if args.skip_higgs or not args.higgs_parquet.is_file():
            if not args.higgs_parquet.is_file():
                print("Skip Higgs multiseed: parquet not found", flush=True)
            multi_cmd.append("--skip-higgs")
        else:
            multi_cmd += [
                "--higgs-dataset-path",
                str(args.higgs_parquet),
                "--higgs-split-scale-factor",
                "10",
                "--higgs-parquet-max-sample-rows",
                "8000000",
                "--higgs-teacher-epochs",
                str(yteach),
                "--higgs-student-epochs",
                str(ystu),
            ]
        manifest["phases"]["multiseed"] = str(sage_dir / "multiseed")
        _uv_run(multi_cmd)
    elif _phase_selected("multiseed", only, skip):
        print(f"Skip multiseed: tuning CSV missing {args.tuning_csv}", flush=True)
        manifest["phases"]["multiseed"] = "skipped_no_tuning_csv"
    else:
        manifest["phases"]["multiseed"] = "skipped_not_selected"

    # 2b) Second OpenML tabular track (ggplot2 diamonds / price); uses the same tuned Year row.
    diamonds_out = run_root / "openml_diamonds"
    if (
        _phase_selected("openml_diamonds_multiseed", only, skip)
        and args.tuning_csv.is_file()
        and diamonds_cache.is_file()
    ):
        manifest["phases"]["openml_diamonds_multiseed"] = str(diamonds_out)
        d_rows = _parquet_num_rows(diamonds_cache)
        d_nl, d_nu, d_nt = _scale_year_split_sizes_for_row_budget(
            n_labeled=int(nl), n_unlabeled=int(nu), n_test=int(nt), n_rows=d_rows
        )
        diamonds_cmd = [
            str(REPO_ROOT / "examples/benchmarks/self_agreement_supervised_gap_multiseed.py"),
            "--tuning-csv",
            str(args.tuning_csv),
            "--year-dataset-path",
            str(diamonds_cache),
            "--no-year-download",
            "--year-benchmark-label",
            "openml_diamonds",
            "--year-n-labeled",
            str(d_nl),
            "--year-n-unlabeled",
            str(d_nu),
            "--year-n-test",
            str(d_nt),
            "--year-teacher-epochs",
            str(yteach),
            "--year-student-epochs",
            str(ystu),
            "--out-dir",
            str(diamonds_out),
            "--skip-higgs",
            "--seeds",
            *[str(s) for s in seeds],
        ]
        _uv_run(diamonds_cmd)
    elif _phase_selected("openml_diamonds_multiseed", only, skip):
        if not args.tuning_csv.is_file():
            manifest["phases"]["openml_diamonds_multiseed"] = "skipped_no_tuning_csv"
        else:
            manifest["phases"]["openml_diamonds_multiseed"] = f"missing_cache:{diamonds_cache}"
    else:
        manifest["phases"]["openml_diamonds_multiseed"] = "skipped_not_selected"

    # 3) Labeled sweep + collate
    sweep_out = run_root / "year_labeled_sweep"
    if _phase_selected("year_labeled_sweep", only, skip):
        sweep_out.mkdir(parents=True, exist_ok=True)
        manifest["phases"]["year_labeled_sweep"] = str(sweep_out)
        for nl_s in n_labeled_sweep:
            csv_p = sweep_out / f"year_direct_nl{nl_s}_nu{sweep_nu}_ufrac1.0.csv"
            js_p = sweep_out / f"year_direct_nl{nl_s}_summary.json"
            _uv_run(
                [
                    str(REPO_ROOT / "examples/benchmarks/self_agreement_realdata_year.py"),
                    "--cache-path",
                    str(year_cache),
                    "--no-download",
                    "--n-labeled",
                    str(nl_s),
                    "--n-unlabeled",
                    str(sweep_nu),
                    "--n-test",
                    str(sweep_nt),
                    "--teacher-epochs",
                    str(yteach),
                    "--student-epochs",
                    str(ystu),
                    "--unlabeled-fractions",
                    "1.0",
                    "--output-csv",
                    str(csv_p),
                    "--summary-json-path",
                    str(js_p),
                ]
            )
        _uv_run(
            [
                str(REPO_ROOT / "tools/collate_sage_year_labeled_sweep.py"),
                "--input-dir",
                str(sweep_out),
                "--output-json",
                str(sweep_out / "year_labeled_sweep_collated.json"),
                "--output-csv",
                str(sweep_out / "year_labeled_sweep_collated.csv"),
            ]
        )
    else:
        manifest["phases"]["year_labeled_sweep"] = "skipped_not_selected"

    # 4) Low-label multiseed nl=2048
    low = run_root / "multiseed_year_nl2048"
    if _phase_selected("multiseed_year_nl2048", only, skip) and args.tuning_csv.is_file():
        manifest["phases"]["multiseed_year_nl2048"] = str(low)
        _uv_run(
            [
                str(REPO_ROOT / "examples/benchmarks/self_agreement_supervised_gap_multiseed.py"),
                "--tuning-csv",
                str(args.tuning_csv),
                "--year-cache-path",
                str(year_cache),
                "--no-year-download",
                "--year-n-labeled",
                "2048",
                "--year-n-unlabeled",
                str(sweep_nu),
                "--year-n-test",
                str(sweep_nt),
                "--year-teacher-epochs",
                str(yteach),
                "--year-student-epochs",
                str(ystu),
                "--out-dir",
                str(low),
                "--skip-higgs",
                "--seeds",
                *[str(s) for s in seeds],
            ]
        )
    elif _phase_selected("multiseed_year_nl2048", only, skip):
        manifest["phases"]["multiseed_year_nl2048"] = "skipped_no_tuning_csv"
    else:
        manifest["phases"]["multiseed_year_nl2048"] = "skipped_not_selected"

    # 5) CatBoost
    cat_out = run_root / "catboost"
    if _phase_selected("catboost", only, skip) and not args.skip_catboost:
        cmd = [
            str(REPO_ROOT / "tools/sage_catboost_baselines.py"),
            "--year-cache",
            str(year_cache),
            "--out-dir",
            str(cat_out),
            "--iterations",
            str(cat_iters),
            "--seeds",
            *[str(s) for s in seeds[:3]],
        ]
        if args.quick:
            cmd += ["--n-labeled", "2048", "4096"]
        if args.higgs_parquet.is_file() and not args.skip_higgs:
            cmd += ["--higgs-parquet", str(args.higgs_parquet), "--higgs-seed", str(seeds[0])]
        if _uv_run(cmd, check=False) != 0:
            print("CatBoost phase failed (install catboost?)", flush=True)
            manifest["phases"]["catboost"] = "failed_or_skipped"
        else:
            manifest["phases"]["catboost"] = str(cat_out)
    elif _phase_selected("catboost", only, skip):
        manifest["phases"]["catboost"] = "skipped"
    else:
        manifest["phases"]["catboost"] = "skipped_not_selected"

    # 6) TabReD
    tabred_out = run_root / "tabred"
    if _phase_selected("tabred", only, skip) and not args.skip_tabred:
        kaggle = Path.home() / ".kaggle" / "kaggle.json"
        kaggle_ok = kaggle.is_file()
        tabred_local_ok = _tabred_local_markers_present(tabred_root)
        if kaggle_ok:
            if _uv_run(
                [
                    str(REPO_ROOT / "tools/fetch_tabred_data.py"),
                    "--out-dir",
                    str(tabred_root),
                    "--skip-if-present",
                ],
                check=False,
            ):
                manifest["phases"]["tabred"] = "fetch_failed"
            else:
                tabred_out.mkdir(parents=True, exist_ok=True)
                if (
                    _uv_run(
                        [
                            str(REPO_ROOT / "examples/benchmarks/tabred_sage_ssl_probe.py"),
                            "--tabred-data-root",
                            str(tabred_root),
                            "--out-dir",
                            str(tabred_out),
                        ]
                        + (["--quick"] if quick else []),
                        check=False,
                    )
                    != 0
                ):
                    manifest["phases"]["tabred"] = "probe_failed"
                else:
                    manifest["phases"]["tabred"] = str(tabred_out)
        elif tabred_local_ok:
            print(
                f"TabReD: using local materialized data under {tabred_root} "
                "(skip fetch; no ~/.kaggle/kaggle.json)",
                flush=True,
            )
            manifest["phases"]["tabred_fetch"] = "skipped_no_kaggle_local_ok"
            tabred_out.mkdir(parents=True, exist_ok=True)
            if (
                _uv_run(
                    [
                        str(REPO_ROOT / "examples/benchmarks/tabred_sage_ssl_probe.py"),
                        "--tabred-data-root",
                        str(tabred_root),
                        "--out-dir",
                        str(tabred_out),
                    ]
                    + (["--quick"] if quick else []),
                    check=False,
                )
                != 0
            ):
                manifest["phases"]["tabred"] = "probe_failed"
            else:
                manifest["phases"]["tabred"] = str(tabred_out)
        else:
            expected = ", ".join(
                str(tabred_root / name / "info.json") for name in TABRED_DEFAULT_DATASETS
            )
            print(
                "Skip TabReD: ~/.kaggle/kaggle.json not found and local TabReD markers missing "
                f"(expected {expected})",
                flush=True,
            )
            manifest["phases"]["tabred"] = "skipped_no_kaggle_no_local"
    elif _phase_selected("tabred", only, skip):
        manifest["phases"]["tabred"] = "skipped"
    else:
        manifest["phases"]["tabred"] = "skipped_not_selected"

    # 7) Synthetic
    syn = run_root / "synthetic"
    if _phase_selected("synthetic", only, skip) and not args.skip_synthetic:
        syn.mkdir(parents=True, exist_ok=True)
        sj = syn / "summary.json"
        syn_cmd = [
            str(REPO_ROOT / "examples/benchmarks/self_agreement_synthetic.py"),
            "--summary-json-path",
            str(sj),
        ]
        # Benchmark only supports --diagnostic-figure-path (uses max unlabeled fraction internally).
        if not quick:
            syn_cmd += [
                "--diagnostic-figure-path",
                str(syn / "stress_diagnostics.png"),
            ]
        _uv_run(syn_cmd)
        manifest["phases"]["synthetic"] = str(syn)
    elif _phase_selected("synthetic", only, skip):
        manifest["phases"]["synthetic"] = "skipped"
    else:
        manifest["phases"]["synthetic"] = "skipped_not_selected"

    # 8) Backbone
    bb = run_root / "backbone"
    if _phase_selected("backbone", only, skip) and not args.skip_backbone:
        bb.mkdir(parents=True, exist_ok=True)
        _uv_run(
            [
                str(REPO_ROOT / "examples/benchmarks/self_agreement_backbone_comparison.py"),
                "--summary-json-path",
                str(bb / "summary.json"),
            ]
        )
        manifest["phases"]["backbone"] = str(bb)
    elif _phase_selected("backbone", only, skip):
        manifest["phases"]["backbone"] = "skipped"
    else:
        manifest["phases"]["backbone"] = "skipped_not_selected"

    # 9) Ablations
    abl = run_root / "ablations"
    if _phase_selected("ablations", only, skip) and not args.skip_ablations:
        if (
            _uv_run(
                [
                    str(REPO_ROOT / "tools/run_sage_year_ablations.py"),
                    "--year-cache",
                    str(year_cache),
                    "--out-dir",
                    str(abl),
                ]
                + (["--quick"] if quick else []),
                check=False,
            )
            != 0
        ):
            manifest["phases"]["ablations"] = "failed"
        else:
            manifest["phases"]["ablations"] = str(abl)
    elif _phase_selected("ablations", only, skip):
        manifest["phases"]["ablations"] = "skipped"
    else:
        manifest["phases"]["ablations"] = "skipped_not_selected"

    if _phase_selected("shifts", only, skip) and not args.skip_shifts:
        shifts_out = args.shifts_out_root.resolve()
        sh_cmd = [
            str(REPO_ROOT / "tools/fetch_shifts_dataset.py"),
            "--out-root",
            str(shifts_out),
            "--dataset",
            str(args.shifts_dataset),
        ]
        sh_rc = _uv_run(sh_cmd, check=False)
        readme = shifts_out / args.shifts_dataset / "README.txt"
        if sh_rc == 0 and readme.is_file():
            manifest["phases"]["shifts"] = str(readme)
        elif sh_rc != 0:
            manifest["phases"]["shifts"] = "helper_failed"
        else:
            manifest["phases"]["shifts"] = "incomplete"
    elif _phase_selected("shifts", only, skip):
        manifest["phases"]["shifts"] = "skipped"
    else:
        manifest["phases"]["shifts"] = "skipped_not_selected"

    if _phase_selected("image_rebuttal", only, skip) and args.include_image_rebuttal:
        image_dir = run_root / "image_rebuttal"
        image_dir.mkdir(parents=True, exist_ok=True)
        _uv_run(
            [
                str(REPO_ROOT / "examples/benchmarks/image_regression_rebuttal.py"),
                "--summary-json-path",
                str(image_dir / "summary.json"),
                "--teacher-epochs",
                "1" if quick else "5",
                "--student-epochs",
                "1" if quick else "5",
            ]
        )
        manifest["phases"]["image_rebuttal"] = str(image_dir)
    elif _phase_selected("image_rebuttal", only, skip):
        manifest["phases"]["image_rebuttal"] = "skipped"
    else:
        manifest["phases"]["image_rebuttal"] = "skipped_not_selected"

    manifest["phases"]["aggregate"] = (
        "selected" if _phase_selected("aggregate", only, skip) else "skipped_not_selected"
    )
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    (run_root / "neurips_sage_reg_full_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    if _phase_selected("aggregate", only, skip):
        _uv_run(
            [
                str(REPO_ROOT / "tools/aggregate_sage_paper_report.py"),
                "--run-root",
                str(run_root),
                "--write-markdown",
            ]
        )
    else:
        print("[run_neurips_sage_reg_full] skip aggregate (phase not selected)", flush=True)
    print(f"\nDone. Run root:\n  {run_root}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(
            f"\n[run_neurips_sage_reg_full] FATAL {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        raise
