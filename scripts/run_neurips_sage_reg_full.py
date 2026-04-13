#!/usr/bin/env python3
"""One-shot SAGE-Reg NeurIPS evidence: Year cache, tabular runs, sweeps, optional extras.

Covers optional TabReD, CatBoost, and Higgs when dependencies or assets exist.

Usage (repo root)::

    uv run python scripts/run_neurips_sage_reg_full.py
    uv run python scripts/run_neurips_sage_reg_full.py --quick

Defaults need no flags. Large raw assets (Higgs parquet, TabReD/Kaggle) are
skipped with logs if missing.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Canonical caches / inputs (override with env or --run-root only changes output tree)
YEAR_CACHE_DEFAULT = REPO_ROOT / "data" / "paper" / "openml_year.csv"
TUNING_CSV_DEFAULT = (
    REPO_ROOT / "docs/research/sage_reg_results/2026-04-10/supervised_gap_tuning_v3/sweep.csv"
)
HIGGS_PARQUET_DEFAULT = (
    REPO_ROOT
    / "docs/research/sage_reg_results/2026-04-09/higgs_public/extracted"
    / "FAIR_Universe_HiggsML_data.parquet"
)
TABRED_ROOT_DEFAULT = REPO_ROOT / "data" / "tabred"
SHIFTS_OUT_ROOT_DEFAULT = REPO_ROOT / "data" / "shifts"
SHIFTS_DATASET_DEFAULT = "solar"

FULL_SEEDS = (260410, 260411, 260412, 260413, 260414, 260415)
QUICK_SEEDS = (260410, 260411)


def _uv_run(parts: list[str], *, cwd: Path | None = None, check: bool = True) -> int:
    cmd = ["uv", "run", "python", *parts]
    print("==", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(cwd or REPO_ROOT), env=os.environ.copy(), check=False)
    if check and r.returncode != 0:
        raise subprocess.CalledProcessError(r.returncode, cmd, None, None)
    return int(r.returncode)


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
    parser.add_argument("--year-cache", type=Path, default=YEAR_CACHE_DEFAULT)
    parser.add_argument("--tuning-csv", type=Path, default=TUNING_CSV_DEFAULT)
    parser.add_argument("--higgs-parquet", type=Path, default=HIGGS_PARQUET_DEFAULT)
    parser.add_argument("--skip-catboost", action="store_true")
    parser.add_argument("--skip-tabred", action="store_true")
    parser.add_argument("--skip-higgs", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--skip-backbone", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
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
    args = parser.parse_args()

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run_root = args.run_root
    if run_root is None:
        run_root = REPO_ROOT / "docs/research/sage_reg_results" / date_str / "neurips_sage_reg_full"
    run_root = run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    year_cache = args.year_cache.resolve()
    allow_y = not args.no_year_download
    _ensure_year_cache(year_cache, allow_download=allow_y)

    quick = args.quick
    seeds = list(QUICK_SEEDS if quick else FULL_SEEDS)
    nl, nu, nt = (512, 4096, 1024) if quick else (4096, 131_072, 32_768)
    yteach, ystu = (1, 1) if quick else (32, 32)
    ufrac = ["0.25", "0.5", "1.0"] if not quick else ["1.0"]
    n_labeled_sweep = [2048, 4096, 8192] if quick else [2048, 4096, 8192, 16384, 32768]
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
        "phases": {},
    }

    sage_dir = run_root / "sage"
    sage_dir.mkdir(parents=True, exist_ok=True)

    # 1) Year direct
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

    # 2) Multiseed (tuned CSV)
    if args.tuning_csv.is_file():
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
    else:
        print(f"Skip multiseed: tuning CSV missing {args.tuning_csv}", flush=True)
        manifest["phases"]["multiseed"] = "skipped_no_tuning_csv"

    # 3) Labeled sweep + collate
    sweep_out = run_root / "year_labeled_sweep"
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

    # 4) Low-label multiseed nl=2048
    low = run_root / "multiseed_year_nl2048"
    if args.tuning_csv.is_file():
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
    else:
        manifest["phases"]["multiseed_year_nl2048"] = "skipped_no_tuning_csv"

    # 5) CatBoost
    cat_out = run_root / "catboost"
    if not args.skip_catboost:
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
    else:
        manifest["phases"]["catboost"] = "skipped"

    # 6) TabReD
    tabred_out = run_root / "tabred"
    if not args.skip_tabred:
        kaggle = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle.is_file():
            if _uv_run(
                [
                    str(REPO_ROOT / "tools/fetch_tabred_data.py"),
                    "--out-dir",
                    str(TABRED_ROOT_DEFAULT),
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
                            str(TABRED_ROOT_DEFAULT),
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
            print("Skip TabReD: ~/.kaggle/kaggle.json not found", flush=True)
            manifest["phases"]["tabred"] = "skipped_no_kaggle"
    else:
        manifest["phases"]["tabred"] = "skipped"

    # 7) Synthetic
    syn = run_root / "synthetic"
    if not args.skip_synthetic:
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
    else:
        manifest["phases"]["synthetic"] = "skipped"

    # 8) Backbone
    bb = run_root / "backbone"
    if not args.skip_backbone:
        bb.mkdir(parents=True, exist_ok=True)
        _uv_run(
            [
                str(REPO_ROOT / "examples/benchmarks/self_agreement_backbone_comparison.py"),
                "--summary-json-path",
                str(bb / "summary.json"),
            ]
        )
        manifest["phases"]["backbone"] = str(bb)
    else:
        manifest["phases"]["backbone"] = "skipped"

    # 9) Ablations
    abl = run_root / "ablations"
    if not args.skip_ablations:
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
    else:
        manifest["phases"]["ablations"] = "skipped"

    if not args.skip_shifts:
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
    else:
        manifest["phases"]["shifts"] = "skipped"

    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    (run_root / "neurips_sage_reg_full_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    _uv_run(
        [
            str(REPO_ROOT / "tools/aggregate_sage_paper_report.py"),
            "--run-root",
            str(run_root),
            "--write-markdown",
        ]
    )
    print(f"\nDone. Run root:\n  {run_root}", flush=True)


if __name__ == "__main__":
    main()
