"""Regression: ``run_neurips_sage_reg_full.py --only-phases`` with ``--quick`` on temp run roots."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_neurips(
    *,
    run_root: Path,
    year_csv: Path,
    tuning: Path,
    shifts_stub: Path,
    only_phases: str,
) -> subprocess.CompletedProcess[str]:
    selected = {x.strip() for x in only_phases.split(",") if x.strip()}
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/run_neurips_sage_reg_full.py"),
        "--quick",
        "--run-root",
        str(run_root),
        "--year-cache",
        str(year_csv),
        "--no-year-download",
        "--tuning-csv",
        str(tuning),
        "--only-phases",
        only_phases,
        "--skip-catboost",
        "--skip-tabred",
        "--skip-higgs",
        "--shifts-out-root",
        str(shifts_stub),
    ]
    if "synthetic" not in selected:
        cmd.append("--skip-synthetic")
    if "backbone" not in selected:
        cmd.append("--skip-backbone")
    if "ablations" not in selected:
        cmd.append("--skip-ablations")
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)


def test_only_phases_disjoint_quick_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    year_csv = REPO_ROOT / "data" / "paper" / "openml_year.csv"
    if not year_csv.is_file():
        pytest.skip(f"missing Year cache: {year_csv}")
    tuning = REPO_ROOT / "tests/fixtures/sage_smoke_tuning.csv"
    shifts_stub = tmp_path / "shifts_stub"

    run_a = tmp_path / "run_a"
    r_a = _run_neurips(
        run_root=run_a,
        year_csv=year_csv,
        tuning=tuning,
        shifts_stub=shifts_stub,
        only_phases="year_direct,aggregate",
    )
    assert r_a.returncode == 0, r_a.stdout + "\n" + r_a.stderr
    man_a = json.loads((run_a / "neurips_sage_reg_full_manifest.json").read_text(encoding="utf-8"))
    assert man_a["phases"]["year_direct"] != "skipped_not_selected"
    assert (run_a / "sage" / "year_direct" / "summary.json").is_file()

    run_b = tmp_path / "run_b"
    r_b = _run_neurips(
        run_root=run_b,
        year_csv=year_csv,
        tuning=tuning,
        shifts_stub=shifts_stub,
        only_phases="synthetic,aggregate",
    )
    assert r_b.returncode == 0, r_b.stdout + "\n" + r_b.stderr
    man_b = json.loads((run_b / "neurips_sage_reg_full_manifest.json").read_text(encoding="utf-8"))
    assert man_b["phases"]["synthetic"] != "skipped_not_selected"
    assert (run_b / "synthetic" / "summary.json").is_file()
    assert man_b["phases"]["year_direct"] == "skipped_not_selected"


def test_unknown_only_phases_exits_nonzero(tmp_path: Path) -> None:
    year_csv = REPO_ROOT / "data" / "paper" / "openml_year.csv"
    if not year_csv.is_file():
        pytest.skip(f"missing Year cache: {year_csv}")
    tuning = REPO_ROOT / "tests/fixtures/sage_smoke_tuning.csv"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/run_neurips_sage_reg_full.py"),
        "--quick",
        "--run-root",
        str(tmp_path / "bad"),
        "--year-cache",
        str(year_csv),
        "--no-year-download",
        "--tuning-csv",
        str(tuning),
        "--only-phases",
        "not_a_real_phase",
    ]
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert r.returncode != 0


def test_canfar_launch_from_plan_dry_run(tmp_path: Path) -> None:
    pytest.importorskip("yaml", reason="YAML work plans need PyYAML (torchregress[canfar])")
    plan = REPO_ROOT / "scripts/canfar/canfar_work_plan.example.yaml"
    if not plan.is_file():
        pytest.skip("example work plan missing")
    manifest = tmp_path / "m.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/canfar_launch_from_plan.py"),
        "--plan",
        str(plan),
        "--run-id",
        "pytest_dry",
        "--dry-run",
        "--manifest-out",
        str(manifest),
        "--max-concurrent",
        "4",
        "--no-wait-between-waves",
    ]
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["artifact"] == "canfar_work_plan_manifest"
    assert len(data["sessions"]) >= 1
    assert all(s.get("status") == "dry_run" for s in data["sessions"])
