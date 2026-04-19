"""Smoke: ``scripts/run_neurips_sage_reg_full.py --quick`` on a synthetic Year-like CSV."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_year_like_csv(path: Path, *, n_rows: int = 16_000) -> None:
    rng = np.random.default_rng(42)
    data = {f"f{i}": rng.standard_normal(n_rows).astype("float32") for i in range(8)}
    x = np.stack([data[f"f{i}"] for i in range(8)], axis=1)
    data["target"] = (x[:, 0] * 0.5 + rng.standard_normal(n_rows) * 0.1).astype("float32")
    pd.DataFrame(data).to_csv(path, index=False)


def _write_minimal_tabred_triplet(root: Path) -> None:
    """Create tiny-but-large-enough TabReD tensor layouts for the default dataset names."""
    pytest.importorskip("polars")
    rng = np.random.default_rng(0)
    n = 65_536
    x = rng.standard_normal((n, 8)).astype(np.float32)
    y = (x[:, 0] * 0.5 + rng.standard_normal(n) * 0.1).astype(np.float32).reshape(-1, 1)

    train = np.arange(0, 50_000, dtype=np.int64)
    val = np.arange(50_000, 55_000, dtype=np.int64)
    test = np.arange(55_000, n, dtype=np.int64)

    for name in ("cooking-time", "delivery-eta", "maps-routing"):
        d = root / name
        sp = d / "split-default"
        d.mkdir(parents=True)
        sp.mkdir(parents=True)
        (d / "info.json").write_text(
            json.dumps({"name": name, "task_type": "regression"}),
            encoding="utf-8",
        )
        np.save(d / "X_num.npy", x)
        np.save(d / "Y.npy", y)
        np.save(sp / "train_idx.npy", train)
        np.save(sp / "val_idx.npy", val)
        np.save(sp / "test_idx.npy", test)


def test_neurips_sage_reg_full_quick_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    year_csv = tmp_path / "openml_year_like.csv"
    _write_year_like_csv(year_csv)
    tuning = REPO_ROOT / "tests/fixtures/sage_smoke_tuning.csv"
    run_root = tmp_path / "sage_run"
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
        "--skip-catboost",
        "--skip-tabred",
        "--skip-higgs",
        "--skip-synthetic",
        "--skip-backbone",
        "--skip-ablations",
        "--shifts-out-root",
        str(tmp_path / "shifts_stub"),
    ]
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert r.returncode == 0, f"stderr:\n{r.stderr}\nstdout:\n{r.stdout}"
    assert (run_root / "neurips_sage_reg_full_manifest.json").is_file()
    assert (run_root / "METRICS.md").is_file()
    assert (run_root / "sage_paper_report.json").is_file()


def test_neurips_sage_reg_full_quick_tabred_local_without_kaggle_json(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("polars")

    year_csv = tmp_path / "openml_year_like.csv"
    _write_year_like_csv(year_csv)
    tuning = REPO_ROOT / "tests/fixtures/sage_smoke_tuning.csv"

    tabred_root = tmp_path / "tabred"
    _write_minimal_tabred_triplet(tabred_root)

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    # Intentionally no ~/.kaggle/kaggle.json under fake HOME.

    run_root = tmp_path / "sage_run_tabred"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/run_neurips_sage_reg_full.py"),
        "--quick",
        "--run-root",
        str(run_root),
        "--year-cache",
        str(year_csv),
        "--no-year-download",
        "--no-openml-download",
        "--openml-diamonds-cache",
        str(tmp_path / "missing_diamonds.parquet"),
        "--tuning-csv",
        str(tuning),
        "--tabred-data-root",
        str(tabred_root),
        "--skip-catboost",
        "--skip-higgs",
        "--skip-synthetic",
        "--skip-backbone",
        "--skip-ablations",
        "--shifts-out-root",
        str(tmp_path / "shifts_stub"),
    ]
    env = os.environ.copy()
    env["HOME"] = str(fake_home)

    r = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"stderr:\n{r.stderr}\nstdout:\n{r.stdout}"

    manifest = json.loads(
        (run_root / "neurips_sage_reg_full_manifest.json").read_text(encoding="utf-8")
    )
    phases = manifest.get("phases", {})
    assert phases.get("tabred_fetch") == "skipped_no_kaggle_local_ok"
    assert isinstance(phases.get("tabred"), str)
    assert (Path(phases["tabred"]) / "bundle_summary.json").is_file()
