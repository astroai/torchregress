"""Smoke: ``scripts/run_neurips_spt_reg_full.py --quick`` on a synthetic Year-like CSV."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_year_like_csv(path: Path, *, n_rows: int = 16_000) -> None:
    rng = np.random.default_rng(43)
    data = {f"f{i}": rng.standard_normal(n_rows).astype("float32") for i in range(8)}
    x = np.stack([data[f"f{i}"] for i in range(8)], axis=1)
    data["target"] = (x[:, 0] * 0.5 + rng.standard_normal(n_rows) * 0.1).astype("float32")
    pd.DataFrame(data).to_csv(path, index=False)


def test_neurips_spt_reg_full_quick_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    year_csv = tmp_path / "openml_year_like.csv"
    _write_year_like_csv(year_csv)
    run_root = tmp_path / "spt_run"
    cmd = [
        "uv",
        "run",
        "python",
        str(REPO_ROOT / "scripts/run_neurips_spt_reg_full.py"),
        "--quick",
        "--run-root",
        str(run_root),
        "--year-cache",
        str(year_csv),
        "--no-year-download",
        "--skip-stage-a-sweep",
        "--shifts-out-root",
        str(tmp_path / "shifts_stub"),
    ]
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert r.returncode == 0, f"stderr:\n{r.stderr}\nstdout:\n{r.stdout}"
    assert (run_root / "neurips_spt_reg_full_manifest.json").is_file()
    assert (run_root / "METRICS.md").is_file()
    assert (run_root / "spt_paper_report.json").is_file()
