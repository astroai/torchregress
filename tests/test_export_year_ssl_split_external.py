from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"


def _load_tool_export():
    path = REPO / "tools" / "export_year_ssl_split_for_external.py"
    spec = importlib.util.spec_from_file_location("_export_year", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("tool spec")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_year_mod() -> object:
    path = EXAMPLES / "benchmarks" / "self_agreement_realdata_year.py"
    spec = importlib.util.spec_from_file_location("_year_bench", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("load year")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    sys.path.insert(0, str(EXAMPLES / "benchmarks"))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


def _write_csv(path: Path, *, n: int = 400) -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(n, 6)).astype("float32")
    y = (x[:, :2].sum(axis=1) + 0.05 * rng.standard_normal(n)).astype("float32")
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(6)])
    df["target"] = y
    df.to_csv(path, index=False)


def test_export_default_split_npz(tmp_path: Path) -> None:
    year = _load_year_mod()
    csv_path = tmp_path / "d.csv"
    _write_csv(csv_path)
    out = tmp_path / "exp"
    exp = _load_tool_export()

    cfg = year.YearRealDataConfig(
        dataset_path=str(csv_path),
        allow_download=False,
        n_labeled=20,
        n_unlabeled=120,
        n_test=40,
        seed=9,
    )
    exp.export_split(
        cfg,
        out_dir=out,
        split_mode="default",
        label_pool_percent=None,
        shift_mode=None,
        min_unlabeled=None,
    )
    assert (out / "x_labeled.npy").exists()
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["protocol"] == "torchregress-year-ssl-export-v1"
    xl = np.load(out / "x_labeled.npy")
    assert xl.shape[1] == 6


def test_export_label_pool_fraction(tmp_path: Path) -> None:
    year = _load_year_mod()
    csv_path = tmp_path / "d.csv"
    _write_csv(csv_path)
    out = tmp_path / "exp2"
    exp = _load_tool_export()

    cfg = year.YearRealDataConfig(
        dataset_path=str(csv_path),
        allow_download=False,
        n_labeled=20,
        n_unlabeled=120,
        n_test=40,
        seed=11,
    )
    exp.export_split(
        cfg,
        out_dir=out,
        split_mode="label_pool_fraction",
        label_pool_percent=30.0,
        shift_mode="none",
        min_unlabeled=32,
    )
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["split_mode"] == "label_pool_fraction"
