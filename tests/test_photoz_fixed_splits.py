from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_example_module(name: str):
    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    path = examples_dir / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load example module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_split(path: Path, n_rows: int) -> None:
    frame = pd.DataFrame(
        {
            "spec_z": [0.1 + 0.01 * idx for idx in range(n_rows)],
            "spec_z_err": [0.02] * n_rows,
            "g_r": [0.6] * n_rows,
            "r_i": [0.4] * n_rows,
            "i_z": [0.3] * n_rows,
            "z_y": [0.2] * n_rows,
            "g_r_err": [0.05] * n_rows,
            "r_i_err": [0.04] * n_rows,
            "i_z_err": [0.04] * n_rows,
            "z_y_err": [0.05] * n_rows,
        }
    )
    frame.to_csv(path, index=False)


def test_photoz_benchmark_make_splits_accepts_explicit_split_paths(tmp_path: Path) -> None:
    module = _load_example_module("photoz_benchmark_comparison")
    train_path = tmp_path / "transferz_train_photoz.csv"
    cal_path = tmp_path / "transferz_cal_photoz.csv"
    test_path = tmp_path / "transferz_test_photoz.csv"
    _write_split(train_path, 12)
    _write_split(cal_path, 8)
    _write_split(test_path, 8)

    cfg = module.PhotoZBenchmarkConfig(
        n_train=12,
        n_cal=8,
        n_test=8,
        train_dataset_path=str(train_path),
        cal_dataset_path=str(cal_path),
        test_dataset_path=str(test_path),
        require_real_data=True,
    )
    splits = module._make_splits(cfg)

    assert tuple(splits["x_train"].shape) == (12, 4)
    assert tuple(splits["x_cal"].shape) == (8, 4)
    assert tuple(splits["x_test"].shape) == (8, 4)
    assert splits["data_source"] == "external_splits:transferz"
