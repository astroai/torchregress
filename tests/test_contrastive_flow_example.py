from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


def _load_example_module(stem: str) -> ModuleType:
    path = EXAMPLES_DIR / f"{stem}.py"
    module_name = f"_contrastive_example_{stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load example module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_contrastive_flow_parameter_estimation_main_smoke() -> None:
    mod = _load_example_module("contrastive_flow_parameter_estimation")
    cfg = mod.ContrastiveFlowConfig(
        n_train=24,
        n_test=6,
        events_per_experiment=32,
        batch_size=8,
        n_epochs=1,
        n_negatives=2,
        mu_grid_size=9,
        nuisance_grid_size=7,
        make_plot=False,
    )

    try:
        metrics = mod.main(cfg)
    except ImportError as exc:
        if "zuko" in str(exc).lower():
            pytest.skip("optional zuko dependency not available")
        raise

    assert set(metrics) == {"mu_mae", "nuisance_mae"}
