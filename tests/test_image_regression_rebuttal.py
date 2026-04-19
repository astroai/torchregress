from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


def _load_module(stem: str) -> ModuleType:
    path = EXAMPLES_DIR / "benchmarks" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"_benchmark_{stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load benchmark module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_image_regression_rebuttal_smoke(tmp_path: Path) -> None:
    mod = _load_module("image_regression_rebuttal")
    summary_path = tmp_path / "image_rebuttal_summary.json"
    cfg = mod.ImageRebuttalConfig(
        n_labeled=64,
        n_unlabeled=128,
        n_test=96,
        teacher_epochs=1,
        student_epochs=1,
        batch_size=32,
    )
    rows = mod.main(cfg, summary_json_path=str(summary_path))
    assert rows
    assert summary_path.exists()
    methods = {str(row["Method"]) for row in rows}
    assert methods == {"SupervisedOnly", "ConfidenceWeightedPseudoLabel", "SAGE-Reg"}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
