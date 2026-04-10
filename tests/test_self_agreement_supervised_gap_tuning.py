from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


def _load_module(stem: str) -> ModuleType:
    path = EXAMPLES_DIR / "benchmarks" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"_benchmark_{stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load benchmark module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(EXAMPLES_DIR / "benchmarks"))
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
        sys.path.pop(0)
    return module


def test_self_agreement_supervised_gap_tuning_smoke(tmp_path: Path) -> None:
    mod = _load_module("self_agreement_supervised_gap_tuning")
    year_path = tmp_path / "year_local.csv"
    higgs_path = tmp_path / "higgs_local.csv"
    csv_path = tmp_path / "tuning.csv"
    fig_path = tmp_path / "tuning.png"
    summary_path = tmp_path / "tuning.json"

    n_year = 4096
    year_df = pd.DataFrame(
        {
            **{f"f{i}": ((pd.Series(range(n_year)) * (i + 1)) % 101).astype("float32") for i in range(8)},
            "target": ((pd.Series(range(n_year)) % 17) / 8.0).astype("float32"),
        }
    )
    year_df.to_csv(year_path, index=False)

    n_higgs = 4096
    higgs_df = pd.DataFrame(
        {
            "PRI_lep_pt": ((pd.Series(range(n_higgs)) * 3) % 97).astype("float32"),
            "PRI_met": ((pd.Series(range(n_higgs)) * 7) % 89).astype("float32"),
            "DER_mass_vis": ((pd.Series(range(n_higgs)) * 11) % 131).astype("float32"),
            "weights": (0.1 + (pd.Series(range(n_higgs)) % 31) / 100.0).astype("float32"),
            "detailed_labels": ["ztautau"] * n_higgs,
            "labels": (pd.Series(range(n_higgs)) % 2).astype("float32"),
        }
    )
    higgs_df.to_csv(higgs_path, index=False)

    cfg = mod.SupervisedGapTuningConfig(
        year_dataset_path=str(year_path),
        year_allow_download=False,
        higgs_dataset_path=str(higgs_path),
        tau_values=(0.18,),
        unlabeled_noise_values=(0.03,),
        feature_drop_prob_values=(0.1,),
        pseudo_weight_values=(0.6,),
        agreement_weight_values=(0.5,),
        weight_power_values=(2.0,),
        hard_weight_threshold_values=(0.8,),
        year_n_labeled=128,
        year_n_unlabeled=512,
        year_n_test=256,
        year_teacher_epochs=2,
        year_student_epochs=2,
        year_batch_size=64,
        year_hidden=24,
        higgs_n_train=128,
        higgs_n_unlabeled_id=256,
        higgs_n_unlabeled_ood=256,
        higgs_n_id_test=128,
        higgs_n_ood_test=128,
        higgs_teacher_epochs=2,
        higgs_student_epochs=2,
        higgs_batch_size=64,
        higgs_hidden=24,
        log_progress=False,
    )

    rows = mod.main(
        cfg,
        output_csv=str(csv_path),
        figure_path=str(fig_path),
        summary_json_path=str(summary_path),
    )
    assert rows
    assert csv_path.exists()
    assert fig_path.exists()
    assert summary_path.exists()
    assert {str(row["Benchmark"]) for row in rows} == {"year", "higgs_public"}

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    assert "supervised-only gap" in payload["task"]

    resumed_rows = mod.main(
        cfg,
        output_csv=str(csv_path),
        figure_path=str(fig_path),
        summary_json_path=str(summary_path),
    )
    assert len(resumed_rows) == len(rows)
