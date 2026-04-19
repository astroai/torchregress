from __future__ import annotations

import csv
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


def test_self_agreement_supervised_gap_multiseed_smoke(tmp_path: Path) -> None:
    mod = _load_module("self_agreement_supervised_gap_multiseed")
    tuning_csv = tmp_path / "tuning.csv"
    tuning_json = tmp_path / "tuning.json"
    out_dir = tmp_path / "multiseed"
    year_path = tmp_path / "year_local.csv"
    higgs_path = tmp_path / "higgs_local.csv"

    n_year = 4096
    pd.DataFrame(
        {
            **{
                f"f{i}": ((pd.Series(range(n_year)) * (i + 1)) % 101).astype("float32")
                for i in range(8)
            },
            "target": ((pd.Series(range(n_year)) % 17) / 8.0).astype("float32"),
        }
    ).to_csv(year_path, index=False)

    n_higgs = 4096
    pd.DataFrame(
        {
            "PRI_lep_pt": ((pd.Series(range(n_higgs)) * 3) % 97).astype("float32"),
            "PRI_met": ((pd.Series(range(n_higgs)) * 7) % 89).astype("float32"),
            "DER_mass_vis": ((pd.Series(range(n_higgs)) * 11) % 131).astype("float32"),
            "weights": (0.1 + (pd.Series(range(n_higgs)) % 31) / 100.0).astype("float32"),
            "detailed_labels": ["ztautau"] * n_higgs,
            "labels": (pd.Series(range(n_higgs)) % 2).astype("float32"),
        }
    ).to_csv(higgs_path, index=False)

    with tuning_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Benchmark",
                "tau",
                "unlabeled_noise",
                "feature_drop_prob",
                "feature_mix_prob",
                "pseudo_weight",
                "agreement_weight",
                "weight_power",
                "hard_weight_threshold",
                "SAGEMinusSupervised",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Benchmark": "year",
                "tau": "0.28",
                "unlabeled_noise": "0.10",
                "feature_drop_prob": "0.1",
                "feature_mix_prob": "0.1",
                "pseudo_weight": "0.4",
                "agreement_weight": "0.5",
                "weight_power": "2.0",
                "hard_weight_threshold": "0.8",
                "SAGEMinusSupervised": "0.2",
            }
        )
        writer.writerow(
            {
                "Benchmark": "higgs_public",
                "tau": "0.18",
                "unlabeled_noise": "0.10",
                "feature_drop_prob": "0.1",
                "feature_mix_prob": "0.2",
                "pseudo_weight": "0.4",
                "agreement_weight": "0.5",
                "weight_power": "2.0",
                "hard_weight_threshold": "0.8",
                "SAGEMinusSupervised": "0.7",
            }
        )

    tuning_json.write_text(
        json.dumps(
            {
                "config": {
                    "seed": 260410,
                    "year_allow_download": False,
                    "year_n_labeled": 128,
                    "year_n_unlabeled": 512,
                    "year_n_test": 256,
                    "year_hidden": 24,
                    "year_batch_size": 64,
                    "year_teacher_epochs": 2,
                    "year_student_epochs": 2,
                    "higgs_n_train": 128,
                    "higgs_n_unlabeled_id": 256,
                    "higgs_n_unlabeled_ood": 256,
                    "higgs_n_id_test": 128,
                    "higgs_n_ood_test": 128,
                    "higgs_hidden": 24,
                    "higgs_batch_size": 64,
                    "higgs_teacher_epochs": 2,
                    "higgs_student_epochs": 2,
                }
            }
        ),
        encoding="utf-8",
    )

    payload = mod.main(
        mod.SupervisedGapMultiSeedConfig(
            tuning_csv_path=str(tuning_csv),
            out_dir=str(out_dir),
            seeds=(260410, 260411),
            year_dataset_path=str(year_path),
            year_benchmark_label="custom_openml_year",
            higgs_dataset_path=str(higgs_path),
        )
    )
    assert payload["seeds"] == [260410, 260411]
    assert len(payload["seed_rows"]) == 4
    assert len(payload["aggregate_rows"]) == 2
    assert {str(r["Benchmark"]) for r in payload["seed_rows"]} == {
        "custom_openml_year",
        "higgs_public",
    }
    assert (out_dir / "multiseed_rows.csv").exists()
    assert (out_dir / "multiseed_summary.csv").exists()
    assert (out_dir / "multiseed_summary.json").exists()
