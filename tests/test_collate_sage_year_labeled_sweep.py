from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.collate_sage_year_labeled_sweep import collate


def test_collate_year_labeled_sweep(tmp_path: Path) -> None:
    def one_summary(nl: int, nll_s: float, nll_g: float) -> None:
        p = tmp_path / f"year_direct_nl{nl}_summary.json"
        p.write_text(
            json.dumps(
                {
                    "config": {"n_unlabeled": 100, "n_test": 50, "n_labeled": nl},
                    "rows": [
                        {
                            "Method": "SupervisedOnly",
                            "UnlabeledFraction": 1.0,
                            "NLL": nll_s,
                            "CRPS": 0.1,
                            "Cov90": 0.9,
                            "CalibMAE": 0.05,
                            "RMSE": 1.0,
                        },
                        {
                            "Method": "SAGE-Reg",
                            "UnlabeledFraction": 1.0,
                            "NLL": nll_g,
                            "CRPS": 0.11,
                            "Cov90": 0.85,
                            "CalibMAE": 0.06,
                            "RMSE": 1.01,
                        },
                        {
                            "Method": "ConfidenceWeightedPseudoLabel",
                            "UnlabeledFraction": 1.0,
                            "NLL": 9.0,
                            "CRPS": 0.2,
                            "Cov90": 0.5,
                            "CalibMAE": 0.2,
                            "RMSE": 1.1,
                        },
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    one_summary(2048, 3.0, 2.5)
    one_summary(4096, 2.0, 2.1)
    paths = sorted(tmp_path.glob("year_direct_nl*_summary.json"))
    report = collate(paths)
    assert report["artifact"] == "sage_year_labeled_sweep_collated"
    assert len(report["rows"]) == 2
    assert report["rows"][0]["n_labeled"] == 2048
    assert report["rows"][0]["NLL_SAGEMinusSupervised"] == -0.5
    assert report["rows"][1]["NLL_SAGEMinusSupervised"] == pytest.approx(0.1)
