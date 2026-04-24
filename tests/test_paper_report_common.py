from __future__ import annotations

from tools import paper_report_common as prc


def test_summarize_multiseed_includes_gap_bootstrap() -> None:
    payload = {
        "tuning_csv_path": "sweep.csv",
        "seeds": [1, 2, 3],
        "aggregate_rows": [],
        "seed_rows": [
            {
                "Benchmark": "year",
                "Seed": 1,
                "SAGEMinusSupervised": -0.05,
                "ConfidenceMinusSupervised": 0.1,
            },
            {
                "Benchmark": "year",
                "Seed": 2,
                "SAGEMinusSupervised": -0.15,
                "ConfidenceMinusSupervised": 0.2,
            },
            {
                "Benchmark": "year",
                "Seed": 3,
                "SAGEMinusSupervised": -0.1,
                "ConfidenceMinusSupervised": 0.15,
            },
        ],
    }
    summary = prc.summarize_multiseed(
        payload,
        multiseed_bootstrap_n=400,
        multiseed_bootstrap_seed=0,
    )
    assert summary["gap_bootstrap_meta"]["n"] == 400
    assert len(summary["gap_bootstrap_95"]) == 1
    row = summary["gap_bootstrap_95"][0]
    assert row["Benchmark"] == "year"
    assert row["SAGEMeanGapBoot95Low"] <= row["SAGEMeanGapBoot95High"]
    assert row["ConfidenceMeanGapBoot95Low"] <= row["ConfidenceMeanGapBoot95High"]


def test_summarize_multiseed_bootstrap_skips_incomplete_gap_rows() -> None:
    payload = {
        "aggregate_rows": [],
        "seed_rows": [
            {
                "Benchmark": "year",
                "Seed": 1,
                "SAGEMinusSupervised": -0.05,
                "ConfidenceMinusSupervised": 0.1,
            },
            {
                "Benchmark": "year",
                "Seed": 2,
                "SAGEMinusSupervised": None,
                "ConfidenceMinusSupervised": 0.2,
            },
            {
                "Benchmark": "diamonds",
                "Seed": 1,
                "SAGEMinusSupervised": "nan",
                "ConfidenceMinusSupervised": 0.2,
            },
        ],
    }

    summary = prc.summarize_multiseed(payload, multiseed_bootstrap_n=50)

    assert [row["Benchmark"] for row in summary["gap_bootstrap_95"]] == ["year"]
    row = summary["gap_bootstrap_95"][0]
    assert row["SAGEMeanGapBoot95Low"] == -0.05
    assert row["SAGEMeanGapBoot95High"] == -0.05
