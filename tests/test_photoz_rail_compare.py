from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import photoz_rail_compare


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_merge_summaries_with_prediction_payloads() -> None:
    manifest = {
        "dataset_id": "dset",
        "split_id": "split",
        "core_baselines": ["flexzboost", "pzflow", "delight", "bpz"],
        "optional_baselines": ["lephare"],
    }
    tr_summary = {
        "artifact": "comparison_example_summary",
        "version": 1,
        "rows": [{"Method": "BinnedCE", "NMAD": 0.05}],
    }
    y_true = [0.1, 0.2, 0.4, 0.6]
    rail_payloads = []
    for method in ("flexzboost", "pzflow", "delight", "bpz"):
        rail_payloads.append(
            {
                "artifact": "rail_photoz_predictions",
                "dataset_id": "dset",
                "split_id": "split",
                "method": method,
                "bin_edges": [0.0, 0.3, 0.6, 0.9],
                "rows": [
                    {"id": i, "z_true": y, "z_point": y + 0.01, "pdf": [0.2, 0.6, 0.2]}
                    for i, y in enumerate(y_true)
                ],
            }
        )

    merged = photoz_rail_compare.merge_summaries(
        manifest=manifest,
        torchregress_summary=tr_summary,
        rail_payloads=rail_payloads,
        paper_parity=True,
    )
    assert merged["artifact"] == "comparison_example_summary"
    rows = merged["rows"]
    assert isinstance(rows, list)
    methods = {str(r.get("Method")) for r in rows if isinstance(r, dict)}
    assert {"BinnedCE", "flexzboost", "pzflow", "delight", "bpz"} <= methods
    rail_row = next(r for r in rows if isinstance(r, dict) and r.get("Method") == "flexzboost")
    assert rail_row["Framework"] == "RAIL"
    assert rail_row["CRPS"] is not None
    assert rail_row["PDF_NLL"] is not None
    assert rail_row["PITChi2"] is not None


def test_merge_summaries_raises_for_manifest_mismatch() -> None:
    manifest = {"dataset_id": "dset", "split_id": "split", "core_baselines": ["flexzboost"]}
    tr_summary = {"artifact": "comparison_example_summary", "rows": [{"Method": "A"}]}
    rail_payloads = [
        {
            "artifact": "rail_photoz_predictions",
            "dataset_id": "different",
            "split_id": "split",
            "method": "flexzboost",
            "rows": [{"z_true": 0.1, "z_point": 0.1}],
        }
    ]
    with pytest.raises(ValueError, match="dataset/split mismatch"):
        photoz_rail_compare.merge_summaries(
            manifest=manifest,
            torchregress_summary=tr_summary,
            rail_payloads=rail_payloads,
            paper_parity=True,
        )


def test_cli_round_trip_with_summary_payloads(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    tr_summary = tmp_path / "tr.json"
    rail_summary = tmp_path / "rail.json"
    out = tmp_path / "merged.json"
    _write(
        manifest,
        {
            "artifact": "rail_photoz_manifest",
            "dataset_id": "dset",
            "split_id": "split",
            "core_baselines": ["flexzboost", "pzflow", "delight", "bpz"],
            "optional_baselines": ["lephare"],
        },
    )
    _write(tr_summary, {"artifact": "comparison_example_summary", "rows": [{"Method": "BinnedCE"}]})
    _write(
        rail_summary,
        {
            "artifact": "rail_photoz_summary",
            "dataset_id": "dset",
            "split_id": "split",
            "rows": [
                {"Method": "flexzboost", "NMAD": 0.05},
                {"Method": "pzflow", "NMAD": 0.05},
                {"Method": "delight", "NMAD": 0.05},
                {"Method": "bpz", "NMAD": 0.05},
            ],
        },
    )

    merged = photoz_rail_compare.merge_summaries(
        manifest=json.loads(manifest.read_text(encoding="utf-8")),
        torchregress_summary=json.loads(tr_summary.read_text(encoding="utf-8")),
        rail_payloads=[json.loads(rail_summary.read_text(encoding="utf-8"))],
        paper_parity=True,
    )
    _write(out, merged)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    methods = {row["Method"] for row in payload["rows"]}
    assert {"BinnedCE", "flexzboost", "pzflow", "delight", "bpz"} <= methods
