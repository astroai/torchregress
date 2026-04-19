from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_merge_mod():
    path = REPO / "tools" / "merge_ssl_official_metrics.py"
    spec = importlib.util.spec_from_file_location("_merge_ssl", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("merge spec")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_merge_computes_deltas(tmp_path: Path) -> None:
    merge = _load_merge_mod()
    ours = tmp_path / "ours.csv"
    with ours.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(
            handle,
            fieldnames=["Method", "Seed", "UnlabeledFraction", "RMSE", "NLL", "CRPS"],
        )
        w.writeheader()
        w.writerow(
            {
                "Method": "RankUp",
                "Seed": "7",
                "UnlabeledFraction": "1.0",
                "RMSE": "1.0",
                "NLL": "2.0",
                "CRPS": "0.5",
            }
        )
    official = tmp_path / "off.json"
    official.write_text(
        json.dumps(
            [
                {
                    "track": "rankup_official",
                    "method": "RankUp",
                    "seed": 7,
                    "RMSE": 0.8,
                    "NLL": 2.2,
                    "CRPS": 0.4,
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    out_j = tmp_path / "m.json"
    out_c = tmp_path / "m.csv"
    merge.merge(
        ours_csv=ours,
        official_json=official,
        methods={"RankUp"},
        seed=7,
        unlabeled_fraction=1.0,
        out_json=out_j,
        out_csv=out_c,
    )
    payload = json.loads(out_j.read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 1
    row = payload["rows"][0]
    assert row["delta_RMSE_ours_minus_official"] == pytest.approx(0.2)
    assert row["delta_NLL_ours_minus_official"] == pytest.approx(-0.2)
