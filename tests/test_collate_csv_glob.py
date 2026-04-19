from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_collate():
    path = REPO / "tools" / "collate_csv_glob.py"
    spec = importlib.util.spec_from_file_location("_collate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("collate spec")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_collate_paths_concat(tmp_path: Path) -> None:
    collate = _load_collate()
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    for p, sid in ((a, "1"), (b, "2")):
        with p.open("w", newline="", encoding="utf-8") as handle:
            w = csv.DictWriter(handle, fieldnames=["Seed", "Method", "RMSE"])
            w.writeheader()
            w.writerow({"Seed": sid, "Method": "RankUp", "RMSE": "0." + sid})
    out = tmp_path / "all.csv"
    assert collate.collate_paths([a, b], out) == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 2
