"""Download OpenML YearPredictionMSD into a CSV cache (one-shot materialization)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml


def materialize(cache_path: Path, *, name: str = "year", version: int = 1) -> Path:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.is_file():
        return cache_path
    bunch = fetch_openml(name=name, version=version, as_frame=True)
    features = bunch.data.copy()
    target = pd.to_numeric(bunch.target, errors="raise")
    frame = features.copy()
    frame["target"] = target.to_numpy()
    if cache_path.suffix.lower() == ".parquet":
        try:
            frame.to_parquet(cache_path, index=False)
        except ImportError:
            csv_fallback = cache_path.with_suffix(".csv")
            frame.to_csv(csv_fallback, index=False)
            return csv_fallback
    else:
        frame.to_csv(cache_path, index=False)
    return cache_path


def main() -> None:
    p = argparse.ArgumentParser(description="Materialize OpenML YearPredictionMSD to CSV.")
    p.add_argument(
        "--cache-path",
        type=Path,
        required=True,
        help="Output CSV path (e.g. data/paper/openml_year.csv)",
    )
    p.add_argument("--name", type=str, default="year")
    p.add_argument("--version", type=int, default=1)
    args = p.parse_args()
    out = materialize(args.cache_path, name=args.name, version=args.version)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
