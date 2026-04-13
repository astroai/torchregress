#!/usr/bin/env python3
"""Materialize a large OpenML regression table to Parquet/CSV (default: diamonds, id 42225).

Use this for a **pinned** cache when you want reproducible bytes independent of live
OpenML fetches. Falls back to ``torchregress.utils.openml_relaxed`` if sklearn's MD5
check fails.

Example::

    uv run python tools/materialize_openml_large_tabular.py \\
      --cache-path data/paper/openml_large_tabular_diamonds.parquet \\
      --max-rows 60000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from torchregress.utils.openml_relaxed import fetch_openml_regression_with_sklearn_fallback


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cache-path",
        type=Path,
        required=True,
        help="Output .parquet or .csv path",
    )
    p.add_argument(
        "--data-id",
        type=int,
        default=42225,
        help="OpenML data id (default: 42225, ggplot2 diamonds / price regression)",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Subsample this many rows after load (deterministic seed 0)",
    )
    p.add_argument("--target-column", type=str, default="target")
    args = p.parse_args()

    frame, tag = fetch_openml_regression_with_sklearn_fallback(
        data_id=int(args.data_id),
        name=None,
        version=1,
        target_column=str(args.target_column),
    )
    if args.max_rows is not None and len(frame) > int(args.max_rows):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(frame), size=int(args.max_rows), replace=False)
        frame = frame.iloc[np.sort(idx)].reset_index(drop=True)

    out = args.cache_path.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".parquet":
        frame.to_parquet(out, index=False)
    else:
        frame.to_csv(out, index=False)
    print(f"Wrote {out} ({len(frame)} rows) tag={tag}")


if __name__ == "__main__":
    main()
