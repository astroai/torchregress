"""Small Stage-A prior-ratio clip sweep on SPT Year track (writes stage_a_sweep.json)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from dataclasses import replace  # noqa: E402

import spt_reg_year_comparison as ycmp  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--clips",
        type=float,
        nargs="+",
        default=[1.5, 2.0, 3.0, 5.0],
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Shrink pools for smoke")
    args = parser.parse_args()

    base = ycmp.SPTRegYearConfig(
        cache_path=str(args.cache_path),
        allow_download=bool(args.allow_download),
    )
    if args.quick:
        base = replace(
            base,
            n_source=256,
            n_target_unlabeled=128,
            n_target_cal=64,
            n_target_test=64,
            target_label_budget=32,
            n_support=64,
        )

    results: list[dict[str, object]] = []
    for clip in args.clips:
        cfg = replace(base, prior_ratio_clip=float(clip))
        rows, _ = ycmp.run_comparison(cfg)
        by_m = {str(r["Method"]): r for r in rows}
        sg = by_m.get("SPTRegGaussian", {})
        results.append(
            {
                "prior_ratio_clip": float(clip),
                "SPTRegGaussian_NLL": sg.get("NLL"),
                "SPTRegGaussian_CRPS": sg.get("CRPS"),
                "SPTRegGaussian_Cov90": sg.get("Cov90"),
                "SourceGaussian_NLL": by_m.get("SourceGaussian", {}).get("NLL"),
            }
        )
        print(f"clip={clip} SPTRegGaussian NLL={sg.get('NLL')}", flush=True)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rows": results, "cache_path": str(args.cache_path)}
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
