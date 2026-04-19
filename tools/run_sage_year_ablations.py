"""Grid over SAGE batch-relative / trust-top-k on Year (writes per-cell summary JSON)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--year-cache", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    modes: list[str | None] = [None, "zscore"]
    topks: list[int | None] = [None, 16]
    if args.quick:
        modes = [None]
        topks = [None]

    rows: list[dict[str, object]] = []
    for mode in modes:
        for topk in topks:
            tag = f"br_{mode or 'none'}_tk_{topk or 'none'}"
            summ = args.out_dir / f"summary_{tag}.json"
            cmd = [
                sys.executable,
                str(repo / "examples/benchmarks/self_agreement_realdata_year.py"),
                "--cache-path",
                str(args.year_cache),
                "--no-download",
                "--n-labeled",
                "2048" if args.quick else "4096",
                "--n-unlabeled",
                "8192" if args.quick else "131072",
                "--n-test",
                "1024" if args.quick else "32768",
                "--teacher-epochs",
                "1" if args.quick else "32",
                "--student-epochs",
                "1" if args.quick else "32",
                "--unlabeled-fractions",
                "1.0",
                "--summary-json-path",
                str(summ),
            ]
            if mode:
                cmd += ["--sage-batch-relative-mode", mode]
            if topk is not None:
                cmd += ["--sage-batch-trust-top-k", str(topk)]
            print("==", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=str(repo), check=True)
            payload = json.loads(summ.read_text(encoding="utf-8"))
            sage_nll = None
            for r in payload.get("rows", []):
                if r.get("Method") == "SAGE-Reg" and float(r.get("UnlabeledFraction", 0)) == 1.0:
                    sage_nll = r.get("NLL")
                    break
            rows.append(
                {
                    "sage_batch_relative_mode": mode,
                    "sage_batch_trust_top_k": topk,
                    "SAGE_NLL_ufrac1": sage_nll,
                    "summary_path": str(summ),
                }
            )

    out = args.out_dir / "summary.json"
    out.write_text(
        json.dumps({"artifact": "sage_year_ablations", "version": 1, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
