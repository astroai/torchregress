"""Before/after benchmark comparison via git stash.

Workflow:
  1. Verify the listed paths have uncommitted changes (else abort early).
  2. ``git stash push -m compare-<label> -- <paths>`` to roll back to HEAD.
  3. Run every requested benchmark module -> capture "before" timings.
  4. ``git stash pop`` to restore the refactor.
  5. Run every requested benchmark again -> capture "after" timings.
  6. Parse median µs from each benchmark's printed output, compute
     per-metric deltas, write ``reports/benchmarks/compare_<label>.json``
     and a small stdout summary.

The report is the durable artifact; the script's stdout is a glance-able
summary table.

Usage:
    python -m tools.benchmarks.compare_against_baseline
    python -m tools.benchmarks.compare_against_baseline --paths src/torchregress/losses/gaussian.py
    python -m tools.benchmarks.compare_against_baseline --benchmarks bench_gaussian
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # tools/benchmarks -> tools -> project root

DEFAULT_PATHS = [
    "src/torchregress/losses/base.py",
    "src/torchregress/losses/gaussian.py",
]
DEFAULT_BENCHMARKS = ("bench_gaussian", "profile_mvn", "profile_compile")

# Matches the "median X us(/iter) (min ..." line printed by tools.benchmarks.<x>.bench().
# Different bench modules print slightly different units (some append
# ``/iter``); the ``us`` token alone is the reliable anchor.
_MEDIAN_LINE = re.compile(
    r"^\s+(?P<label>.+?)\s+median\s+(?P<median>[\d.]+)\s+us(?:/iter)?\s+(?:\(min|\(|$)",
    re.MULTILINE,
)


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess, echo the command, raise on failure."""
    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
        **kwargs,
    )


def _run_quiet(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a subprocess, suppressing output unless it fails."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def _has_uncommitted_changes(paths: list[str]) -> bool:
    """True iff any of ``paths`` differs from HEAD in the working tree."""
    res = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *paths],
        cwd=ROOT,
    )
    return res.returncode != 0


def _parse_bench_output(stdout: str) -> dict[str, float]:
    """Pull every ``median X us`` row out of a benchmark's stdout."""
    out: dict[str, float] = {}
    for match in _MEDIAN_LINE.finditer(stdout):
        out[match.group("label").strip()] = float(match.group("median"))
    return out


def _run_bench(name: str, label: str) -> dict[str, float]:
    """Run ``tools.benchmarks.<name>`` and parse the median timings."""
    res = _run([sys.executable, "-m", f"tools.benchmarks.{name}", "--label", label])
    parsed = _parse_bench_output(res.stdout)
    if not parsed:
        # Surface the raw output for debugging; the regex is a likely
        # culprit if a benchmark changes its print format.
        print(
            f"  [warn] no median rows parsed from {name}; first 5 lines:\n"
            + "\n".join(res.stdout.splitlines()[:5]),
            file=sys.stderr,
        )
    return parsed


def _git_state() -> dict[str, str]:
    head = _run_quiet(["git", "rev-parse", "HEAD"]).stdout.strip()
    branch = _run_quiet(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    return {"head": head, "branch": branch}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--paths",
        nargs="+",
        default=DEFAULT_PATHS,
        help="Files to stash for the 'before' run (default: losses/base.py + losses/gaussian.py).",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(DEFAULT_BENCHMARKS),
        help="Benchmark module names under tools.benchmarks (without .py).",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/benchmarks",
        help="Directory to write the JSON report to.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Label for this comparison (default: YYYY-MM-DD_HHMMSS).",
    )
    parser.add_argument(
        "--no-stash",
        action="store_true",
        help="Skip the stash/pop dance; run 'after' only (useful when HEAD is already the baseline).",
    )
    args = parser.parse_args()

    label = args.label or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    has_changes = _has_uncommitted_changes(args.paths)
    if not has_changes and not args.no_stash:
        print(
            f"No uncommitted changes in {args.paths}; nothing to compare. "
            "Re-run with --no-stash to capture the current state as 'after'.",
            file=sys.stderr,
        )
        return 1

    report: dict = {
        "label": label,
        "git": {**_git_state(), "paths": args.paths},
        "has_changes": has_changes,
        "benchmarks": {},
    }

    stash_name = f"compare-{label}"
    if args.no_stash:
        for bench_name in args.benchmarks:
            report["benchmarks"][bench_name] = {
                "before": None,
                "after": _run_bench(bench_name, f"{label}-current"),
            }
    else:
        _run(["git", "stash", "push", "-m", stash_name, "--", *args.paths])
        try:
            for bench_name in args.benchmarks:
                print(f"\n>>> running {bench_name} (BEFORE)", file=sys.stderr)
                report["benchmarks"][bench_name] = {
                    "before": _run_bench(bench_name, f"{label}-before"),
                }
        finally:
            _run(["git", "stash", "pop"])

        for bench_name in args.benchmarks:
            print(f"\n>>> running {bench_name} (AFTER)", file=sys.stderr)
            report["benchmarks"][bench_name]["after"] = _run_bench(bench_name, f"{label}-after")

    # Per-metric delta in percent.
    for bench_name, bench_data in report["benchmarks"].items():
        before = bench_data.get("before")
        after = bench_data.get("after")
        if not before or not after:
            continue
        deltas: dict[str, float] = {}
        for metric, before_us in before.items():
            after_us = after.get(metric)
            if after_us is None or before_us == 0:
                continue
            deltas[metric] = round((after_us - before_us) / before_us * 100.0, 2)
        bench_data["delta_pct"] = deltas

    out_path = out_dir / f"compare_{label}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        rel = out_path.relative_to(ROOT)
    except ValueError:
        # Out-of-tree output dir (e.g. a test using tmp_path).  Print the
        # absolute path so the operator still knows where the report landed.
        rel = out_path
    print(f"\nWrote {rel}", file=sys.stderr)

    # Stdout summary.
    print(f"\n=== Comparison: {label} ===")
    for bench_name, bench_data in report["benchmarks"].items():
        deltas = bench_data.get("delta_pct")
        if not deltas:
            print(f"\n{bench_name}: (no before/after deltas)")
            continue
        print(f"\n{bench_name}:")
        for metric, delta in sorted(deltas.items(), key=lambda kv: -abs(kv[1])):
            arrow = "↓" if delta < 0 else "↑"
            print(f"  {arrow} {delta:+6.2f}%   {metric}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
