"""Run all benchmarks or a selected subset.

Usage:
    python -m tools.benchmarks                # run every benchmark
    python -m tools.benchmarks bench_gaussian # run a single one
    python -m tools.benchmarks bench_gaussian profile_mvn
"""

import importlib
import sys

_BENCHMARKS = (
    "bench_gaussian",
    "profile_mvn",
    "profile_compile",
    "compare_against_baseline",
)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    selected = argv or list(_BENCHMARKS)
    for name in selected:
        if name not in _BENCHMARKS:
            print(f"unknown benchmark: {name!r}; choices: {_BENCHMARKS}", file=sys.stderr)
            return 2
        if name == "compare_against_baseline":
            # compare_against_baseline owns its own subcommand-style CLI
            # and stash/bench/restore workflow; just hand control over.
            return importlib.import_module(f"tools.benchmarks.{name}").main()
        print(f"\n========== tools.benchmarks.{name} ==========")
        importlib.import_module(f"tools.benchmarks.{name}").main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
