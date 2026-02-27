"""Parse mypy output and summarize type-checking debt by package/module."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MYPY_LINE_RE = re.compile(
    r"^(?P<path>[^:\n]+):(?P<line>\d+): error: (?P<message>.*?)(?:\s+\[(?P<code>[^\]]+)\])?$"
)


@dataclass(frozen=True)
class MypyError:
    path: str
    line: int
    message: str
    code: str | None
    package: str
    module_group: str


def _classify_path(path: str) -> tuple[str, str]:
    norm = path.replace("\\", "/")
    if "torchregress/" not in norm:
        return ("other", "other")
    rel = norm.split("torchregress/", 1)[1]
    parts = rel.split("/")
    if len(parts) <= 1:
        stem = parts[0].rsplit(".", 1)[0]
        return ("root", stem)
    package = parts[0]
    return (package, package)


def parse_mypy_output(text: str) -> list[MypyError]:
    errors: list[MypyError] = []
    for raw_line in text.splitlines():
        match = MYPY_LINE_RE.match(raw_line.strip())
        if not match:
            continue
        path = match.group("path")
        package, module_group = _classify_path(path)
        errors.append(
            MypyError(
                path=path,
                line=int(match.group("line")),
                message=match.group("message"),
                code=match.group("code"),
                package=package,
                module_group=module_group,
            )
        )
    return errors


def summarize_errors(errors: list[MypyError]) -> dict[str, Any]:
    by_package = Counter(err.package for err in errors)
    by_code = Counter((err.code or "unknown") for err in errors)
    by_file = Counter(err.path for err in errors)

    package_codes: dict[str, Counter[str]] = defaultdict(Counter)
    for err in errors:
        package_codes[err.package][err.code or "unknown"] += 1

    return {
        "total_errors": len(errors),
        "packages": dict(sorted(by_package.items(), key=lambda kv: (-kv[1], kv[0]))),
        "error_codes": dict(sorted(by_code.items(), key=lambda kv: (-kv[1], kv[0]))),
        "top_files": [
            {"path": path, "count": count}
            for path, count in sorted(by_file.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
        ],
        "package_error_codes": {
            pkg: dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))
            for pkg, counter in sorted(package_codes.items(), key=lambda kv: kv[0])
        },
    }


def build_report(text: str) -> dict[str, Any]:
    errors = parse_mypy_output(text)
    summary = summarize_errors(errors)
    return {
        "artifact": "mypy_triage",
        "version": 1,
        "summary": summary,
        "errors": [asdict(e) for e in errors],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize mypy output by package/error code.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to raw mypy stdout/stderr text",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to JSON summary report",
    )
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8")
    report = build_report(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote mypy triage report: {args.output}")
    print(f"Total errors: {report['summary']['total_errors']}")
    print(f"Packages: {report['summary']['packages']}")


if __name__ == "__main__":
    main()
