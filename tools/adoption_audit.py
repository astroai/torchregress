"""Adoption-readiness audit tooling for torchregress.

This script provides non-mutating repository audits used to:
- inventory public exports
- detect docs/API drift (markdown `tr.*` references and extras)
- detect broken torchregress imports in examples/docs python files
- summarize example coverage and comparability signals

It can be used interactively and from tests.
"""

import argparse
import ast
import importlib
import json
import re
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
EXAMPLES_DIR = REPO_ROOT / "examples"
TESTS_DIR = REPO_ROOT / "tests"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

TOP_MODULES = ("losses", "metrics", "ensemble", "algorithms", "utils", "viz")
DOC_ATTR_REF_RE = re.compile(
    r"\btr\.(losses|metrics|ensemble|algorithms|utils|viz)\.([A-Za-z_][A-Za-z0-9_]*)\b"
)
EXTRA_REF_RE = re.compile(r"torchregress\[([^\]]+)\]")
PYTHON_FENCE_RE = re.compile(r"```(?:python|py)\n(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class SymbolRefIssue:
    path: str
    module: str
    symbol: str
    kind: str  # "markdown_attr" | "python_import"


@dataclass(frozen=True)
class ExtraIssue:
    path: str
    extra: str


@dataclass(frozen=True)
class ImportIssue:
    path: str
    module: str
    symbol: str
    kind: str  # "missing_symbol" | "import_error"
    detail: str


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _load_pyproject() -> dict[str, Any]:
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


def get_valid_extras() -> set[str]:
    pyproject = _load_pyproject()
    return set(pyproject.get("project", {}).get("optional-dependencies", {}).keys())


def _import_torchregress_modules() -> dict[str, Any]:
    import torchregress as tr

    return {
        "torchregress": tr,
        "losses": tr.losses,
        "metrics": tr.metrics,
        "ensemble": tr.ensemble,
        "algorithms": tr.algorithms,
        "utils": tr.utils,
        "viz": tr.viz,
    }


def inventory_public_exports() -> dict[str, list[dict[str, str]]]:
    modules = _import_torchregress_modules()
    inventory: dict[str, list[dict[str, str]]] = {}
    for module_name, module in modules.items():
        names = getattr(module, "__all__", None)
        if names is None:
            names = [n for n in dir(module) if not n.startswith("_")]
        rows: list[dict[str, str]] = []
        for name in sorted(set(names)):
            kind = "missing"
            qualname = ""
            if hasattr(module, name):
                obj = getattr(module, name)
                kind = type(obj).__name__
                qualname = getattr(obj, "__module__", "")
            rows.append({"name": name, "kind": kind, "object_module": qualname})
        inventory[module_name] = rows
    return inventory


def _iter_markdown_files() -> Iterable[Path]:
    yield REPO_ROOT / "README.md"
    if DOCS_DIR.exists():
        yield from sorted(DOCS_DIR.rglob("*.md"))


def find_invalid_doc_attr_refs() -> list[SymbolRefIssue]:
    modules = _import_torchregress_modules()
    issues: list[SymbolRefIssue] = []
    for path in _iter_markdown_files():
        text = path.read_text(encoding="utf-8")
        for match in DOC_ATTR_REF_RE.finditer(text):
            module_name, symbol = match.groups()
            module = modules[module_name]
            if not hasattr(module, symbol):
                issues.append(
                    SymbolRefIssue(
                        path=_rel(path),
                        module=module_name,
                        symbol=symbol,
                        kind="markdown_attr",
                    )
                )
    return sorted(issues, key=lambda i: (i.path, i.module, i.symbol, i.kind))


def find_invalid_extra_refs() -> list[ExtraIssue]:
    valid = get_valid_extras()
    issues: list[ExtraIssue] = []
    for path in _iter_markdown_files():
        text = path.read_text(encoding="utf-8")
        for extra in EXTRA_REF_RE.findall(text):
            if extra not in valid:
                issues.append(ExtraIssue(path=_rel(path), extra=extra))
    return sorted(set(issues), key=lambda i: (i.path, i.extra))


def _iter_python_files_for_import_audit() -> Iterable[Path]:
    if EXAMPLES_DIR.exists():
        yield from sorted(EXAMPLES_DIR.rglob("*.py"))


def _iter_markdown_python_blocks(path: Path) -> Iterable[tuple[int, str]]:
    text = path.read_text(encoding="utf-8")
    for idx, match in enumerate(PYTHON_FENCE_RE.finditer(text), start=1):
        yield idx, match.group(1)


def _validate_import(module_name: str, symbol: str) -> tuple[str, str] | None:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - exercised by runtime env
        return ("import_error", f"{type(exc).__name__}: {exc}")

    if symbol == "*":
        return None
    if not hasattr(module, symbol):
        return ("missing_symbol", "attribute not found")
    return None


def _scan_python_source_imports(path_label: str, source: str) -> list[ImportIssue]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        # Markdown python fences often include pseudocode / partial snippets.
        # Ignore parse errors here; this audit focuses on import drift.
        _ = exc
        return []

    issues: list[ImportIssue] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module is None:
            continue
        if not node.module.startswith("torchregress"):
            continue
        for alias in node.names:
            symbol = alias.name
            result = _validate_import(node.module, symbol)
            if result is None:
                continue
            kind, detail = result
            issues.append(
                ImportIssue(
                    path=path_label,
                    module=node.module,
                    symbol=symbol,
                    kind=kind,
                    detail=detail,
                )
            )
    return issues


def find_invalid_example_imports() -> list[ImportIssue]:
    issues: list[ImportIssue] = []
    for path in _iter_python_files_for_import_audit():
        issues.extend(_scan_python_source_imports(_rel(path), path.read_text(encoding="utf-8")))
    return sorted(issues, key=lambda i: (i.path, i.module, i.symbol))


def find_invalid_markdown_python_imports() -> list[ImportIssue]:
    issues: list[ImportIssue] = []
    for path in _iter_markdown_files():
        for block_idx, source in _iter_markdown_python_blocks(path):
            label = f"{_rel(path)}::python_block_{block_idx}"
            issues.extend(_scan_python_source_imports(label, source))
    return sorted(issues, key=lambda i: (i.path, i.module, i.symbol))


def summarize_examples() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    topic_totals: dict[str, int] = {
        "noisy_labels": 0,
        "noisy_features_eiv": 0,
        "imbalance": 0,
        "calibration": 0,
        "ood": 0,
        "multimodal": 0,
        "multi_target": 0,
        "uncertainty": 0,
        "benchmark_or_compare": 0,
    }

    for path in sorted(EXAMPLES_DIR.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        lower = text.lower()
        name_lower = path.name.lower()

        def has_any(*terms: str) -> bool:
            return any(t in lower or t in name_lower for t in terms)

        topics = {
            "noisy_labels": has_any("noisy label", "label noise", "coteaching", "rent"),
            "noisy_features_eiv": has_any(
                "eiv", "simex", "regression calibration", "feature noise"
            ),
            "imbalance": has_any("imbalanc", "tail_extremes", "densityweightedloss", "ldsloss"),
            "calibration": has_any("calibration", "ece", "reliability", "conformal"),
            "ood": has_any("ood", "out-of-distribution", "typicality", "mahalanobis"),
            "multimodal": has_any("mixture density", "mdn", "normalizing flow", "multimodal"),
            "multi_target": has_any(
                "multitarget", "multi-target", "multi target", "full covariance"
            ),
            "uncertainty": has_any("uncertainty", "variance", "epistemic", "aleatoric"),
            "benchmark_or_compare": has_any("benchmark", "compare", "comparison", "sweep"),
        }

        for topic, present in topics.items():
            if present:
                topic_totals[topic] += 1

        comparability_signals = {
            "seeded": bool(
                re.search(
                    r"(manual_seed|np\.random\.seed|set_seed|set_all_seeds)",
                    text,
                )
            ),
            "prints_metrics": bool(re.search(r"\b(rmse|mae|mse|ece|picp|crps)\b", lower)),
            "compares_multiple_methods": bool(
                re.search(r"(losses_to_compare|methods\s*=|compare.*approach|comparison)", lower)
            ),
            "uses_argparse": "argparse" in lower or "parser = argparse" in lower,
        }

        rows.append(
            {
                "path": _rel(path),
                "topics": [k for k, v in topics.items() if v],
                "comparability_signals": comparability_signals,
            }
        )

    return {"examples": rows, "topic_totals": topic_totals, "count": len(rows)}


def summarize_direct_test_file_coverage() -> dict[str, Any]:
    source_files = [
        p
        for p in (REPO_ROOT / "torchregress").rglob("*.py")
        if p.name != "__init__.py" and "__pycache__" not in p.parts
    ]
    test_names = {p.name for p in TESTS_DIR.rglob("test_*.py")}
    rows: list[dict[str, str]] = []
    for path in sorted(source_files):
        direct = f"test_{path.stem}.py" in test_names
        rows.append({"path": _rel(path), "direct_test": "yes" if direct else "no"})
    direct_count = sum(1 for row in rows if row["direct_test"] == "yes")
    return {
        "count": len(rows),
        "direct_count": direct_count,
        "direct_fraction": (direct_count / len(rows)) if rows else 0.0,
        "rows": rows,
    }


def run_full_audit() -> dict[str, Any]:
    invalid_doc_attr = find_invalid_doc_attr_refs()
    invalid_doc_imports = find_invalid_markdown_python_imports()
    invalid_extras = find_invalid_extra_refs()
    invalid_example_imports = find_invalid_example_imports()
    return {
        "public_exports": inventory_public_exports(),
        "docs": {
            "invalid_attr_refs": [asdict(i) for i in invalid_doc_attr],
            "invalid_python_imports": [asdict(i) for i in invalid_doc_imports],
            "invalid_extras": [asdict(i) for i in invalid_extras],
            "counts": {
                "invalid_attr_refs": len(invalid_doc_attr),
                "invalid_python_imports": len(invalid_doc_imports),
                "invalid_extras": len(invalid_extras),
            },
        },
        "examples": {
            "invalid_imports": [asdict(i) for i in invalid_example_imports],
            "counts": {"invalid_imports": len(invalid_example_imports)},
            **summarize_examples(),
        },
        "tests": {"direct_file_coverage": summarize_direct_test_file_coverage()},
        "environment": {"python": sys.version.split()[0]},
    }


def _write_json(data: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _format_issue_lines(items: Iterable[Any], formatter: Any) -> str:
    return "\n".join(formatter(item) for item in items)


def _read_baseline_lines(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def _check_no_regression(current: set[str], baseline_path: Path) -> tuple[bool, list[str]]:
    baseline = _read_baseline_lines(baseline_path)
    new_items = sorted(current - baseline)
    return (len(new_items) == 0), new_items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run adoption audit checks for torchregress.")
    parser.add_argument("--json", type=Path, help="Write full audit JSON to this path.")
    parser.add_argument("--print-summary", action="store_true", help="Print concise summary.")
    parser.add_argument(
        "--check-doc-attr-baseline",
        type=Path,
        help="No-regression baseline for docs `tr.*` refs.",
    )
    parser.add_argument(
        "--check-doc-import-baseline",
        type=Path,
        help="No-regression baseline for markdown fenced python import issues.",
    )
    parser.add_argument(
        "--check-extra-baseline",
        type=Path,
        help="No-regression baseline for invalid extras refs.",
    )
    parser.add_argument(
        "--check-example-import-baseline",
        type=Path,
        help="No-regression baseline for example import issues.",
    )
    args = parser.parse_args(argv)

    data = run_full_audit()

    if args.json:
        _write_json(data, args.json)

    if args.print_summary:
        print("docs.invalid_attr_refs", data["docs"]["counts"]["invalid_attr_refs"])
        print("docs.invalid_python_imports", data["docs"]["counts"]["invalid_python_imports"])
        print("docs.invalid_extras", data["docs"]["counts"]["invalid_extras"])
        print("examples.invalid_imports", data["examples"]["counts"]["invalid_imports"])
        print("examples.count", data["examples"]["count"])
        print(
            "tests.direct_file_coverage",
            data["tests"]["direct_file_coverage"]["direct_fraction"],
        )

    failures: list[str] = []
    if args.check_doc_attr_baseline:
        current = {
            f"{i['path']}|{i['module']}|{i['symbol']}|{i['kind']}"
            for i in data["docs"]["invalid_attr_refs"]
        }
        ok, new_items = _check_no_regression(current, args.check_doc_attr_baseline)
        if not ok:
            failures.append(
                "New invalid docs attr refs:\n" + "\n".join(f"  {line}" for line in new_items)
            )
    if args.check_doc_import_baseline:
        current = {
            f"{i['path']}|{i['module']}|{i['symbol']}|{i['kind']}"
            for i in data["docs"]["invalid_python_imports"]
        }
        ok, new_items = _check_no_regression(current, args.check_doc_import_baseline)
        if not ok:
            failures.append(
                "New invalid markdown python imports:\n"
                + "\n".join(f"  {line}" for line in new_items)
            )
    if args.check_extra_baseline:
        current = {f"{i['path']}|{i['extra']}" for i in data["docs"]["invalid_extras"]}
        ok, new_items = _check_no_regression(current, args.check_extra_baseline)
        if not ok:
            failures.append(
                "New invalid extras refs:\n" + "\n".join(f"  {line}" for line in new_items)
            )
    if args.check_example_import_baseline:
        current = {
            f"{i['path']}|{i['module']}|{i['symbol']}|{i['kind']}"
            for i in data["examples"]["invalid_imports"]
        }
        ok, new_items = _check_no_regression(current, args.check_example_import_baseline)
        if not ok:
            failures.append(
                "New invalid example imports:\n" + "\n".join(f"  {line}" for line in new_items)
            )

    if failures:
        print("\n\n".join(failures), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
