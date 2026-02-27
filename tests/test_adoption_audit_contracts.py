"""Non-regression checks for adoption-audit drift signals.

These tests intentionally use a baseline so they do not fail on existing debt.
They prevent the docs/examples/API drift from getting worse while the cleanup
roadmap is executed.
"""

from pathlib import Path

from tools import adoption_audit

BASELINE_DIR = Path(__file__).resolve().parent / "data" / "adoption_audit_baselines"


def _read_baseline(name: str) -> set[str]:
    path = BASELINE_DIR / name
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def test_docs_tr_symbol_refs_do_not_regress() -> None:
    current = {
        f"{i.path}|{i.module}|{i.symbol}|{i.kind}"
        for i in adoption_audit.find_invalid_doc_attr_refs()
    }
    baseline = _read_baseline("docs_invalid_attr_refs.txt")
    assert current <= baseline, sorted(current - baseline)


def test_docs_markdown_imports_do_not_regress() -> None:
    current = {
        f"{i.path}|{i.module}|{i.symbol}|{i.kind}"
        for i in adoption_audit.find_invalid_markdown_python_imports()
    }
    baseline = _read_baseline("docs_invalid_markdown_imports.txt")
    assert current <= baseline, sorted(current - baseline)


def test_docs_extras_refs_do_not_regress() -> None:
    current = {f"{i.path}|{i.extra}" for i in adoption_audit.find_invalid_extra_refs()}
    baseline = _read_baseline("docs_invalid_extras.txt")
    assert current <= baseline, sorted(current - baseline)


def test_examples_torchregress_imports_do_not_regress() -> None:
    current = {
        f"{i.path}|{i.module}|{i.symbol}|{i.kind}"
        for i in adoption_audit.find_invalid_example_imports()
    }
    baseline = _read_baseline("examples_invalid_imports.txt")
    assert current <= baseline, sorted(current - baseline)
