from __future__ import annotations

import ast
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)


REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PAGES = [
    REPO_ROOT / "docs" / "usage" / "quickstart.md",
    REPO_ROOT / "docs" / "usage" / "practical_usage.md",
    REPO_ROOT / "docs" / "guides" / "method_selection_matrix.md",
    REPO_ROOT / "docs" / "guides" / "choosing_by_constraint.md",
    REPO_ROOT / "docs" / "guides" / "comparative_evidence_matrix.md",
]
PYTHON_FENCE_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


def _extract_python_snippets(text: str) -> list[str]:
    return [match.group(1).strip() for match in PYTHON_FENCE_RE.finditer(text)]


def _is_exec_import_node(node: ast.stmt) -> bool:
    if isinstance(node, ast.Import):
        return True
    if isinstance(node, ast.ImportFrom):
        return node.module is not None and (
            node.module.startswith("torchregress")
            or node.module.startswith("torch")
            or node.module.startswith("numpy")
            or node.module.startswith("matplotlib")
        )
    return False


def _import_statements(snippet: str) -> list[str]:
    module = ast.parse(snippet)
    lines = snippet.splitlines()
    statements: list[str] = []
    for node in module.body:
        if not _is_exec_import_node(node):
            continue
        if node.end_lineno is None:
            continue
        statements.append("\n".join(lines[node.lineno - 1 : node.end_lineno]))
    return statements


def test_onboarding_docs_python_snippets_compile_and_import_smoke() -> None:
    for page in DOC_PAGES:
        text = page.read_text(encoding="utf-8")
        snippets = _extract_python_snippets(text)
        assert snippets, f"No python snippets found in {page}"

        for idx, snippet in enumerate(snippets):
            compile(snippet, f"{page.name}::snippet{idx}", "exec")

            for stmt in _import_statements(snippet):
                namespace: dict[str, object] = {}
                exec(stmt, namespace, namespace)
