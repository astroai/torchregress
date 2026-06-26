from __future__ import annotations

import ast
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)


REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PAGES = [
    REPO_ROOT / "docs" / "getting-started" / "quickstart.md",
    REPO_ROOT / "docs" / "guide" / "practical-usage.md",
    REPO_ROOT / "docs" / "guide" / "method-selection.md",
    REPO_ROOT / "docs" / "guide" / "choosing-by-constraint.md",
    REPO_ROOT / "docs" / "reports" / "comparative_evidence_matrix.md",
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


def test_quickstart_workflows_run_without_errors() -> None:
    """Verify the quickstart.md code examples actually run without errors."""
    import torch
    import torch.nn as nn

    from torchregress.ensemble import DeepEnsemble
    from torchregress.losses import (
        AdaptiveRobustLoss,
        CauchyLoss,
        CQR,
        GaussianNLLLoss,
        SplitConformal,
        TukeyBiweightLoss,
        WeightedHuberLoss,
    )
    from torchregress.metrics import (
        crps_gaussian,
        gaussian_nll,
        mae,
        r2_score,
        rmse,
        uncertainty_decomposition,
    )

    B, D = 32, 10
    x = torch.randn(B, D)
    y = torch.randn(B, 1)

    # -- Section 1: Point regression with mask/weight support ----------
    model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
    loss_fn = WeightedHuberLoss(delta=1.0)
    mask = torch.ones(B, 1).bool()
    w = torch.ones(B, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(2):
        pred = model(x)
        loss = loss_fn(pred, y, mask=mask, weights=w)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    with torch.no_grad():
        y_pred = model(x)
        r = rmse(y_pred, y)
        m = mae(y_pred, y)
        r2 = r2_score(y_pred, y)
    assert isinstance(r, float)
    assert isinstance(m, float)
    assert isinstance(r2, float)

    # -- Section 2: Heteroscedastic Gaussian regression ---------------
    model2 = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))
    loss_fn2 = GaussianNLLLoss()
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)

    for _ in range(2):
        out = model2(x)
        loss = loss_fn2(out, y)
        loss.backward()
        opt2.step()
        opt2.zero_grad()

    with torch.no_grad():
        out = model2(x)
        mu, logvar = out[:, 0], out[:, 1]
        var = torch.exp(logvar)
        std = torch.sqrt(var)
        crps = crps_gaussian(mu, y.squeeze(), std)
        gnll = gaussian_nll(mu, y.squeeze(), var)
    assert isinstance(crps, float)
    assert isinstance(gnll, float)

    # -- Section 3: Robust regression ---------------------------------
    CauchyLoss(c=1.0)
    TukeyBiweightLoss(c=4.685)
    adaptive = AdaptiveRobustLoss()
    params = list(adaptive.parameters())
    assert len(params) > 0

    # -- Section 4: Conformal prediction ------------------------------
    y_pred_cal = torch.randn(50, 1)
    y_cal_target = torch.randn(50, 1)
    y_pred_test = torch.randn(20, 1)

    cp = SplitConformal(alpha=0.1)
    cp.calibrate(y_pred_cal, y_cal_target)
    lower, upper = cp.predict_interval(y_pred_test)
    assert lower.shape == y_pred_test.shape
    assert upper.shape == y_pred_test.shape

    quantile_preds_cal = torch.cat([y_pred_cal - 0.5, y_pred_cal + 0.5], dim=-1)
    cqr = CQR(alpha=0.1)
    cqr.calibrate(quantile_preds_cal, y_cal_target)
    quantile_preds_test = torch.cat([y_pred_test - 0.5, y_pred_test + 0.5], dim=-1)
    lower, upper = cqr.predict_interval(quantile_preds_test)
    assert lower.shape == y_pred_test.shape

    # -- Section 5: Epistemic uncertainty via ensembles ----------------
    def heteroscedastic_head():
        return nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))

    ens = DeepEnsemble(base_model=heteroscedastic_head(), ensemble_size=3, base_seed=0)
    assert len(ens.models) == 3

    mu_per_member = torch.stack([m(x) for m in ens.models], dim=0)  # [M, B, 2]
    mu_stack = mu_per_member[:, :, 0]  # [M, B]
    logvar_stack = mu_per_member[:, :, 1]  # [M, B]
    var_per_member = torch.exp(logvar_stack)

    decomp = uncertainty_decomposition(mu_stack, var_per_member)
    assert "epistemic_uncertainty" in decomp
    assert "aleatoric_uncertainty" in decomp
    assert "total_uncertainty" in decomp


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
