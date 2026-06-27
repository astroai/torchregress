import re

import pytest

import torchregress as tr
from torchregress import method_catalog


def test_method_catalog_includes_peer_uq_methods_without_experimental_default_label() -> None:
    names = {m.name for m in method_catalog._METHODS}
    assert {"SWAG", "BayesianNeuralNetwork", "MDNLoss"} <= names
    assert {"BayesianLinearHead", "RecursiveBayesianHead"} <= names

    for name in ("SWAG", "BayesianNeuralNetwork", "MDNLoss"):
        meta = method_catalog.get_method_metadata(name)
        assert meta["maturity"] != "Experimental"
        assert meta["family"] in {"swag", "bnn", "mdn"}

    for name in ("BayesianLinearHead", "RecursiveBayesianHead"):
        meta = method_catalog.get_method_metadata(name)
        assert meta["maturity"] == "Available"
        assert meta["family"] == "test_time"


def test_method_catalog_filtering_by_capability_and_task_tag() -> None:
    multimodal = method_catalog.list_methods(capability_filters={"multimodal": "yes"})
    multimodal_names = {row["name"] for row in multimodal}
    assert {"MDNLoss", "NormalizingFlowLoss", "ContrastiveFlowLoss"} <= multimodal_names

    eiv = method_catalog.list_methods(task_tag="noisy_features")
    eiv_names = {row["name"] for row in eiv}
    assert "FunctionalEIVLoss" in eiv_names
    assert "OrthogonalDistanceRegressionLoss" in eiv_names

    decomposition = method_catalog.list_methods(capability_filters={"decomposition": "yes"})
    decomp_names = {row["name"] for row in decomposition}
    assert {
        "HeteroscedasticEnsembleModel",
        "HeteroscedasticBNN",
        "MDNEnsembleModel",
    } <= decomp_names
    assert "MDNLoss" not in decomp_names

    inference = method_catalog.list_methods(task_tag="inference")
    inference_names = {row["name"] for row in inference}
    assert "PredictionPoweredInference" in inference_names

    param_est = method_catalog.list_methods(task_tag="parameter_estimation")
    param_est_names = {row["name"] for row in param_est}
    assert "ContrastiveFlowLoss" in param_est_names

    ordinal = method_catalog.list_methods(task_tag="ordinal")
    ordinal_names = {row["name"] for row in ordinal}
    assert {"OrdinalCrossEntropyLoss", "CumulativeLinkLoss", "CORALLoss"} <= ordinal_names

    censored = method_catalog.list_methods(task_tag="censored")
    censored_names = {row["name"] for row in censored}
    assert {"CensoredGaussianNLLLoss", "CensoredQuantileLoss", "AFTLoss"} <= censored_names

    selection_bias = method_catalog.list_methods(task_tag="selection_bias")
    selection_names = {row["name"] for row in selection_bias}
    assert "PropensityWeightedLoss" in selection_names

    posthoc = method_catalog.list_methods(task_tag="posthoc_calibration")
    posthoc_names = {row["name"] for row in posthoc}
    assert {
        "VarianceTemperatureScaler",
        "IsotonicMeanCalibrator",
        "PITCalibrator",
    } <= posthoc_names

    uncertain_gt = method_catalog.list_methods(task_tag="uncertain_ground_truth")
    uncertain_gt_names = {row["name"] for row in uncertain_gt}
    assert {
        "NoisyTargetGaussianNLL",
        "PseudoLabelNLL",
        "ConsistencyRegLoss",
        "PseudoLabelConsistencyLoss",
    } <= uncertain_gt_names

    target_transforms = method_catalog.list_methods(task_tag="target_transform")
    transform_names = {row["name"] for row in target_transforms}
    assert {
        "LogTransformLoss",
        "BoxCoxTransformLoss",
        "SqrtTransformLoss",
        "YeoJohnsonTransformLoss",
    } <= transform_names

    density_cp = method_catalog.list_methods(task_tag="density_conformal")
    density_cp_names = {row["name"] for row in density_cp}
    assert {"DensityConformal", "PrevalenceAdjustedCP", "MonteCarloConformal"} <= density_cp_names

    causal = method_catalog.list_methods(task_tag="causal_inference")
    causal_names = {row["name"] for row in causal}
    assert {"dr_ate", "dr_cate"} <= causal_names

    low_compute = method_catalog.list_methods(task_tag="low_compute")
    low_compute_names = {row["name"] for row in low_compute}
    assert {"HeteroscedasticBatchEnsembleModel", "MCDropoutWrapper"} <= low_compute_names

    low_shot = method_catalog.list_methods(task_tag="low_shot")
    low_shot_names = {row["name"] for row in low_shot}
    assert {"BayesianLinearHead", "RecursiveBayesianHead"} <= low_shot_names


def test_method_catalog_is_exposed_via_top_level_module_namespace() -> None:
    assert hasattr(tr, "method_catalog")
    assert tr.method_catalog.get_method_metadata("ConformalLoss")["family"] == "conformal"


def test_method_catalog_unknown_method_raises_key_error() -> None:
    with pytest.raises(KeyError):
        method_catalog.get_method_metadata("DoesNotExist")


def test_task_recommendations_include_hard_problem_rows_and_peer_methods() -> None:
    rows = method_catalog.list_task_recommendations()
    tasks = {row["task"] for row in rows}
    assert {
        "Imbalanced / rare-target regression",
        "Calibrated intervals with coverage guarantees",
        "Population inference with few labels",
        "Ordinal / ordered targets",
        "Censored / interval-censored regression",
        "Selection bias / covariate-dependent missing labels",
        "Output constraints / monotonicity",
        "Post-hoc calibration transforms",
        "Density-aware conformal under long-tail targets",
        "Uncertain ground-truth / weak labels",
        "Semi-supervised regression",
        "Target transforms for skewed / multiplicative-noise regression",
        "Causal inference regression (ATE/CATE)",
        "OOD scoring / selective prediction",
        "Noisy features / measurement error",
        "Multimodal targets",
        "Low-shot / streaming linear head on fixed features",
    } <= tasks

    ood_row = next(row for row in rows if row["task"] == "OOD scoring / selective prediction")
    assert ood_row["recommended_start"] == "BaseEnsembleModel + OOD metrics"
    assert any("HeteroscedasticBatchEnsembleModel" in alt for alt in ood_row["strong_alternatives"])
    assert any("SWAG" in alt for alt in ood_row["strong_alternatives"])
    assert any("BayesianNeuralNetwork" in alt for alt in ood_row["strong_alternatives"])


def test_decision_workflow_and_comparative_evidence_metadata_cover_hard_tasks() -> None:
    workflow = method_catalog.list_decision_workflow_steps()
    assert any("coverage guarantees" in row["question"].lower() for row in workflow)
    assert any("OOD" in row["question"] for row in workflow)
    assert any("SWAG" in alt for row in workflow for alt in row["alternatives"])

    evidence = method_catalog.list_comparative_evidence_rows()
    tasks = {row["task"] for row in evidence}
    assert {
        "Imbalanced / rare-target regression",
        "Calibrated intervals / coverage",
        "Ordinal regression / ordered targets",
        "Censored / interval-censored regression",
        "Output constraints + post-hoc calibration transforms",
        "Target transforms for skewed regression",
        "Semi-supervised regression / limited labels",
        "Uncertain ground-truth + density-aware conformal",
        "Causal inference regression (DR ATE/CATE)",
        "OOD robustness / selective prediction",
        "Noisy features / EIV",
        "Multimodal / multi-target non-Gaussian",
        "Low-shot linear adaptation on fixed features (last layer)",
    } <= tasks
    grades = {row["comparison_grade"] for row in evidence}
    assert {"Decision-grade", "Strong"} <= grades
    assert "Missing" not in grades
    assert "Emerging" not in grades
    assert "Demo-only" not in grades


# ---------------------------------------------------------------------------
# Phantom-class guard: every class/function name referenced by the public catalog
# must be importable. This is the single source of truth for the
# "never document a class that doesn't exist" rule from AGENTS.md.
# ---------------------------------------------------------------------------


# Task-tag / acronym tokens that look like class names but are not.
_NON_CLASS_TOKENS: frozenset[str] = frozenset({"OOD"})

# English stopwords that survive identifier extraction from prose.
# Only add words that never appear as class/function names in this library.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "A",
        "An",
        "The",
        "Use",
        "Prefer",
        "Start",
        "Keep",
        "Move",
        "Fit",
        "Match",
        "Consider",
        "Apply",
        "Look",
        "Switch",
        "Retain",
        "Always",
        "Never",
        "Avoid",
        "All",
        "Any",
        "For",
        "And",
        "Or",
        "Not",
        "But",
        "If",
        "Then",
        "After",
        "Before",
        "From",
        "Into",
        "On",
        "Of",
        "To",
        "In",
        "By",
        "With",
        "Without",
        "Across",
        "Against",
        "Plus",
        "Tune",
        "Train",
        "Calibrate",
        "Align",
        "Validate",
        "Evaluate",
        "Pair",
        "Penalise",
        "Rewards",
        "Requires",
        "Defaults",
        "Rescale",
        "Stronger",
        "Strongest",
        "Standard",
        "Same",
        "Different",
        "New",
        "First",
        "Second",
        "Third",
        "Both",
        "One",
        "Two",
        "Three",
        "Four",
        "Five",
        "Many",
        "Most",
        "Some",
        "Cheat",
        "Mix",
        "Cover",
        "Scale",
        "Combine",
        "Set",
        "Useful",
        "Easy",
        "Hard",
        "Aggressive",
        "Conservative",
        "Robust",
        "Adds",
        "Compares",
        "Includes",
        "Reports",
        "Captures",
        "Pushes",
        "Sample",
        "Stage",
        "Trick",
        "Strong",
        "tail",  # as in "tail-slice evaluation" in imbalanced-tail recommendations
    }
)

# Public submodules that hold importable classes/functions exposed in the catalog.
# Note: ``torchregress.health`` is intentionally excluded — it's a script entry
# point (``check_health``) registered via ``project.scripts``, not a lazy-loaded
# top-level submodule. ``torchregress.method_catalog`` is also excluded because
# it only exposes catalog helper functions, not loss/algorithm classes.
_PUBLIC_SUBMODULES: dict[str, object] = {
    "torchregress.losses": tr.losses,
    "torchregress.ensemble": tr.ensemble,
    "torchregress.calibration": tr.calibration,
    "torchregress.algorithms": tr.algorithms,
    "torchregress.causal": tr.causal,
    "torchregress.constraints": tr.constraints,
    "torchregress.test_time": tr.test_time,
    "torchregress.inference": tr.inference,
    "torchregress.metrics": tr.metrics,
    "torchregress.semi_supervised": tr.semi_supervised,
    "torchregress.prediction": tr.prediction,
    "torchregress.viz": tr.viz,
    "torchregress.utils": tr.utils,
    "torchregress.comparison": tr.comparison,
}


def _extract_class_names(text: str) -> list[str]:
    """Extract identifiers that look like class/function names from prose.

    Handles:
    - "X / Y / Z"             slash-separated alternatives
    - "X + Y"                 combination
    - "X, Y"                  comma list
    - "X (comment, ...)"      parenthetical comments
    - "X on top of ..."       leading prose
    - "X + OOD metrics"       trailing prose
    - snake_case functions like ``dr_ate`` (not just CamelCase classes)

    Filters out a curated English stopword list and the ``_NON_CLASS_TOKENS``
    set (e.g. ``OOD``).
    """
    if not text:
        return []
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text)
    candidates = re.split(r"\s*(?:/|,|\+)\s*", text)
    names: list[str] = []
    for c in candidates:
        c = c.strip()
        if not c:
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)", c)
        if not m:
            continue
        name = m.group(1)
        if name in _STOPWORDS or name in _NON_CLASS_TOKENS:
            continue
        names.append(name)
    return names


def _is_valid_identifier(name: str) -> bool:
    """True if ``name`` is a clean Python identifier (no spaces or hyphens)."""
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name))


def _resolve_public_path(name: str) -> str | None:
    """Resolve a method name to its fully qualified public path.

    Looks up the catalog first, then falls back to scanning the public
    submodules listed in ``_PUBLIC_SUBMODULES``.
    """
    for m in method_catalog._METHODS:
        if m.name == name:
            return m.public_path

    for path, mod in _PUBLIC_SUBMODULES.items():
        if hasattr(mod, name):
            return f"{path}.{name}"
    return None


def test_task_recommendations_reference_importable_classes() -> None:
    """Phantom-class guard for _TASK_RECOMMENDATIONS.

    Every CamelCase token in ``recommended_start`` and ``strong_alternatives``
    must resolve to an importable public symbol. Fix the catalog (correct the
    name) or add the class to the public API.
    """
    failures: list[str] = []

    for rec in method_catalog.list_task_recommendations():
        task = rec["task"]
        names: set[str] = set(_extract_class_names(rec["recommended_start"]))
        for alt in rec["strong_alternatives"]:
            names.update(_extract_class_names(alt))
        names -= _NON_CLASS_TOKENS

        for name in sorted(names):
            if _resolve_public_path(name) is None:
                failures.append(
                    f"  - {task!r}: {name!r} referenced in prose "
                    f"but not importable from any public submodule"
                )

    if failures:
        pytest.fail(
            "Catalog references classes that don't exist or aren't importable. "
            "Fix the catalog (correct name) or add the class to the public API.\n"
            + "\n".join(failures)
        )


def test_decision_workflow_references_importable_classes() -> None:
    """Every class name in _DECISION_WORKFLOW must be importable."""
    failures: list[str] = []

    for step in method_catalog.list_decision_workflow_steps():
        question = step["question"]
        names: set[str] = set(_extract_class_names(step["primary_recommendation"]))
        for alt in step.get("alternatives", ()):
            names.update(_extract_class_names(alt))
        names -= _NON_CLASS_TOKENS

        for name in sorted(names):
            if _resolve_public_path(name) is None:
                failures.append(
                    f"  - {question!r}: {name!r} referenced in workflow "
                    f"but not importable from any public submodule"
                )

    if failures:
        pytest.fail(
            "Decision workflow references classes that don't exist:\n" + "\n".join(failures)
        )


def test_evidence_rows_peer_methods_are_importable() -> None:
    """Every peer method in _COMPARATIVE_EVIDENCE_ROWS.peer_methods_visible
    must be importable from the public API.
    """
    failures: list[str] = []

    for row in method_catalog.list_comparative_evidence_rows():
        task = row["task"]
        for name in row.get("peer_methods_visible", ()):
            # ``peer_methods_visible`` is supposed to be a list of clean class
            # identifiers. Filter out prose entries (e.g. "naive difference-in-means")
            # so the test focuses on real callables.
            if not _is_valid_identifier(name):
                continue
            if _resolve_public_path(name) is None:
                failures.append(
                    f"  - {task!r}: {name!r} in peer_methods_visible "
                    f"but not importable from any public submodule"
                )

    if failures:
        pytest.fail(
            "Comparative evidence rows reference classes that don't exist:\n" + "\n".join(failures)
        )


def test_peer_methods_visible_contains_only_identifiers() -> None:
    """Prose guard: every entry in ``peer_methods_visible`` must be a valid
    Python identifier. Catches regressions that re-introduce descriptive
    strings (e.g. "naive difference-in-means") into a field that is supposed
    to be a clean list of class names.
    """
    failures: list[str] = []
    for row in method_catalog.list_comparative_evidence_rows():
        task = row["task"]
        for name in row.get("peer_methods_visible", ()):
            if not _is_valid_identifier(name):
                failures.append(
                    f"  - {task!r}: {name!r} is prose, not an identifier. "
                    f"Move descriptive context to the row's `notes` field."
                )

    if failures:
        pytest.fail(
            "Comparative evidence rows have prose entries in peer_methods_visible:\n"
            + "\n".join(failures)
        )


def test_every_catalog_public_path_resolves() -> None:
    """Belt-and-suspenders: every MethodMetadata.public_path must be importable.

    Catches catalog entries added without a corresponding class.
    """
    import importlib

    failures: list[str] = []

    for m in method_catalog._METHODS:
        mod_path, _, attr = m.public_path.rpartition(".")
        try:
            mod = importlib.import_module(mod_path)
        except ImportError as e:
            failures.append(f"  - {m.name}: {m.public_path} module import failed: {e}")
            continue
        if not hasattr(mod, attr):
            failures.append(f"  - {m.name}: {m.public_path} attribute not found on module")

    if failures:
        pytest.fail("Catalog entries with broken public_path:\n" + "\n".join(failures))
