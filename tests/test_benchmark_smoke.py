import json
from pathlib import Path

from tools import benchmark_smoke


def test_benchmark_smoke_harness_runs_and_writes_json(tmp_path: Path) -> None:
    output_path = tmp_path / "benchmark_smoke.json"

    payload = benchmark_smoke.write_benchmark_smoke_report(
        output_path=output_path,
        iterations=2,
        warmup=0,
        seed=123,
        device="cpu",
    )

    assert output_path.exists()
    assert payload["artifact"] == "benchmark_smoke"
    assert payload["config"]["iterations"] == 2
    assert payload["config"]["device"] == "cpu"
    assert payload["aggregate"]["n_cases"] == len(payload["cases"])
    assert payload["aggregate"]["n_error"] == 0

    required_cases = {
        "gaussian_diag_nll_forward",
        "gaussian_multivariate_full_forward",
        "gaussian_low_rank_forward",
        "mdn_diagonal_forward",
        "mdn_full_forward",
        "functional_eiv_forward",
        "ensemble_variance_decomposition",
        "calibration_score_gaussian",
        "ood_metrics_report_combo",
        "normalizing_flow_forward_optional",
    }
    by_name = {c["name"]: c for c in payload["cases"]}
    assert required_cases <= set(by_name)

    for name in required_cases - {"normalizing_flow_forward_optional"}:
        case = by_name[name]
        assert case["status"] == "ok", f"{name} failed: {case['detail']}"
        assert case["mean_ms"] is not None and case["mean_ms"] >= 0.0

    optional_case = by_name["normalizing_flow_forward_optional"]
    assert optional_case["status"] in {"ok", "skipped"}
    if optional_case["status"] == "ok":
        assert optional_case["mean_ms"] is not None and optional_case["mean_ms"] >= 0.0


def test_benchmark_sweep_harness_runs_and_emits_params(tmp_path: Path) -> None:
    output_path = tmp_path / "benchmark_sweep.json"

    payload = benchmark_smoke.write_benchmark_sweep_report(
        output_path=output_path,
        iterations=1,
        warmup=0,
        seed=123,
        device="cpu",
    )

    assert output_path.exists()
    assert payload["artifact"] == "benchmark_sweep"
    assert payload["config"]["device"] == "cpu"
    assert payload["aggregate"]["n_error"] == 0

    case_names = {c["name"] for c in payload["cases"]}
    assert {
        "sweep_gaussian_multivariate_full_forward",
        "sweep_gaussian_low_rank_forward",
        "sweep_mdn_diagonal_forward",
        "sweep_mdn_full_forward",
        "sweep_ensemble_variance_decomposition",
    } <= case_names

    for case in payload["cases"]:
        assert "params" in case
        assert case["status"] == "ok"
        assert case["mean_ms"] is not None and case["mean_ms"] >= 0.0

    full_gauss_cases = [
        c for c in payload["cases"] if c["name"] == "sweep_gaussian_multivariate_full_forward"
    ]
    assert len(full_gauss_cases) == 4  # 2 batch sizes x 2 target dims
    assert all("batch" in c["params"] and "target_dim" in c["params"] for c in full_gauss_cases)


def test_threshold_derivation_and_evaluation_logic() -> None:
    report = {
        "artifact": "benchmark_sweep",
        "config": {"device": "cpu"},
        "cases": [
            {
                "name": "case_a",
                "status": "ok",
                "mean_ms": 1.0,
                "params": {"batch": 32},
            },
            {
                "name": "case_b",
                "status": "ok",
                "mean_ms": 2.0,
                "params": None,
            },
            {
                "name": "case_skip",
                "status": "skipped",
                "mean_ms": None,
                "params": None,
            },
        ],
    }

    thresholds = benchmark_smoke.derive_thresholds_from_report(report, multiplier=2.0, floor_ms=0.1)
    assert thresholds["artifact"] == "benchmark_thresholds"
    assert "case_a|batch=32" in thresholds["limits"]
    assert thresholds["limits"]["case_b"]["max_mean_ms"] == 4.0

    verdict_ok = benchmark_smoke.evaluate_report_against_thresholds(report, thresholds)
    assert verdict_ok["ok"] is True
    assert verdict_ok["failed_cases"] == 0

    thresholds["limits"]["case_b"]["max_mean_ms"] = 1.5
    verdict_fail = benchmark_smoke.evaluate_report_against_thresholds(report, thresholds)
    assert verdict_fail["ok"] is False
    assert verdict_fail["failed_cases"] == 1
    assert verdict_fail["failures"][0]["case_key"] == "case_b"


def test_committed_cpu_threshold_baseline_path_and_schema() -> None:
    path = Path("reports/benchmark_thresholds/cpu/sweep.json")
    assert path.exists()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["artifact"] == "benchmark_thresholds"
    assert payload["target_artifact"] == "benchmark_sweep"
    assert payload["target_device"] == "cpu"
    assert isinstance(payload["limits"], dict) and payload["limits"]


def test_committed_cpu_smoke_threshold_baseline_path_and_schema() -> None:
    path = Path("reports/benchmark_thresholds/cpu/smoke.json")
    assert path.exists()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["artifact"] == "benchmark_thresholds"
    assert payload["target_artifact"] == "benchmark_smoke"
    assert payload["target_device"] == "cpu"
    assert isinstance(payload["limits"], dict) and payload["limits"]
