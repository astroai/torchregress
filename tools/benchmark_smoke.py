"""Fast benchmark smoke harness for representative hard paths.

This is intentionally a *smoke* benchmark:
- tiny tensor sizes
- few iterations
- structured JSON output

It seeds the performance/scalability workstream with reproducible numbers and a
CI-friendly execution path, without claiming production-grade benchmarking rigor.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Any, Callable

import torch
import torch.nn as nn

from torchregress.losses import (
    FunctionalEIVLoss,
    GaussianNLLLoss,
    LowRankGaussianLoss,
    MDNLoss,
    MultivariateGaussianLoss,
)
from torchregress.metrics import (
    calibration_score,
    ensemble_variance_decomposition,
    ood_metrics_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BenchCaseResult:
    name: str
    status: str  # "ok" | "skipped" | "error"
    iterations: int
    warmup: int
    mean_ms: float | None
    std_ms: float | None
    min_ms: float | None
    max_ms: float | None
    detail: str = ""
    params: dict[str, int] | None = None


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _benchmark_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "num_threads": torch.get_num_threads(),
    }


def _case_key(case: dict[str, Any]) -> str:
    params = case.get("params") or {}
    if not params:
        return case["name"]
    parts = [f"{k}={params[k]}" for k in sorted(params)]
    return f"{case['name']}|" + "|".join(parts)


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _consume(value: Any) -> float:
    """Force eager computation and return a scalar summary."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().float().mean().item())
    if isinstance(value, (tuple, list)):
        return sum(_consume(v) for v in value)
    if isinstance(value, dict):
        return sum(_consume(v) for v in value.values())
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _time_case(
    *,
    name: str,
    fn: Callable[[], Any],
    device: torch.device,
    iterations: int,
    warmup: int,
    params: dict[str, int] | None = None,
) -> BenchCaseResult:
    try:
        for _ in range(warmup):
            _consume(fn())
            _sync_if_needed(device)

        samples_ms: list[float] = []
        for _ in range(iterations):
            _sync_if_needed(device)
            t0 = perf_counter()
            out = fn()
            _ = _consume(out)
            _sync_if_needed(device)
            samples_ms.append((perf_counter() - t0) * 1e3)

        return BenchCaseResult(
            name=name,
            status="ok",
            iterations=iterations,
            warmup=warmup,
            mean_ms=mean(samples_ms),
            std_ms=pstdev(samples_ms) if len(samples_ms) > 1 else 0.0,
            min_ms=min(samples_ms),
            max_ms=max(samples_ms),
            params=params,
        )
    except ImportError as exc:
        return BenchCaseResult(
            name=name,
            status="skipped",
            iterations=iterations,
            warmup=warmup,
            mean_ms=None,
            std_ms=None,
            min_ms=None,
            max_ms=None,
            detail=f"{type(exc).__name__}: {exc}",
            params=params,
        )
    except Exception as exc:  # pragma: no cover - surfaced in CI/runtime
        return BenchCaseResult(
            name=name,
            status="error",
            iterations=iterations,
            warmup=warmup,
            mean_ms=None,
            std_ms=None,
            min_ms=None,
            max_ms=None,
            detail=f"{type(exc).__name__}: {exc}",
            params=params,
        )


def _make_spd_cov(batch: int, dim: int, device: torch.device) -> torch.Tensor:
    a = torch.randn(batch, dim, dim, device=device)
    eye = torch.eye(dim, device=device).expand(batch, -1, -1)
    return a @ a.transpose(-1, -2) + 0.1 * eye


def _aggregate_case_results(results: list[BenchCaseResult]) -> dict[str, Any]:
    ok_cases = [r for r in results if r.status == "ok"]
    return {
        "n_cases": len(results),
        "n_ok": len(ok_cases),
        "n_skipped": sum(r.status == "skipped" for r in results),
        "n_error": sum(r.status == "error" for r in results),
        "mean_of_means_ms": (
            float(sum(r.mean_ms for r in ok_cases if r.mean_ms is not None) / len(ok_cases))
            if ok_cases
            else None
        ),
    }


def run_benchmark_smoke(
    *,
    iterations: int = 5,
    warmup: int = 1,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run benchmark smoke suite and return structured results."""
    torch.manual_seed(seed)
    device_obj = _resolve_device(device)

    # Reduce run-to-run noise for smoke checks.
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass

    results: list[BenchCaseResult] = []

    # ------------------------------------------------------------------
    # Representative hard-path setups (small tensors, reproducible)
    # ------------------------------------------------------------------
    b, d, rank = 32, 4, 2
    y_true = torch.randn(b, 1, device=device_obj)
    diag_pred = torch.randn(b, 2, device=device_obj)
    diag_loss = GaussianNLLLoss().to(device_obj)

    mv_mean = torch.randn(b, d, device=device_obj)
    mv_target = torch.randn(b, d, device=device_obj)
    mv_cov = _make_spd_cov(b, d, device_obj)
    mv_loss = MultivariateGaussianLoss(reduction="mean").to(device_obj)

    lr_mean = torch.randn(b, d, device=device_obj)
    lr_target = torch.randn(b, d, device=device_obj)
    lr_cov_factor = torch.randn(b, d, rank, device=device_obj) * 0.1
    lr_cov_diag = torch.rand(b, d, device=device_obj) + 0.1
    lr_loss = LowRankGaussianLoss(reduction="mean").to(device_obj)

    mdn_diag = MDNLoss(n_components=3, n_features=d, covariance_type="diagonal").to(device_obj)
    mdn_diag_pred = torch.randn(b, mdn_diag.expected_output_size, device=device_obj)
    mdn_full = MDNLoss(n_components=3, n_features=d, covariance_type="full").to(device_obj)
    mdn_full_pred = torch.randn(b, mdn_full.expected_output_size, device=device_obj)
    mdn_target = torch.randn(b, d, device=device_obj)

    eiv_model = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 1)).to(device_obj)
    eiv_loss = FunctionalEIVLoss(model=eiv_model, sigma_x=0.1, sigma_y=0.2, monte_carlo=False).to(
        device_obj
    )
    eiv_x_obs = torch.randn(16, 3, device=device_obj)
    eiv_y_obs = torch.randn(16, 1, device=device_obj)

    ens_means = torch.randn(5, 64, 2, device=device_obj)
    ens_vars = torch.rand(5, 64, 2, device=device_obj) + 0.05

    cal_y = torch.randn(128, 1, device=device_obj)
    cal_mean = cal_y + 0.1 * torch.randn_like(cal_y)
    cal_std = torch.rand_like(cal_y) * 0.5 + 0.1

    ood_mean = torch.zeros(64, 1, device=device_obj)
    ood_var = torch.ones(64, 1, device=device_obj) * 0.25
    ood_x_test = torch.randn(64, 1, device=device_obj)
    ood_x_ref = torch.randn(128, 1, device=device_obj)
    ood_samples = torch.randn(24, 64, 1, device=device_obj)
    ood_maha_mean = torch.tensor([0.0], device=device_obj)
    ood_maha_cov = torch.eye(1, device=device_obj)

    results.append(
        _time_case(
            name="gaussian_diag_nll_forward",
            fn=lambda: diag_loss(diag_pred, y_true),
            device=device_obj,
            iterations=iterations,
            warmup=warmup,
        )
    )
    results.append(
        _time_case(
            name="gaussian_multivariate_full_forward",
            fn=lambda: mv_loss(mv_mean, mv_target, mv_cov),
            device=device_obj,
            iterations=iterations,
            warmup=warmup,
        )
    )
    results.append(
        _time_case(
            name="gaussian_low_rank_forward",
            fn=lambda: lr_loss(lr_mean, lr_target, lr_cov_factor, lr_cov_diag),
            device=device_obj,
            iterations=iterations,
            warmup=warmup,
        )
    )
    results.append(
        _time_case(
            name="mdn_diagonal_forward",
            fn=lambda: mdn_diag(mdn_diag_pred, mdn_target),
            device=device_obj,
            iterations=iterations,
            warmup=warmup,
        )
    )
    results.append(
        _time_case(
            name="mdn_full_forward",
            fn=lambda: mdn_full(mdn_full_pred, mdn_target),
            device=device_obj,
            iterations=iterations,
            warmup=warmup,
        )
    )
    results.append(
        _time_case(
            name="functional_eiv_forward",
            fn=lambda: eiv_loss(eiv_x_obs, eiv_y_obs),
            device=device_obj,
            iterations=iterations,
            warmup=warmup,
        )
    )
    results.append(
        _time_case(
            name="ensemble_variance_decomposition",
            fn=lambda: ensemble_variance_decomposition(ens_means, ens_vars),
            device=device_obj,
            iterations=iterations,
            warmup=warmup,
        )
    )
    results.append(
        _time_case(
            name="calibration_score_gaussian",
            fn=lambda: calibration_score(cal_y, cal_mean, cal_std, n_levels=9),
            device=device_obj,
            iterations=iterations,
            warmup=warmup,
        )
    )
    results.append(
        _time_case(
            name="ood_metrics_report_combo",
            fn=lambda: ood_metrics_report(
                model_output={"mean": ood_mean, "variance": ood_var},
                x_test=ood_x_test,
                x_reference=ood_x_ref,
                mean=ood_maha_mean,
                cov=ood_maha_cov,
                samples=ood_samples,
            ),
            device=device_obj,
            iterations=iterations,
            warmup=warmup,
        )
    )

    def _flow_forward() -> torch.Tensor:
        from torchregress.losses import (
            create_flow_loss,
            create_flow_model,
        )

        context_dim = 8
        flow_loss = create_flow_loss(
            n_features=2,
            context_dim=context_dim,
            flow_type="maf",
            n_transforms=2,
            hidden_features=[16, 16],
            reduction="mean",
        ).to(device_obj)
        context = torch.randn(32, context_dim, device=device_obj)
        target = torch.randn(32, 2, device=device_obj)
        # Also create model once to exercise helper path without timing construction separately.
        _ = create_flow_model(
            n_features=2,
            context_dim=context_dim,
            flow_type="maf",
            n_transforms=2,
            hidden_features=[16, 16],
        )
        return flow_loss(context, target)

    results.append(
        _time_case(
            name="normalizing_flow_forward_optional",
            fn=_flow_forward,
            device=device_obj,
            iterations=max(1, min(2, iterations)),  # keep optional path cheap
            warmup=0,
        )
    )

    aggregate = _aggregate_case_results(results)

    payload = {
        "artifact": "benchmark_smoke",
        "version": 1,
        "config": {
            "iterations": iterations,
            "warmup": warmup,
            "seed": seed,
            "requested_device": device,
            "device": str(device_obj),
        },
        "environment": _benchmark_environment(),
        "cases": [asdict(r) for r in results],
        "aggregate": aggregate,
    }
    return payload


def run_benchmark_sweep(
    *,
    iterations: int = 2,
    warmup: int = 0,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run a small scale sweep across representative dimensions."""
    torch.manual_seed(seed)
    device_obj = _resolve_device(device)
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass

    results: list[BenchCaseResult] = []

    # 1) Batch size + target dimension sweeps (full Gaussian)
    for batch in (16, 64):
        for dim in (2, 8):
            mean_t = torch.randn(batch, dim, device=device_obj)
            target_t = torch.randn(batch, dim, device=device_obj)
            cov_t = _make_spd_cov(batch, dim, device_obj)
            loss = MultivariateGaussianLoss(reduction="mean").to(device_obj)
            results.append(
                _time_case(
                    name="sweep_gaussian_multivariate_full_forward",
                    fn=lambda m=mean_t, t=target_t, c=cov_t, loss_fn=loss: loss_fn(m, t, c),
                    device=device_obj,
                    iterations=iterations,
                    warmup=warmup,
                    params={"batch": batch, "target_dim": dim},
                )
            )

    # 2) Batch size + target dimension sweeps (low-rank Gaussian)
    for batch in (16, 64):
        for dim in (4, 8):
            rank = 2 if dim >= 4 else 1
            mean_t = torch.randn(batch, dim, device=device_obj)
            target_t = torch.randn(batch, dim, device=device_obj)
            cov_factor = torch.randn(batch, dim, rank, device=device_obj) * 0.1
            cov_diag = torch.rand(batch, dim, device=device_obj) + 0.1
            loss = LowRankGaussianLoss(reduction="mean").to(device_obj)
            results.append(
                _time_case(
                    name="sweep_gaussian_low_rank_forward",
                    fn=lambda m=mean_t, t=target_t, f=cov_factor, d=cov_diag, loss_fn=loss: loss_fn(
                        m, t, f, d
                    ),
                    device=device_obj,
                    iterations=iterations,
                    warmup=warmup,
                    params={"batch": batch, "target_dim": dim, "rank": rank},
                )
            )

    # 3) Mixture component and target dimension sweeps (MDN)
    for covariance_type in ("diagonal", "full"):
        for n_components in (2, 5):
            for dim in (2, 4):
                batch = 32
                loss = MDNLoss(
                    n_components=n_components,
                    n_features=dim,
                    covariance_type=covariance_type,
                ).to(device_obj)
                pred = torch.randn(batch, loss.expected_output_size, device=device_obj)
                target_t = torch.randn(batch, dim, device=device_obj)
                results.append(
                    _time_case(
                        name=f"sweep_mdn_{covariance_type}_forward",
                        fn=lambda p=pred, t=target_t, loss_fn=loss: loss_fn(p, t),
                        device=device_obj,
                        iterations=iterations,
                        warmup=warmup,
                        params={
                            "batch": batch,
                            "target_dim": dim,
                            "mixture_components": n_components,
                        },
                    )
                )

    # 4) Ensemble size and batch size sweeps (decomposition)
    for ensemble_size in (3, 10):
        for batch in (32, 128):
            means_t = torch.randn(ensemble_size, batch, 2, device=device_obj)
            vars_t = torch.rand(ensemble_size, batch, 2, device=device_obj) + 0.05
            results.append(
                _time_case(
                    name="sweep_ensemble_variance_decomposition",
                    fn=lambda m=means_t, v=vars_t: ensemble_variance_decomposition(m, v),
                    device=device_obj,
                    iterations=iterations,
                    warmup=warmup,
                    params={"ensemble_size": ensemble_size, "batch": batch},
                )
            )

    aggregate = _aggregate_case_results(results)
    return {
        "artifact": "benchmark_sweep",
        "version": 1,
        "config": {
            "iterations": iterations,
            "warmup": warmup,
            "seed": seed,
            "requested_device": device,
            "device": str(device_obj),
        },
        "environment": _benchmark_environment(),
        "cases": [asdict(r) for r in results],
        "aggregate": aggregate,
    }


def write_benchmark_smoke_report(
    *,
    output_path: Path,
    iterations: int = 5,
    warmup: int = 1,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    payload = run_benchmark_smoke(
        iterations=iterations,
        warmup=warmup,
        seed=seed,
        device=device,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def write_benchmark_sweep_report(
    *,
    output_path: Path,
    iterations: int = 2,
    warmup: int = 0,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    payload = run_benchmark_sweep(
        iterations=iterations,
        warmup=warmup,
        seed=seed,
        device=device,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def derive_thresholds_from_report(
    report: dict[str, Any],
    *,
    multiplier: float = 4.0,
    floor_ms: float = 0.5,
) -> dict[str, Any]:
    """Create a threshold payload from a benchmark report."""
    limits: dict[str, dict[str, float]] = {}
    for case in report.get("cases", []):
        if case.get("status") != "ok" or case.get("mean_ms") is None:
            continue
        key = _case_key(case)
        base = float(case["mean_ms"])
        limits[key] = {"max_mean_ms": max(floor_ms, base * multiplier)}

    return {
        "artifact": "benchmark_thresholds",
        "version": 1,
        "target_artifact": report.get("artifact", "unknown"),
        "target_device": report.get("config", {}).get("device"),
        "source_config": {
            "multiplier": multiplier,
            "floor_ms": floor_ms,
        },
        "limits": limits,
    }


def evaluate_report_against_thresholds(
    report: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Compare benchmark report to thresholds and return a structured verdict."""
    limits = thresholds.get("limits", {})
    failures: list[dict[str, Any]] = []
    checked = 0

    for case in report.get("cases", []):
        if case.get("status") != "ok" or case.get("mean_ms") is None:
            continue
        key = _case_key(case)
        if key not in limits:
            continue
        checked += 1
        max_mean_ms = limits[key].get("max_mean_ms")
        if max_mean_ms is None:
            continue
        if float(case["mean_ms"]) > float(max_mean_ms):
            failures.append(
                {
                    "case_key": key,
                    "mean_ms": float(case["mean_ms"]),
                    "max_mean_ms": float(max_mean_ms),
                    "ratio": float(case["mean_ms"]) / float(max_mean_ms)
                    if float(max_mean_ms) > 0
                    else None,
                }
            )

    return {
        "ok": len(failures) == 0,
        "checked_cases": checked,
        "failed_cases": len(failures),
        "failures": failures,
        "report_artifact": report.get("artifact"),
        "threshold_target_artifact": thresholds.get("target_artifact"),
        "report_device": report.get("config", {}).get("device"),
        "threshold_target_device": thresholds.get("target_device"),
    }


def _default_output_path() -> Path:
    return REPO_ROOT / "reports" / "benchmark_smoke_latest.json"


def _default_sweep_output_path() -> Path:
    return REPO_ROOT / "reports" / "benchmark_sweep_latest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fast benchmark smoke harness.")
    parser.add_argument(
        "--mode",
        choices=("smoke", "sweep"),
        default="smoke",
        help="Benchmark mode: small representative cases or parameter sweeps.",
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=None,
        help="Optional path to benchmark threshold JSON for regression checking.",
    )
    parser.add_argument(
        "--write-thresholds",
        type=Path,
        default=None,
        help="Write thresholds JSON derived from the produced report.",
    )
    parser.add_argument("--threshold-multiplier", type=float, default=4.0)
    parser.add_argument("--threshold-floor-ms", type=float, default=0.5)
    parser.add_argument(
        "--fail-on-thresholds",
        action="store_true",
        help="Exit non-zero if threshold check fails.",
    )
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        if args.mode == "smoke":
            output_path = _default_output_path()
        else:
            output_path = _default_sweep_output_path()

    if args.mode == "smoke":
        payload = write_benchmark_smoke_report(
            output_path=output_path,
            iterations=args.iterations,
            warmup=args.warmup,
            seed=args.seed,
            device=args.device,
        )
    else:
        payload = write_benchmark_sweep_report(
            output_path=output_path,
            iterations=args.iterations,
            warmup=args.warmup,
            seed=args.seed,
            device=args.device,
        )

    print(f"Wrote benchmark {args.mode} report: {output_path}")
    print(
        "Cases: "
        f"{payload['aggregate']['n_ok']} ok / "
        f"{payload['aggregate']['n_skipped']} skipped / "
        f"{payload['aggregate']['n_error']} error"
    )
    for case in payload["cases"]:
        status = case["status"]
        mean_ms = case["mean_ms"]
        suffix = f"{mean_ms:.3f} ms" if mean_ms is not None else case["detail"]
        key_suffix = ""
        if case.get("params"):
            joined = ", ".join(f"{k}={v}" for k, v in sorted(case["params"].items()))
            key_suffix = f" [{joined}]"
        print(f"- {case['name']}{key_suffix}: {status} ({suffix})")

    if args.write_thresholds is not None:
        thresholds = derive_thresholds_from_report(
            payload,
            multiplier=args.threshold_multiplier,
            floor_ms=args.threshold_floor_ms,
        )
        args.write_thresholds.parent.mkdir(parents=True, exist_ok=True)
        args.write_thresholds.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
        print(f"Wrote thresholds: {args.write_thresholds}")

    if args.thresholds is not None:
        thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
        verdict = evaluate_report_against_thresholds(payload, thresholds)
        print(
            "Threshold check: "
            f"{'PASS' if verdict['ok'] else 'FAIL'} "
            f"({verdict['checked_cases']} checked, {verdict['failed_cases']} failed)"
        )
        if not verdict["ok"]:
            for failure in verdict["failures"]:
                print(
                    "- "
                    f"{failure['case_key']}: "
                    f"{failure['mean_ms']:.3f} ms > {failure['max_mean_ms']:.3f} ms"
                )
            if args.fail_on_thresholds:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
