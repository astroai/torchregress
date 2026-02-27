"""Derive and evaluate regression thresholds for comparison example summary artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return data


def _summary_paths(base_dir: Path, profile: str) -> list[Path]:
    suffix = f"_{profile}.json"
    paths: list[Path] = []
    for p in sorted(base_dir.glob(f"*{suffix}")):
        if not p.is_file() or p.name.startswith("profile_comparison_"):
            continue
        try:
            payload = load_json(p)
        except Exception:
            continue
        if payload.get("artifact") == "comparison_example_summary":
            paths.append(p)
    return paths


def _stem_without_profile(path: Path, profile: str) -> str:
    suffix = f"_{profile}"
    if path.stem.endswith(suffix):
        return path.stem[: -len(suffix)]
    return path.stem


def _iter_numeric_row_metrics(
    payload: dict[str, Any],
) -> list[tuple[str, str, float]]:
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return []
    out: list[tuple[str, str, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        method = row.get("Method")
        if not isinstance(method, str):
            continue
        for key, value in row.items():
            if key == "Method" or value is None or isinstance(value, str):
                continue
            if not isinstance(value, (int, float)):
                continue
            v = float(value)
            if v != v:  # NaN
                continue
            out.append((method, key, v))
    return out


def _is_probability_key(key: str) -> bool:
    k = key.lower()
    return ("cov" in k or "coverage" in k) and "covariance" not in k


def _is_runtime_key(key: str) -> bool:
    return key.endswith("_s")


def _is_r2_key(key: str) -> bool:
    return key.lower() == "r2"


def _is_nonnegative_metric_key(key: str) -> bool:
    k = key.lower()
    if _is_probability_key(key) or _is_runtime_key(key) or _is_r2_key(key):
        return False
    # Continuous-density log-likelihood metrics (often labeled NLL/loss) may be negative.
    # Keep those on signed bounds so we don't force an invalid 0.0 lower bound.
    if "nll" in k or "loss" in k:
        return False
    return any(token in k for token in ("mse", "mae", "width", "energy", "aurc", "risk", "is"))


def _derive_bounds(
    key: str,
    baseline: float,
    *,
    runtime_multiplier: float,
    runtime_floor: float,
    metric_multiplier: float,
    metric_floor: float,
    prob_delta: float,
    r2_delta: float,
) -> tuple[float | None, float | None, str]:
    if _is_probability_key(key):
        return (
            max(0.0, baseline - prob_delta),
            min(1.0, baseline + prob_delta),
            "probability",
        )
    if _is_runtime_key(key):
        return (
            0.0,
            max(runtime_floor, runtime_multiplier * max(0.0, baseline)),
            "runtime",
        )
    if _is_r2_key(key):
        return (baseline - r2_delta, baseline + r2_delta, "r2")
    if _is_nonnegative_metric_key(key):
        return (
            0.0,
            max(metric_floor, metric_multiplier * max(0.0, baseline)),
            "nonnegative",
        )
    delta = max(metric_floor, abs(baseline) * metric_multiplier)
    return (baseline - delta, baseline + delta, "signed")


def derive_thresholds_from_artifacts(
    base_dir: Path,
    *,
    profile: str,
    threshold_profile: str = "ci_conservative",
    runtime_multiplier: float = 8.0,
    runtime_floor: float = 0.5,
    metric_multiplier: float = 4.0,
    metric_floor: float = 0.25,
    prob_delta: float = 0.35,
    r2_delta: float = 1.5,
) -> dict[str, Any]:
    limits: dict[str, dict[str, Any]] = {}
    files = _summary_paths(base_dir, profile)
    for path in files:
        payload = load_json(path)
        example = _stem_without_profile(path, profile)
        for method, key, value in _iter_numeric_row_metrics(payload):
            lo, hi, policy = _derive_bounds(
                key,
                value,
                runtime_multiplier=runtime_multiplier,
                runtime_floor=runtime_floor,
                metric_multiplier=metric_multiplier,
                metric_floor=metric_floor,
                prob_delta=prob_delta,
                r2_delta=r2_delta,
            )
            case_key = f"{example}|{method}|{key}"
            limits[case_key] = {
                "example": example,
                "method": method,
                "metric": key,
                "baseline": value,
                "min": lo,
                "max": hi,
                "policy": policy,
            }
    return {
        "artifact": "example_summary_thresholds",
        "version": 1,
        "target_profile": profile,
        "threshold_profile": threshold_profile,
        "n_artifacts": len(files),
        "n_limits": len(limits),
        "limits": limits,
        "config": {
            "runtime_multiplier": runtime_multiplier,
            "runtime_floor": runtime_floor,
            "metric_multiplier": metric_multiplier,
            "metric_floor": metric_floor,
            "prob_delta": prob_delta,
            "r2_delta": r2_delta,
        },
    }


def evaluate_artifacts_against_thresholds(
    base_dir: Path,
    *,
    profile: str,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    limits = thresholds.get("limits", {})
    failures: list[dict[str, Any]] = []
    checked = 0
    seen: set[str] = set()

    for path in _summary_paths(base_dir, profile):
        payload = load_json(path)
        example = _stem_without_profile(path, profile)
        for method, key, value in _iter_numeric_row_metrics(payload):
            case_key = f"{example}|{method}|{key}"
            seen.add(case_key)
            spec = limits.get(case_key)
            if not isinstance(spec, dict):
                continue
            checked += 1
            lo = spec.get("min")
            hi = spec.get("max")
            if isinstance(lo, (int, float)) and value < float(lo):
                failures.append(
                    {
                        "case_key": case_key,
                        "value": value,
                        "limit": float(lo),
                        "direction": "below-min",
                    }
                )
            if isinstance(hi, (int, float)) and value > float(hi):
                failures.append(
                    {
                        "case_key": case_key,
                        "value": value,
                        "limit": float(hi),
                        "direction": "above-max",
                    }
                )

    missing_limits = sorted(set(limits) - seen)
    return {
        "artifact": "example_summary_threshold_verdict",
        "version": 1,
        "target_profile": profile,
        "threshold_target_profile": thresholds.get("target_profile"),
        "threshold_profile": thresholds.get("threshold_profile"),
        "ok": not failures and not missing_limits,
        "checked_limits": checked,
        "total_limits": len(limits),
        "failed_limits": len(failures),
        "missing_limits": len(missing_limits),
        "failures": failures[:200],
        "missing_limit_keys_sample": missing_limits[:50],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive/evaluate thresholds for example summary artifacts."
    )
    parser.add_argument("--base-dir", type=Path, default=Path("reports/example_summaries"))
    parser.add_argument("--profile", default="full")
    parser.add_argument(
        "--threshold-profile",
        default="ci_conservative",
        help="Label describing strictness policy (e.g., ci_conservative, review_strict).",
    )
    parser.add_argument("--write-thresholds", type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--output-verdict", type=Path)
    parser.add_argument("--runtime-multiplier", type=float, default=8.0)
    parser.add_argument("--runtime-floor", type=float, default=0.5)
    parser.add_argument("--metric-multiplier", type=float, default=4.0)
    parser.add_argument("--metric-floor", type=float, default=0.25)
    parser.add_argument("--prob-delta", type=float, default=0.35)
    parser.add_argument("--r2-delta", type=float, default=1.5)
    parser.add_argument(
        "--fail-on-thresholds",
        action="store_true",
        help="Exit non-zero if threshold evaluation fails.",
    )
    args = parser.parse_args()

    if args.write_thresholds is not None:
        thresholds = derive_thresholds_from_artifacts(
            args.base_dir,
            profile=args.profile,
            threshold_profile=args.threshold_profile,
            runtime_multiplier=args.runtime_multiplier,
            runtime_floor=args.runtime_floor,
            metric_multiplier=args.metric_multiplier,
            metric_floor=args.metric_floor,
            prob_delta=args.prob_delta,
            r2_delta=args.r2_delta,
        )
        args.write_thresholds.parent.mkdir(parents=True, exist_ok=True)
        args.write_thresholds.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
        print(f"Wrote thresholds: {args.write_thresholds}")

    if args.thresholds is not None:
        thresholds = load_json(args.thresholds)
        verdict = evaluate_artifacts_against_thresholds(
            args.base_dir,
            profile=args.profile,
            thresholds=thresholds,
        )
        if args.output_verdict is not None:
            args.output_verdict.parent.mkdir(parents=True, exist_ok=True)
            args.output_verdict.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
            print(f"Wrote verdict: {args.output_verdict}")
        print(
            "Threshold verdict:",
            f"ok={verdict['ok']}",
            f"checked={verdict['checked_limits']}",
            f"failed={verdict['failed_limits']}",
            f"missing={verdict['missing_limits']}",
        )
        if args.fail_on_thresholds and not verdict["ok"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
