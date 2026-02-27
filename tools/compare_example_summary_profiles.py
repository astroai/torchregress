"""Compare comparison-example summary artifacts across profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tools import render_example_summaries
except ModuleNotFoundError:  # pragma: no cover - script execution path
    import render_example_summaries  # type: ignore[no-redef]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_by_method(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("rows", [])
    return {str(row.get("Method")): row for row in rows if isinstance(row, dict)}


def _numeric_metric_keys(row: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for k, v in row.items():
        if k == "Method":
            continue
        if isinstance(v, (int, float)):
            out.add(k)
    return out


def _extract_budget_signals(payload: dict[str, Any]) -> dict[str, float]:
    cfg = payload.get("config", {})
    if not isinstance(cfg, dict):
        return {}
    out: dict[str, float] = {}
    for key in (
        "epochs",
        "n_train",
        "n_test",
        "n_cal",
        "ensemble_size",
        "mc_samples",
        "eval_samples",
    ):
        value = cfg.get(key)
        if isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def _is_probability_key(key: str) -> bool:
    k = key.lower()
    return (
        (("cov" in k or "coverage" in k) and "covariance" not in k)
        or "contains_true" in k
        or k in {"overlaprate"}
    )


def _row_domain_issues(row: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    method = str(row.get("Method"))
    allow_nan_keys = {"EvalLossOnObs"}
    for key, value in row.items():
        if key == "Method" or value is None or isinstance(value, str):
            continue
        if not isinstance(value, (int, float)):
            issues.append(f"{method}:{key}:non-numeric")
            continue
        v = float(value)
        if v != v:  # NaN
            if key in allow_nan_keys:
                continue
            issues.append(f"{method}:{key}:nan")
            continue
        if key.endswith("_s") and v < 0.0:
            issues.append(f"{method}:{key}:negative-runtime")
        if _is_probability_key(key) and not (0.0 <= v <= 1.0):
            issues.append(f"{method}:{key}:probability-range")
        if (
            any(
                token in key.lower()
                for token in ("mse", "mae", "nll", "loss", "width", "energy", "aurc", "risk")
            )
            and v < 0.0
        ):
            issues.append(f"{method}:{key}:negative-metric")
    return issues


def _payload_semantic_issues(payload: dict[str, Any]) -> list[str]:
    rows = payload.get("rows", [])
    out: list[str] = []
    if not isinstance(rows, list):
        return ["rows:not-list"]
    for row in rows:
        if isinstance(row, dict):
            out.extend(_row_domain_issues(row))
        else:
            out.append("row:not-dict")
    return out


def _sum_runtime(rows: dict[str, dict[str, Any]], *, key: str) -> float | None:
    total = 0.0
    seen = 0
    for row in rows.values():
        value = row.get(key)
        if isinstance(value, (int, float)):
            total += float(value)
            seen += 1
    return total if seen else None


_LOWER_BETTER_KEYS = {
    "MSE",
    "MAE",
    "NLL",
    "Energy",
    "MCE",
    "AURC",
    "CleanMSE",
    "CleanMAE",
    "ObsMSE",
    "ConformalWidth90",
    "ConformalIS90",
    "NativeWidth90",
    "rej20_risk",
    "clean_mse",
    "obs_mse",
    "obs_input_clean_target_mse",
    "OrdinalMAE",
    "MAE_true",
    "ObsMAE",
    "TailMAE90",
    "TailRMSE90",
    "CrossingRate",
    "BoundViolation",
    "PITChi2",
    "NoisyTargetNLL",
    "ConsistencyLoss",
    "PseudoLabelNLL",
    "ATE_abs_error",
    "CI_width",
}

_UPPER_BETTER_KEYS = {
    "rej20_cov",
    "R2",
    "ood_unc_gap",
    "Accuracy",
    "QWK",
    "CIndex",
    "CI_contains_true",
}


def _directionality_issues(
    *,
    task: str,
    source_rows: dict[str, dict[str, Any]],
    target_rows: dict[str, dict[str, Any]],
) -> list[str]:
    """Detect catastrophic target-profile regressions on selected metrics.

    This is intentionally permissive. It should catch obvious breakage/noise explosions,
    not enforce strict monotonic improvement with larger budgets.
    """
    issues: list[str] = []
    task_lower = task.lower()
    for method in sorted(set(source_rows) & set(target_rows)):
        srow = source_rows[method]
        trow = target_rows[method]
        for key in sorted(_LOWER_BETTER_KEYS):
            sv = srow.get(key)
            tv = trow.get(key)
            if not isinstance(sv, (int, float)) or not isinstance(tv, (int, float)):
                continue
            s = float(sv)
            t = float(tv)
            if s != s or t != t:  # NaN
                continue
            # Ignore tiny values where ratios are unstable/noisy.
            if abs(s) < 1e-6:
                continue
            max_allowed = max(0.25, 8.0 * abs(s))
            if t > max_allowed:
                issues.append(f"{method}:{key}:catastrophic-increase:{s:.4g}->{t:.4g}")
        for key in sorted(_UPPER_BETTER_KEYS):
            sv = srow.get(key)
            tv = trow.get(key)
            if not isinstance(sv, (int, float)) or not isinstance(tv, (int, float)):
                continue
            s = float(sv)
            t = float(tv)
            if s != s or t != t:
                continue
            if key in {"ConformalCov90", "NativeCov90", "rej20_cov"}:
                if t < s - 0.35:
                    issues.append(f"{method}:{key}:large-drop:{s:.4g}->{t:.4g}")
            elif key == "R2":
                if t < s - 1.0:
                    issues.append(f"{method}:{key}:large-drop:{s:.4g}->{t:.4g}")
            elif key == "ood_unc_gap":
                if t < -0.25:
                    issues.append(f"{method}:{key}:negative-gap:{s:.4g}->{t:.4g}")

        # Tradeoff-aware checks: large coverage drops are only concerning if
        # intervals did not narrow.
        for cov_key, width_key in (
            ("ConformalCov90", "ConformalWidth90"),
            ("NativeCov90", "NativeWidth90"),
        ):
            sv = srow.get(cov_key)
            tv = trow.get(cov_key)
            sw = srow.get(width_key)
            tw = trow.get(width_key)
            if not all(isinstance(v, (int, float)) for v in (sv, tv, sw, tw)):
                continue
            s_cov = float(sv)
            t_cov = float(tv)
            s_w = float(sw)
            t_w = float(tw)
            if any(v != v for v in (s_cov, t_cov, s_w, t_w)):  # NaN
                continue
            if s_cov - t_cov > 0.35:
                width_shrank = s_w > 0 and (t_w <= 0.9 * s_w)
                if not width_shrank:
                    issues.append(
                        f"{method}:{cov_key}:large-drop-without-width-tradeoff:"
                        f"{s_cov:.4g}->{t_cov:.4g}"
                    )

        # Task-specific OOD sanity: if OOD error is much worse than ID error, a strongly
        # negative OOD uncertainty gap likely indicates a broken uncertainty signal.
        if "ood" in task_lower:
            mse_id = trow.get("MSE_ID")
            mse_ood = trow.get("MSE_OOD")
            unc_gap = trow.get("ood_unc_gap")
            if (
                isinstance(mse_id, (int, float))
                and isinstance(mse_ood, (int, float))
                and isinstance(unc_gap, (int, float))
            ):
                mse_id_f = float(mse_id)
                mse_ood_f = float(mse_ood)
                unc_gap_f = float(unc_gap)
                if all(v == v for v in (mse_id_f, mse_ood_f, unc_gap_f)):
                    if mse_ood_f > max(0.25, 1.5 * mse_id_f) and unc_gap_f < -0.15:
                        issues.append(
                            f"{method}:ood_unc_gap:ood-error-gap-mismatch:"
                            f"id={mse_id_f:.4g},ood={mse_ood_f:.4g},gap={unc_gap_f:.4g}"
                        )

        # Task-specific multimodal sanity: if calibration error explodes and NLL also
        # blows up versus audit, flag it as a likely broken training/eval path.
        if "multimodal" in task_lower:
            s_nll = srow.get("NLL")
            t_nll = trow.get("NLL")
            t_mce = trow.get("MCE")
            if (
                isinstance(s_nll, (int, float))
                and isinstance(t_nll, (int, float))
                and isinstance(t_mce, (int, float))
            ):
                s_nll_f = float(s_nll)
                t_nll_f = float(t_nll)
                t_mce_f = float(t_mce)
                if all(v == v for v in (s_nll_f, t_nll_f, t_mce_f)):
                    if t_mce_f > 1.0 and t_nll_f > max(10.0, 8.0 * max(0.25, abs(s_nll_f))):
                        issues.append(
                            f"{method}:multimodal_nll_mce:blowup:"
                            f"nll={s_nll_f:.4g}->{t_nll_f:.4g},mce={t_mce_f:.4g}"
                        )

        # Task-specific noisy-label sanity: if observed-label error is much worse than
        # clean-target error, interval coverage should not collapse across both
        # conformal and native intervals without any width tradeoff.
        if "noisy" in task_lower and "label" in task_lower:
            clean_mse = trow.get("CleanMSE")
            obs_mse = trow.get("ObsMSE")
            conf_cov = trow.get("ConformalCov90")
            native_cov = trow.get("NativeCov90")
            conf_w = trow.get("ConformalWidth90")
            native_w = trow.get("NativeWidth90")
            s_conf_w = srow.get("ConformalWidth90")
            s_native_w = srow.get("NativeWidth90")
            if all(
                isinstance(v, (int, float))
                for v in (
                    clean_mse,
                    obs_mse,
                    conf_cov,
                    native_cov,
                    conf_w,
                    native_w,
                    s_conf_w,
                    s_native_w,
                )
            ):
                clean_mse_f = float(clean_mse)
                obs_mse_f = float(obs_mse)
                conf_cov_f = float(conf_cov)
                native_cov_f = float(native_cov)
                conf_w_f = float(conf_w)
                native_w_f = float(native_w)
                s_conf_w_f = float(s_conf_w)
                s_native_w_f = float(s_native_w)
                if all(
                    v == v
                    for v in (
                        clean_mse_f,
                        obs_mse_f,
                        conf_cov_f,
                        native_cov_f,
                        conf_w_f,
                        native_w_f,
                        s_conf_w_f,
                        s_native_w_f,
                    )
                ):
                    noisy_gap_large = obs_mse_f > max(0.25, 2.0 * clean_mse_f)
                    dual_coverage_collapse = conf_cov_f < 0.45 and native_cov_f < 0.45
                    width_tradeoff_present = (
                        s_conf_w_f > 0.0 and conf_w_f <= 0.75 * s_conf_w_f
                    ) or (s_native_w_f > 0.0 and native_w_f <= 0.75 * s_native_w_f)
                    if noisy_gap_large and dual_coverage_collapse and not width_tradeoff_present:
                        issues.append(
                            f"{method}:noisy_label_cov_collapse:mismatch:"
                            f"clean={clean_mse_f:.4g},obs={obs_mse_f:.4g},"
                            f"conf_cov={conf_cov_f:.4g},native_cov={native_cov_f:.4g}"
                        )

        # Task-specific photo-z sanity: if multiple photo-z quality metrics blow up at once,
        # flag it as a likely broken training/eval path rather than benign budget noise.
        if "photo-z" in task_lower or "photometric redshift" in task_lower:
            s_nmad = srow.get("NMAD")
            t_nmad = trow.get("NMAD")
            s_cat = srow.get("CatastrophicRate")
            t_cat = trow.get("CatastrophicRate")
            s_highz = srow.get("HighZ_MAE")
            t_highz = trow.get("HighZ_MAE")
            if all(
                isinstance(v, (int, float))
                for v in (s_nmad, t_nmad, s_cat, t_cat, s_highz, t_highz)
            ):
                s_nmad_f = float(s_nmad)
                t_nmad_f = float(t_nmad)
                s_cat_f = float(s_cat)
                t_cat_f = float(t_cat)
                s_highz_f = float(s_highz)
                t_highz_f = float(t_highz)
                if all(
                    v == v for v in (s_nmad_f, t_nmad_f, s_cat_f, t_cat_f, s_highz_f, t_highz_f)
                ):
                    nmad_blowup = t_nmad_f > max(0.03, 2.5 * max(0.01, s_nmad_f))
                    cat_blowup = t_cat_f > min(1.0, s_cat_f + 0.35)
                    highz_blowup = t_highz_f > max(0.05, 2.5 * max(0.02, s_highz_f))
                    if nmad_blowup and (cat_blowup or highz_blowup):
                        issues.append(
                            f"{method}:photoz_quality:blowup:"
                            f"NMAD={s_nmad_f:.4g}->{t_nmad_f:.4g},"
                            f"Cat={s_cat_f:.4g}->{t_cat_f:.4g},"
                            f"HighZ_MAE={s_highz_f:.4g}->{t_highz_f:.4g}"
                        )
            # Ordered-bin photo-z sanity: if both CRPS and PDF_NLL blow up heavily
            # in target profile, flag likely regression in binned-PDF path.
            s_crps = srow.get("CRPS")
            t_crps = trow.get("CRPS")
            s_pdf_nll = srow.get("PDF_NLL")
            t_pdf_nll = trow.get("PDF_NLL")
            if all(isinstance(v, (int, float)) for v in (s_crps, t_crps, s_pdf_nll, t_pdf_nll)):
                s_crps_f = float(s_crps)
                t_crps_f = float(t_crps)
                s_pdf_nll_f = float(s_pdf_nll)
                t_pdf_nll_f = float(t_pdf_nll)
                if all(v == v for v in (s_crps_f, t_crps_f, s_pdf_nll_f, t_pdf_nll_f)):
                    crps_blowup = t_crps_f > max(0.03, 3.0 * max(0.01, s_crps_f))
                    nll_blowup = t_pdf_nll_f > max(0.1, 3.0 * max(0.02, s_pdf_nll_f))
                    if crps_blowup and nll_blowup:
                        issues.append(
                            f"{method}:photoz_pdf_quality:blowup:"
                            f"CRPS={s_crps_f:.4g}->{t_crps_f:.4g},"
                            f"PDF_NLL={s_pdf_nll_f:.4g}->{t_pdf_nll_f:.4g}"
                        )
    return issues


def compare_profiles(
    *,
    base_dir: Path,
    source_profile: str,
    target_profile: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for example_name, spec in render_example_summaries.EXAMPLE_SPECS.items():
        stem = spec["filename"]
        source_path = base_dir / f"{stem}_{source_profile}.json"
        target_path = base_dir / f"{stem}_{target_profile}.json"
        source = _load_json(source_path)
        target = _load_json(target_path)
        task_name = str(target.get("task") or source.get("task") or example_name)
        source_rows = _rows_by_method(source)
        target_rows = _rows_by_method(target)
        source_methods = set(source_rows)
        target_methods = set(target_rows)
        common = sorted(source_methods & target_methods)

        numeric_key_coverage_ok = True
        for method in common:
            source_numeric = _numeric_metric_keys(source_rows[method])
            target_numeric = _numeric_metric_keys(target_rows[method])
            if not source_numeric.issubset(target_numeric):
                numeric_key_coverage_ok = False
                break

        source_budget = _extract_budget_signals(source)
        target_budget = _extract_budget_signals(target)
        monotone_budget_keys: list[str] = []
        for key, source_val in source_budget.items():
            target_val = target_budget.get(key)
            if target_val is None:
                continue
            if target_val >= source_val:
                monotone_budget_keys.append(key)

        source_semantic_issues = _payload_semantic_issues(source)
        target_semantic_issues = _payload_semantic_issues(target)
        directionality_issues = _directionality_issues(
            task=task_name,
            source_rows=source_rows,
            target_rows=target_rows,
        )
        source_train_total = _sum_runtime(source_rows, key="train_s")
        target_train_total = _sum_runtime(target_rows, key="train_s")
        source_eval_total = _sum_runtime(source_rows, key="eval_s")
        target_eval_total = _sum_runtime(target_rows, key="eval_s")
        runtime_scaling_ok = True
        runtime_scaling_notes: list[str] = []
        if source_train_total is not None and target_train_total is not None:
            # Ignore tiny totals where timer noise dominates.
            if source_train_total >= 0.01 and target_train_total < 0.25 * source_train_total:
                runtime_scaling_ok = False
                runtime_scaling_notes.append("train_s_total_scaled_too_low")
        if source_eval_total is not None and target_eval_total is not None:
            if source_eval_total >= 0.01 and target_eval_total < 0.20 * source_eval_total:
                runtime_scaling_ok = False
                runtime_scaling_notes.append("eval_s_total_scaled_too_low")

        rows.append(
            {
                "example": example_name,
                "source_profile": source_profile,
                "target_profile": target_profile,
                "method_set_equal": source_methods == target_methods,
                "source_method_count": len(source_methods),
                "target_method_count": len(target_methods),
                "numeric_key_coverage_ok": numeric_key_coverage_ok,
                "source_semantic_ok": not source_semantic_issues,
                "target_semantic_ok": not target_semantic_issues,
                "directionality_ok": not directionality_issues,
                "directionality_issue_count": len(directionality_issues),
                "directionality_issues": directionality_issues[:10],
                "source_semantic_issue_count": len(source_semantic_issues),
                "target_semantic_issue_count": len(target_semantic_issues),
                "source_semantic_issues": source_semantic_issues[:10],
                "target_semantic_issues": target_semantic_issues[:10],
                "runtime_scaling_ok": runtime_scaling_ok,
                "runtime_scaling_notes": runtime_scaling_notes,
                "source_train_s_total": source_train_total,
                "target_train_s_total": target_train_total,
                "source_eval_s_total": source_eval_total,
                "target_eval_s_total": target_eval_total,
                "monotone_budget_keys": monotone_budget_keys,
                "source_path": str(source_path),
                "target_path": str(target_path),
            }
        )

    ok = all(
        row["method_set_equal"]
        and row["numeric_key_coverage_ok"]
        and row["source_semantic_ok"]
        and row["target_semantic_ok"]
        and row["directionality_ok"]
        and row["runtime_scaling_ok"]
        and bool(row["monotone_budget_keys"])
        for row in rows
    )
    return {
        "artifact": "example_summary_profile_comparison",
        "version": 1,
        "source_profile": source_profile,
        "target_profile": target_profile,
        "ok": ok,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare example summary artifacts across profiles."
    )
    parser.add_argument("--base-dir", type=Path, default=Path("reports/example_summaries"))
    parser.add_argument("--source-profile", default="audit")
    parser.add_argument("--target-profile", default="full")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = compare_profiles(
        base_dir=args.base_dir,
        source_profile=args.source_profile,
        target_profile=args.target_profile,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote profile comparison report: {args.output}")
    print(f"OK: {report['ok']}")


if __name__ == "__main__":
    main()
