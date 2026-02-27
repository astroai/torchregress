"""Merge torchregress photo-z summaries with RAIL baseline outputs.

This tool keeps RAIL integration optional and artifact-driven. It accepts either:

1) `rail_photoz_summary` payloads that already provide summary rows, or
2) `rail_photoz_predictions` payloads with per-object point/PDF predictions, then
   computes summary metrics for fair comparison against torchregress rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

CORE_BASELINES = ("flexzboost", "pzflow", "delight", "bpz")
OPTIONAL_BASELINES = ("lephare",)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _nmad_and_catastrophic(y_pred: torch.Tensor, y_true: torch.Tensor) -> tuple[float, float]:
    residual = (y_pred - y_true) / (1.0 + y_true.clamp_min(0.0))
    med = torch.median(residual)
    nmad = 1.48 * torch.median(torch.abs(residual - med))
    catastrophic = (torch.abs(y_pred - y_true) > (0.15 * (1.0 + y_true))).float().mean()
    return float(nmad.item()), float(catastrophic.item())


def _high_z_mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    q80 = torch.quantile(y_true, 0.80)
    mask = y_true >= q80
    if bool(mask.any()):
        return float(torch.mean(torch.abs(y_pred[mask] - y_true[mask])).item())
    return float("nan")


def _bin_targets(targets: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    return torch.searchsorted(edges[1:], targets).clamp(min=0, max=edges.numel() - 2).long()


def _pdf_quantile(pdf: torch.Tensor, edges: torch.Tensor, q: float) -> torch.Tensor:
    cdf = torch.cumsum(pdf, dim=-1)
    idx = torch.argmax((cdf >= q).long(), dim=-1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers[idx]


def _pdf_metrics(
    pdf: torch.Tensor,
    y_true: torch.Tensor,
    edges: torch.Tensor,
) -> tuple[float, float, float, float, float]:
    y_bin = _bin_targets(y_true, edges)
    probs = pdf[torch.arange(pdf.shape[0], device=pdf.device), y_bin].clamp_min(1e-12)
    pdf_nll = float((-torch.log(probs)).mean().item())

    widths = (edges[1:] - edges[:-1]).clamp_min(1e-8)
    cdf_pred = torch.cumsum(pdf, dim=-1)
    grid = torch.arange(pdf.shape[-1], device=pdf.device).unsqueeze(0)
    cdf_true = (grid >= y_bin.unsqueeze(1)).to(pdf.dtype)
    crps = float(
        torch.mean(torch.sum((cdf_pred - cdf_true) ** 2 * widths.unsqueeze(0), dim=-1)).item()
    )

    pit = cdf_pred[torch.arange(cdf_pred.shape[0], device=pdf.device), y_bin].clamp(0.0, 1.0)
    hist_edges = torch.linspace(0.0, 1.0, 21, device=pdf.device)
    counts = torch.histogram(pit, hist_edges)[0].float()
    expected = pit.numel() / 20.0
    pit_chi2 = float(torch.sum((counts - expected) ** 2 / max(expected, 1.0)).item())

    q05 = _pdf_quantile(pdf, edges, 0.05)
    q95 = _pdf_quantile(pdf, edges, 0.95)
    lower, upper = torch.minimum(q05, q95), torch.maximum(q05, q95)
    cov90 = float(((y_true >= lower) & (y_true <= upper)).float().mean().item())
    width90 = float(torch.mean(upper - lower).item())
    return pdf_nll, crps, pit_chi2, cov90, width90


def _summary_row_from_predictions(payload: dict[str, Any]) -> dict[str, Any]:
    method = str(payload.get("method", "unknown"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("rail_photoz_predictions payload must contain non-empty 'rows'.")

    y_true_vals: list[float] = []
    y_pred_vals: list[float] = []
    pdf_rows: list[list[float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        y_true_vals.append(float(row["z_true"]))
        y_pred_vals.append(float(row["z_point"]))
        pdf_val = row.get("pdf")
        if isinstance(pdf_val, list):
            pdf_rows.append([float(x) for x in pdf_val])

    y_true = torch.tensor(y_true_vals, dtype=torch.float32)
    y_pred = torch.tensor(y_pred_vals, dtype=torch.float32)
    nmad, catastrophic = _nmad_and_catastrophic(y_pred, y_true)
    highz_mae = _high_z_mae(y_pred, y_true)

    out: dict[str, Any] = {
        "Method": method,
        "Framework": "RAIL",
        "NMAD": nmad,
        "CatastrophicRate": catastrophic,
        "HighZ_MAE": highz_mae,
        "CRPS": None,
        "PDF_NLL": None,
        "PITChi2": None,
        "NativeCov90": None,
        "NativeWidth90": None,
        "train_s": payload.get("train_s"),
        "eval_s": payload.get("eval_s"),
        "Notes": "RAIL baseline",
    }

    if pdf_rows:
        edges_val = payload.get("bin_edges")
        if not isinstance(edges_val, list):
            first_row = rows[0]
            if isinstance(first_row, dict):
                edges_val = first_row.get("bin_edges")
        if not isinstance(edges_val, list):
            raise ValueError("RAIL PDF payload requires `bin_edges` at payload or row level.")
        edges = torch.tensor([float(x) for x in edges_val], dtype=torch.float32)
        pdf = torch.tensor(pdf_rows, dtype=torch.float32)
        pdf = pdf / pdf.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        pdf_nll, crps, pit_chi2, cov90, width90 = _pdf_metrics(pdf, y_true, edges)
        out.update(
            {
                "CRPS": crps,
                "PDF_NLL": pdf_nll,
                "PITChi2": pit_chi2,
                "NativeCov90": cov90,
                "NativeWidth90": width90,
            }
        )
    return out


def _extract_rail_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    artifact = payload.get("artifact")
    if artifact == "rail_photoz_summary":
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError("rail_photoz_summary payload must include a rows list.")
        out: list[dict[str, Any]] = []
        for row in rows:
            if isinstance(row, dict):
                rr = dict(row)
                rr.setdefault("Framework", "RAIL")
                out.append(rr)
        return out
    if artifact == "rail_photoz_predictions":
        return [_summary_row_from_predictions(payload)]
    raise ValueError(
        "Unsupported RAIL payload artifact. Expected `rail_photoz_summary` "
        "or `rail_photoz_predictions`."
    )


def _check_manifest_parity(
    *,
    manifest: dict[str, Any],
    payloads: list[dict[str, Any]],
    paper_parity: bool,
) -> list[str]:
    notes: list[str] = []
    if not paper_parity:
        notes.append("paper_parity_mode_disabled")
        return notes

    expected_dataset = str(manifest.get("dataset_id", "")).strip()
    expected_split = str(manifest.get("split_id", "")).strip()
    if not expected_dataset or not expected_split:
        raise ValueError("Manifest requires dataset_id and split_id for paper parity mode.")

    for payload in payloads:
        dataset_id = str(payload.get("dataset_id", "")).strip()
        split_id = str(payload.get("split_id", "")).strip()
        if dataset_id != expected_dataset or split_id != expected_split:
            raise ValueError(
                "Paper parity mode failed: payload dataset/split mismatch "
                f"(expected dataset={expected_dataset}, split={expected_split}, "
                f"got dataset={dataset_id}, split={split_id})."
            )

    required = {m.lower() for m in manifest.get("core_baselines", CORE_BASELINES)}
    seen: set[str] = set()
    for payload in payloads:
        method = payload.get("method")
        if isinstance(method, str):
            seen.add(method.lower())
        rows = payload.get("rows")
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and isinstance(row.get("Method"), str):
                    seen.add(str(row["Method"]).lower())

    missing = sorted(required - seen)
    if missing:
        raise ValueError(
            "Paper parity mode failed: missing required RAIL baseline methods: "
            + ", ".join(missing)
        )
    optional_seen = sorted({m for m in OPTIONAL_BASELINES if m in seen})
    notes.append(f"optional_baselines_present={optional_seen}")
    return notes


def merge_summaries(
    *,
    manifest: dict[str, Any],
    torchregress_summary: dict[str, Any],
    rail_payloads: list[dict[str, Any]],
    paper_parity: bool,
) -> dict[str, Any]:
    if torchregress_summary.get("artifact") != "comparison_example_summary":
        raise ValueError("torchregress summary must be `comparison_example_summary`.")

    parity_notes = _check_manifest_parity(
        manifest=manifest,
        payloads=rail_payloads,
        paper_parity=paper_parity,
    )
    rows = torchregress_summary.get("rows")
    if not isinstance(rows, list):
        raise ValueError("torchregress summary rows must be a list.")
    merged_rows: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            rr = dict(row)
            rr.setdefault("Framework", "torchregress")
            merged_rows.append(rr)

    for payload in rail_payloads:
        merged_rows.extend(_extract_rail_rows(payload))

    return {
        "artifact": "comparison_example_summary",
        "version": 1,
        "example": "tools/photoz_rail_compare.py",
        "task": "Photo-z tabular benchmark (RAIL baselines vs torchregress)",
        "config": {
            "manifest_dataset_id": manifest.get("dataset_id"),
            "manifest_split_id": manifest.get("split_id"),
            "paper_parity_mode": paper_parity,
            "core_baselines": list(manifest.get("core_baselines", CORE_BASELINES)),
            "optional_baselines": list(manifest.get("optional_baselines", OPTIONAL_BASELINES)),
        },
        "rows": merged_rows,
        "notes": [
            "Merged torchregress and RAIL tabular baseline summaries for audit comparison.",
            "RAIL baseline core set: flexzboost, pzflow, delight, bpz (lephare optional).",
            *parity_notes,
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge RAIL and torchregress photo-z summaries.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/rail/rail_photoz_manifest.json"),
    )
    parser.add_argument("--torchregress-summary", type=Path, required=True)
    parser.add_argument("--rail-inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--paper-parity",
        action="store_true",
        default=True,
        help="Enforce dataset/split/method parity against manifest (default: true).",
    )
    parser.add_argument(
        "--no-paper-parity",
        action="store_false",
        dest="paper_parity",
        help="Disable strict manifest parity checks.",
    )
    args = parser.parse_args()

    manifest = _load_json(args.manifest)
    tr_summary = _load_json(args.torchregress_summary)
    rail_payloads = [_load_json(path) for path in args.rail_inputs]
    merged = merge_summaries(
        manifest=manifest,
        torchregress_summary=tr_summary,
        rail_payloads=rail_payloads,
        paper_parity=args.paper_parity,
    )
    _save_json(args.output, merged)
    print(f"Wrote merged summary: {args.output}")
    print(f"Rows: {len(merged.get('rows', []))}")


if __name__ == "__main__":
    main()
