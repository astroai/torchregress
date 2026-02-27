"""Collect real photo-z datasets without manual URL hunting.

This tool supports two sources:

1) Rubin DP0.2 simulated-but-realistic sample via TAP query
2) NNC paper released catalog via Zenodo record API
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DP02_TAP_SYNC_URL = "https://data.lsst.cloud/api/tap/sync"
DEFAULT_NNC_ZENODO_RECORD = 18410731


def _http_get(url: str, *, headers: dict[str, str] | None = None) -> bytes:
    request = Request(url, headers=headers or {})
    with urlopen(request) as response:  # nosec: URLs are user-selected public endpoints
        return cast(bytes, response.read())


def _basic_auth_header(username: str, password: str) -> str:
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def _build_dp02_adql(
    *,
    limit: int,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    radius_deg: float | None = None,
) -> str:
    where_parts = [
        "ts.truth_type = 1",
        "obj.detect_isPrimary = 1",
        "obj.u_cModelFlux > 0",
        "obj.g_cModelFlux > 0",
        "obj.r_cModelFlux > 0",
        "obj.i_cModelFlux > 0",
        "obj.z_cModelFlux > 0",
    ]
    if ra_deg is not None and dec_deg is not None and radius_deg is not None:
        where_parts.append(
            "CONTAINS(POINT('ICRS', obj.coord_ra, obj.coord_dec), "
            f"CIRCLE('ICRS', {ra_deg:.8f}, {dec_deg:.8f}, {radius_deg:.8f})) = 1"
        )
    where_clause = " AND ".join(where_parts)

    return (
        f"SELECT TOP {int(limit)} "
        "obj.objectId AS object_id, "
        "obj.coord_ra AS ra_deg, "
        "obj.coord_dec AS dec_deg, "
        "obj.u_cModelFlux AS flux_u, "
        "obj.g_cModelFlux AS flux_g, "
        "obj.r_cModelFlux AS flux_r, "
        "obj.i_cModelFlux AS flux_i, "
        "obj.z_cModelFlux AS flux_z, "
        "ts.redshift AS spec_z "
        "FROM dp02_dc2_catalogs.MatchesTruth AS mt "
        "JOIN dp02_dc2_catalogs.TruthSummary AS ts "
        "ON mt.id_truth_type = ts.id_truth_type "
        "JOIN dp02_dc2_catalogs.Object AS obj "
        "ON mt.match_objectId = obj.objectId "
        f"WHERE {where_clause}"
    )


def _run_dp02_tap_query(
    *,
    tap_sync_url: str,
    adql_query: str,
    token: str,
) -> pd.DataFrame:
    params = urlencode(
        {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "csv",
            "QUERY": adql_query,
        }
    )
    request_url = f"{tap_sync_url}?{params}"
    payload = _http_get(
        request_url,
        headers={"Authorization": _basic_auth_header("x-oauth-basic", token)},
    )
    text = payload.decode("utf-8", errors="replace")
    if text.lstrip().startswith("<"):
        raise RuntimeError(
            "DP0.2 TAP returned XML/error content instead of CSV. "
            "Check Rubin token, data-rights access, and query limits."
        )
    return pd.read_csv(io.StringIO(text))


def _flux_to_abmag(flux_njy: pd.Series) -> pd.Series:
    flux = pd.to_numeric(flux_njy, errors="coerce").astype(float)
    mag = pd.Series(np.nan, index=flux.index, dtype=float)
    positive = flux > 0
    mag.loc[positive] = 22.5 - 2.5 * np.log10(flux.loc[positive])
    return mag


def _convert_dp02_to_photoz_frame(dp02_df: pd.DataFrame) -> pd.DataFrame:
    needed = {"object_id", "flux_u", "flux_g", "flux_r", "flux_i", "flux_z", "spec_z"}
    missing = sorted(needed - set(dp02_df.columns))
    if missing:
        raise ValueError(f"DP0.2 payload missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "objid": dp02_df["object_id"],
            "spec_z": pd.to_numeric(dp02_df["spec_z"], errors="coerce"),
            "spec_z_err": 0.01,
            "u": _flux_to_abmag(dp02_df["flux_u"]),
            "g": _flux_to_abmag(dp02_df["flux_g"]),
            "r": _flux_to_abmag(dp02_df["flux_r"]),
            "i": _flux_to_abmag(dp02_df["flux_i"]),
            "z_mag": _flux_to_abmag(dp02_df["flux_z"]),
            "u_err": 0.03,
            "g_err": 0.02,
            "r_err": 0.02,
            "i_err": 0.02,
            "z_mag_err": 0.03,
        }
    )

    out = out.dropna(subset=["spec_z", "u", "g", "r", "i", "z_mag"]).reset_index(drop=True)
    out["u_g"] = out["u"] - out["g"]
    out["g_r"] = out["g"] - out["r"]
    out["r_i"] = out["r"] - out["i"]
    out["i_z"] = out["i"] - out["z_mag"]
    out["u_g_err"] = np.sqrt(out["u_err"] ** 2 + out["g_err"] ** 2)
    out["g_r_err"] = np.sqrt(out["g_err"] ** 2 + out["r_err"] ** 2)
    out["r_i_err"] = np.sqrt(out["r_err"] ** 2 + out["i_err"] ** 2)
    out["i_z_err"] = np.sqrt(out["i_err"] ** 2 + out["z_mag_err"] ** 2)
    return out


def collect_dp02_sample(
    *,
    token: str,
    tap_sync_url: str,
    limit: int,
    raw_output_path: Path,
    photoz_output_path: Path,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    radius_deg: float | None = None,
) -> dict[str, Any]:
    adql = _build_dp02_adql(
        limit=limit,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        radius_deg=radius_deg,
    )
    df = _run_dp02_tap_query(tap_sync_url=tap_sync_url, adql_query=adql, token=token)
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_output_path, index=False)

    photoz_df = _convert_dp02_to_photoz_frame(df)
    photoz_output_path.parent.mkdir(parents=True, exist_ok=True)
    photoz_df.to_csv(photoz_output_path, index=False)

    return {
        "source": "dp02_tap",
        "tap_sync_url": tap_sync_url,
        "rows_raw": int(len(df)),
        "rows_photoz": int(len(photoz_df)),
        "raw_output_path": str(raw_output_path),
        "photoz_output_path": str(photoz_output_path),
        "adql_query": adql,
    }


def _load_zenodo_record(record_id: int) -> dict[str, Any]:
    payload = _http_get(f"https://zenodo.org/api/records/{record_id}")
    data = json.loads(payload.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Zenodo API returned non-object payload.")
    return data


def _select_zenodo_file(record: dict[str, Any], *, preferred_suffix: str) -> dict[str, Any]:
    files = record.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Zenodo record has no files.")

    preferred_suffix_l = preferred_suffix.lower()
    selected: dict[str, Any] | None = None
    for entry in files:
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        if isinstance(key, str) and key.lower().endswith(preferred_suffix_l):
            selected = entry
            break
    if selected is None:
        first = files[0]
        if not isinstance(first, dict):
            raise ValueError("Invalid first file entry in Zenodo record.")
        selected = first
    return selected


def collect_nnc_catalog(
    *,
    record_id: int,
    preferred_suffix: str,
    output_dir: Path,
    output_filename: str | None = None,
) -> dict[str, Any]:
    record = _load_zenodo_record(record_id)
    file_entry = _select_zenodo_file(record, preferred_suffix=preferred_suffix)

    key = file_entry.get("key")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("Selected Zenodo file is missing `key`.")
    links = file_entry.get("links")
    if not isinstance(links, dict):
        raise ValueError("Selected Zenodo file is missing `links`.")
    download_url = links.get("download") or links.get("self")
    if not isinstance(download_url, str) or not download_url.strip():
        raise ValueError("Selected Zenodo file does not expose download URL.")

    target_name = output_filename or key
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / target_name
    payload = _http_get(download_url)
    target_path.write_bytes(payload)

    metadata = record.get("metadata")
    doi = metadata.get("doi") if isinstance(metadata, dict) else None
    return {
        "source": "nnc_zenodo",
        "record_id": int(record_id),
        "doi": doi,
        "selected_file_key": key,
        "download_url": download_url,
        "output_path": str(target_path),
        "bytes_written": int(len(payload)),
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect DP0.2/NNC real-data assets for photo-z.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dp02 = subparsers.add_parser("dp02", help="Collect Rubin DP0.2 sample via TAP.")
    dp02.add_argument("--token", type=str, default=None, help="Rubin token (or use --token-env).")
    dp02.add_argument(
        "--token-env",
        type=str,
        default="RSP_TOKEN",
        help="Environment variable containing Rubin token.",
    )
    dp02.add_argument("--tap-sync-url", type=str, default=DEFAULT_DP02_TAP_SYNC_URL)
    dp02.add_argument("--limit", type=int, default=250000)
    dp02.add_argument("--ra-deg", type=float, default=None)
    dp02.add_argument("--dec-deg", type=float, default=None)
    dp02.add_argument("--radius-deg", type=float, default=None)
    dp02.add_argument(
        "--raw-output",
        type=Path,
        default=Path("data/dp02/datasets/dp02_photoz_truth_sample.csv"),
    )
    dp02.add_argument(
        "--photoz-output",
        type=Path,
        default=Path("data/sdss/sdss_photoz_real.csv"),
    )
    dp02.add_argument(
        "--report",
        type=Path,
        default=Path("reports/example_summaries/photoz_dp02_collection_latest.json"),
    )

    nnc = subparsers.add_parser("nnc", help="Download NNC paper released catalog from Zenodo.")
    nnc.add_argument("--record-id", type=int, default=DEFAULT_NNC_ZENODO_RECORD)
    nnc.add_argument("--preferred-suffix", type=str, default=".fits")
    nnc.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/nnc_crps/catalogs"),
    )
    nnc.add_argument("--output-filename", type=str, default=None)
    nnc.add_argument(
        "--report",
        type=Path,
        default=Path("reports/example_summaries/photoz_nnc_catalog_collection_latest.json"),
    )

    args = parser.parse_args()
    if args.command == "dp02":
        token = args.token or os.environ.get(args.token_env)
        if not token:
            raise ValueError(
                "Missing Rubin token. Set --token or export the variable from --token-env."
            )
        report = collect_dp02_sample(
            token=token,
            tap_sync_url=args.tap_sync_url,
            limit=args.limit,
            raw_output_path=args.raw_output,
            photoz_output_path=args.photoz_output,
            ra_deg=args.ra_deg,
            dec_deg=args.dec_deg,
            radius_deg=args.radius_deg,
        )
    elif args.command == "nnc":
        report = collect_nnc_catalog(
            record_id=args.record_id,
            preferred_suffix=args.preferred_suffix,
            output_dir=args.output_dir,
            output_filename=args.output_filename,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")

    _write_report(args.report, report)
    print(f"Wrote report: {args.report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
