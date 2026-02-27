from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tools import photoz_collect_real_data as collect


def test_build_dp02_adql_includes_filters_and_limit() -> None:
    query = collect._build_dp02_adql(
        limit=1234,
        ra_deg=55.5,
        dec_deg=-12.25,
        radius_deg=0.75,
    )
    assert "SELECT TOP 1234" in query
    assert "dp02_dc2_catalogs.Object" in query
    assert "dp02_dc2_catalogs.TruthSummary" in query
    assert "CONTAINS(POINT('ICRS'" in query


def test_convert_dp02_to_photoz_frame_shape_and_columns() -> None:
    frame = pd.DataFrame(
        {
            "object_id": [1, 2],
            "flux_u": [100.0, 90.0],
            "flux_g": [110.0, 80.0],
            "flux_r": [120.0, 70.0],
            "flux_i": [130.0, 60.0],
            "flux_z": [140.0, 50.0],
            "spec_z": [0.2, 0.4],
        }
    )
    out = collect._convert_dp02_to_photoz_frame(frame)
    assert len(out) == 2
    expected = {
        "objid",
        "spec_z",
        "spec_z_err",
        "u",
        "g",
        "r",
        "i",
        "z_mag",
        "u_g",
        "g_r",
        "r_i",
        "i_z",
    }
    assert expected <= set(out.columns)


def test_select_zenodo_file_prefers_suffix() -> None:
    record = {
        "files": [
            {"key": "catalog.csv", "links": {"download": "https://example.org/catalog.csv"}},
            {"key": "catalog.fits", "links": {"download": "https://example.org/catalog.fits"}},
        ]
    }
    selected = collect._select_zenodo_file(record, preferred_suffix=".fits")
    assert selected["key"] == "catalog.fits"


def test_collect_nnc_catalog_writes_selected_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = {
        "metadata": {"doi": "10.5281/zenodo.18410731"},
        "files": [
            {
                "key": "nnc_catalog.fits",
                "links": {"download": "https://example.org/nnc_catalog.fits"},
            }
        ],
    }

    monkeypatch.setattr(collect, "_load_zenodo_record", lambda record_id: record)
    monkeypatch.setattr(collect, "_http_get", lambda url, headers=None: b"fits-bytes")

    report = collect.collect_nnc_catalog(
        record_id=18410731,
        preferred_suffix=".fits",
        output_dir=tmp_path,
    )
    out = Path(report["output_path"])
    assert out.exists()
    assert out.read_bytes() == b"fits-bytes"
    assert report["doi"] == "10.5281/zenodo.18410731"
