from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError

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


def test_collect_nnc_catalog_falls_back_from_stale_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    fallback_record = {
        "metadata": {"doi": "10.5281/zenodo.5528827"},
        "files": [
            {
                "key": "HSC_v6.csv",
                "links": {"download": "https://example.org/HSC_v6.csv"},
            }
        ],
    }

    def _fake_load(record_id: int):
        calls.append(record_id)
        if record_id == collect.DEFAULT_NNC_ZENODO_RECORD:
            raise HTTPError(
                url=f"https://zenodo.org/api/records/{record_id}",
                code=404,
                msg="Not Found",
                hdrs=None,  # type: ignore[arg-type]
                fp=None,
            )
        if record_id == collect.DEFAULT_NNC_FALLBACK_RECORD:
            return fallback_record
        raise AssertionError(record_id)

    monkeypatch.setattr(collect, "_load_zenodo_record", _fake_load)
    monkeypatch.setattr(collect, "_http_get", lambda url, headers=None: b"csv-bytes")

    report = collect.collect_nnc_catalog(
        record_id=collect.DEFAULT_NNC_ZENODO_RECORD,
        preferred_suffix=".csv",
        output_dir=tmp_path,
    )

    assert calls == [collect.DEFAULT_NNC_ZENODO_RECORD, collect.DEFAULT_NNC_FALLBACK_RECORD]
    assert report["source"] == "nnc_zenodo_fallback"
    assert report["record_id"] == collect.DEFAULT_NNC_FALLBACK_RECORD
    assert report["requested_record_id"] == collect.DEFAULT_NNC_ZENODO_RECORD


def test_collect_transferz_splits_downloads_and_normalizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = {
        "metadata": {"doi": "10.5281/zenodo.16541823"},
        "files": [
            {
                "key": key,
                "links": {"self": f"https://example.org/{key}"},
                "checksum": "md5:stub",
            }
            for key in collect.TRANSFERZ_SPLIT_KEYS.values()
        ],
    }
    csv_payload = (
        "hsc_id,g_cmodel_mag,r_cmodel_mag,i_cmodel_mag,z_cmodel_mag,y_cmodel_mag,"
        "g_cmodel_magsigma,r_cmodel_magsigma,i_cmodel_magsigma,z_cmodel_magsigma,"
        "y_cmodel_magsigma,z\n"
        "1,22.1,21.8,21.5,21.2,21.0,0.05,0.04,0.04,0.05,0.06,0.42\n"
        "2,23.1,22.6,22.1,21.7,21.4,0.06,0.05,0.05,0.06,0.07,0.91\n"
    ).encode("utf-8")

    monkeypatch.setattr(collect, "_load_zenodo_record", lambda record_id: record)
    monkeypatch.setattr(collect, "_http_get", lambda url, headers=None: csv_payload)

    report = collect.collect_transferz_splits(
        record_id=collect.DEFAULT_TRANSFERZ_ZENODO_RECORD,
        raw_output_dir=tmp_path / "raw",
        normalized_output_dir=tmp_path / "normalized",
        default_target_err=0.02,
    )

    assert report["artifact"] == "photoz_transferz_collection_report"
    assert report["record_id"] == collect.DEFAULT_TRANSFERZ_ZENODO_RECORD
    for split in ("train", "cal", "test", "conformal"):
        normalized_path = Path(report["normalized_paths"][split])
        assert normalized_path.exists()
        frame = pd.read_csv(normalized_path)
        assert {"spec_z", "spec_z_err", "g_r", "r_i", "i_z", "z_y"} <= set(frame.columns)
        assert frame["spec_z_err"].iloc[0] == pytest.approx(0.02)
