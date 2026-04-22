"""Tests for scripts/canfar_vcp_prepare.py (loaded by path; not an installed package)."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path


def _load_script():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "canfar_vcp_prepare.py"
    spec = importlib.util.spec_from_file_location("canfar_vcp_prepare", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_clean_local_data_junk() -> None:
    mod = _load_script()
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "a" / "__pycache__").mkdir(parents=True)
        (d / "a" / "__pycache__" / "x.pyc").write_text("1", encoding="utf-8")
        (d / ".DS_Store").write_text("", encoding="utf-8")
        log = mod.clean_local_data_junk(d)
        assert not (d / "a" / "__pycache__").exists()
        assert not (d / ".DS_Store").exists()
        assert len(log) >= 2


def test_require_vos_base() -> None:
    mod = _load_script()
    assert mod._require_vos_base("vos:x/y") == "vos:x/y"
    try:
        mod._require_vos_base("x/y")
    except SystemExit:
        return
    raise AssertionError("expected SystemExit")


def test_write_vcp_specs(tmp_path: Path) -> None:
    mod = _load_script()
    repo = tmp_path / "repo"
    data = repo / "data" / "paper"
    data.mkdir(parents=True)
    (data / "f.csv").write_text("a", encoding="utf-8")
    out = tmp_path / "spec.txt"
    n = mod._write_vcp_specs(repo, repo / "data", out)
    assert n == 1
    assert "data/paper/f.csv|data/paper/f.csv" in out.read_text(encoding="utf-8")


def test_write_vcp_specs_skips_tabred_vendor(tmp_path: Path) -> None:
    mod = _load_script()
    repo = tmp_path / "repo"
    (repo / "data" / "tabred" / "cooking-time").mkdir(parents=True)
    (repo / "data" / "tabred" / "cooking-time" / "info.json").write_text("{}", encoding="utf-8")
    v = repo / "data" / "tabred" / ".vendor" / "yandex-tabred" / "README"
    v.parent.mkdir(parents=True)
    v.write_text("x", encoding="utf-8")
    out = tmp_path / "spec.txt"
    n = mod._write_vcp_specs(repo, repo / "data", out)
    assert n == 1
    text = out.read_text(encoding="utf-8")
    assert "cooking-time/info.json" in text
    assert ".vendor" not in text
