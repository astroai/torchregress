"""Unit tests for TabReD fetch helper (text patch only; no Kaggle/git in CI)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_fetch_module():
    path = REPO_ROOT / "tools" / "fetch_tabred_data.py"
    spec = importlib.util.spec_from_file_location("_fetch_tabred_data", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fetch_mod():
    return _load_fetch_module()


def test_redirect_tabred_util_data_dir_text_applies_and_idempotent(
    fetch_mod, tmp_path: Path
) -> None:
    line = fetch_mod._DATA_DIR_LINE  # noqa: SLF001
    sample = f"PROJECT_DIR = Path(__file__).parent.parent\n{line}\nEXP_DIR = PROJECT_DIR / 'exp'\n"
    target = tmp_path / "tabred_out"
    out = fetch_mod.redirect_tabred_util_data_dir_text(sample, target)
    assert "torchregress: redirected DATA_DIR" in out
    assert str(target.resolve().as_posix()) in out
    assert line not in out
    out2 = fetch_mod.redirect_tabred_util_data_dir_text(out, target)
    assert out2 == out


def test_redirect_tabred_util_data_dir_text_missing_line_raises(fetch_mod) -> None:
    with pytest.raises(ValueError, match="expected"):
        fetch_mod.redirect_tabred_util_data_dir_text("no data dir here", Path("/x"))


def test_prune_upstream_preprocessing_tmp_removes_tmp(fetch_mod, tmp_path: Path) -> None:
    vendor = tmp_path / ".vendor" / "yandex-tabred"
    tmp = vendor / "preprocessing" / "tmp"
    tmp.mkdir(parents=True)
    (tmp / "scratch.bin").write_bytes(b"x")
    fetch_mod._prune_upstream_preprocessing_tmp(vendor)  # noqa: SLF001
    assert not tmp.exists()
