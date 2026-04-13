"""Clone TabReD upstream, patch ``DATA_DIR``, run Kaggle-backed preprocessing into ``data/tabred/``.

This mirrors `yandex-research/tabred` preprocessing (see their ``preprocessing/README.md``).
Requires:

- ``git`` on ``PATH``
- Kaggle API credentials at ``~/.kaggle/kaggle.json`` (and Kaggle account + dataset access)
- Extra Python packages (not torchregress core deps): ``polars``, ``kaggle``, ``loguru``, ``scikit-learn``

Example::

    uv pip install -e ".[tabred]"   # polars, pyarrow, kaggle, loguru (+ sklearn in most envs)
    uv run python tools/fetch_tabred_data.py --out-dir data/tabred --skip-if-present

Preprocessed tensors are written next to ``--out-dir`` (e.g. ``data/tabred/cooking-time/``).
The upstream repo is cloned to ``<out-dir>/.vendor/yandex-tabred`` (gitignored via ``data/``).
Preprocessing subprocesses set ``PYTHONPATH`` to that repo root so ``import lib`` resolves (upstream
scripts are started from paths under ``preprocessing/``).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

TABRED_REPO_URL = "https://github.com/yandex-research/tabred.git"
VENDOR_SUBDIR = Path(".vendor") / "yandex-tabred"

# TabReD lib/util.py (stable line; re-check if upstream refactors)
_DATA_DIR_LINE = "DATA_DIR = PROJECT_DIR / 'data'"
_REDIRECT_MARKER = "# torchregress: redirected DATA_DIR\n"


def redirect_tabred_util_data_dir_text(util_text: str, data_dir: Path) -> str:
    """Return ``lib/util.py`` contents with ``DATA_DIR`` pointing at ``data_dir`` (idempotent)."""
    if _REDIRECT_MARKER in util_text:
        return util_text
    if _DATA_DIR_LINE not in util_text:
        raise ValueError(f"expected `{_DATA_DIR_LINE}` in tabred lib/util.py")
    resolved = data_dir.resolve()
    replacement = f"{_REDIRECT_MARKER}DATA_DIR = Path(r'{resolved.as_posix()}')\n"
    return util_text.replace(_DATA_DIR_LINE, replacement, 1)


_DATASET_SCRIPTS: dict[str, str] = {
    "cooking-time": "cooking-time.py",
    "delivery-eta": "delivery-eta.py",
    "maps-routing": "maps-routing.py",
}


def _check_imports() -> None:
    missing: list[str] = []
    for mod in ("polars", "kaggle", "loguru", "sklearn"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        need = " ".join(m if m != "sklearn" else "scikit-learn" for m in missing)
        raise SystemExit(
            "Missing dependencies for TabReD preprocessing.\n"
            f"  pip install {need}\n"
            "Example: uv pip install polars kaggle loguru scikit-learn"
        )


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _ensure_vendor_clone(vendor: Path, *, force: bool) -> None:
    if vendor.is_dir() and any(vendor.iterdir()):
        if not force:
            return
        shutil.rmtree(vendor)
    vendor.parent.mkdir(parents=True, exist_ok=True)
    _run(
        ["git", "clone", "--depth", "1", TABRED_REPO_URL, str(vendor)],
        cwd=vendor.parent,
    )


def _patch_lib_data_dir(vendor: Path, data_dir: Path) -> None:
    util_py = vendor / "lib" / "util.py"
    if not util_py.is_file():
        raise FileNotFoundError(f"Expected {util_py} after clone")
    text = util_py.read_text(encoding="utf-8")
    new_text = redirect_tabred_util_data_dir_text(text, data_dir)
    if new_text != text:
        util_py.write_text(new_text, encoding="utf-8")
        print(f"Patched DATA_DIR -> {data_dir.resolve()}", flush=True)


def _env_with_tabred_repo_root(vendor: Path) -> dict[str, str]:
    """TabReD scripts live under ``preprocessing/`` but ``import lib`` needs the repo root on path."""
    env = os.environ.copy()
    root = str(vendor.resolve())
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{root}{os.pathsep}{prev}" if prev else root
    return env


def _run_preprocessing(vendor: Path, script_name: str) -> None:
    script = vendor / "preprocessing" / script_name
    if not script.is_file():
        raise FileNotFoundError(f"Missing preprocessing script {script}")
    cmd = [sys.executable, str(script)]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(
        cmd,
        cwd=str(vendor),
        env=_env_with_tabred_repo_root(vendor),
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "tabred",
        help="Where TabReD datasets are written (cooking-time/, delivery-eta/, …).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(_DATASET_SCRIPTS.keys()),
        choices=sorted(_DATASET_SCRIPTS.keys()),
        help="Which TabReD regression tasks to materialize.",
    )
    parser.add_argument(
        "--skip-if-present",
        action="store_true",
        help="Skip a dataset if <out-dir>/<name>/info.json already exists.",
    )
    parser.add_argument(
        "--force-vendor-clone",
        action="store_true",
        help="Delete and re-clone the upstream tabred checkout under .vendor/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Clone + patch only; do not run preprocessing (no Kaggle download).",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    vendor = (out_dir / VENDOR_SUBDIR).resolve()

    _check_imports()
    _ensure_vendor_clone(vendor, force=bool(args.force_vendor_clone))
    _patch_lib_data_dir(vendor, out_dir)

    if args.dry_run:
        print("Dry run: skipping preprocessing scripts.", flush=True)
        return

    for name in args.datasets:
        ds_dir = out_dir / name
        info = ds_dir / "info.json"
        if args.skip_if_present and info.is_file():
            print(f"Skip (present): {name} -> {info}", flush=True)
            continue
        script = _DATASET_SCRIPTS[name]
        print(f"== Preprocessing {name} ({script})", flush=True)
        try:
            _run_preprocessing(vendor, script)
        except subprocess.CalledProcessError as e:
            raise SystemExit(
                f"Preprocessing failed for {name} (exit {e.returncode}).\n"
                "Check Kaggle credentials (~/.kaggle/kaggle.json), dataset access, and disk/RAM.\n"
                "Upstream docs: https://github.com/yandex-research/tabred/tree/main/preprocessing"
            ) from e


if __name__ == "__main__":
    main()
