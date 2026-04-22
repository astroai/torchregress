#!/usr/bin/env python3
"""Mirror ``<repo>/data/`` to CANFAR VOSpace (one recursive ``vcp``).

Defaults (no flags needed if you run from the torchregress clone)::

    cd /path/to/torchregress
    uv run python scripts/canfar_vcp_prepare.py

This will:

1. Remove small local junk under ``data/`` (``__pycache__``, ``.DS_Store``, TabReD
   ``.vendor/.../preprocessing/tmp`` when present — safe upstream scratch).
2. Check VOS auth with ``vls`` (if it fails, refresh cert: ``cadc-get-cert -u USER``).
3. ``vmkdir -p`` the remote ``…/data`` container.
4. Run one recursive ``vcp`` of ``./data/`` → ``<vos-base>/data/``, with
   ``--exclude`` so ``tabred/.vendor/`` (upstream yandex checkout) is **not**
   uploaded — you only need materialized datasets under ``tabred/<task>/``.

``vcp`` is always recursive; re-run this script after an interruption — it will
resume/skip content that is already on VOS (checksum-style behaviour).

Requires ``vcp`` / ``vls`` / ``vmkdir`` on ``PATH`` (``uv sync --extra vos`` or
``uv pip install vos``).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

DEFAULT_VOS_BASE = "vos:sfabbro/torchregress"
# CADC vcp: comma-separated substrings matched against the copy destination path
# (see vos.commands.vcp). Keeps TabReD upstream clone off VOS; still uploads
# data/tabred/cooking-time/, etc.
DEFAULT_VCP_EXCLUDE = "/.vendor/,__pycache__"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def clean_local_data_junk(data_dir: Path) -> list[str]:
    """Remove only disposable caches under *data_dir*. Returns log lines."""
    log: list[str] = []
    if not data_dir.is_dir():
        return log
    for p in sorted(data_dir.rglob("__pycache__"), reverse=True):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            log.append(f"removed {p.relative_to(data_dir)}")
    for p in data_dir.rglob(".DS_Store"):
        if p.is_file():
            p.unlink(missing_ok=True)
            log.append(f"removed {p.relative_to(data_dir)}")
    tmp = data_dir / "tabred" / ".vendor" / "yandex-tabred" / "preprocessing" / "tmp"
    if tmp.is_dir():
        shutil.rmtree(tmp, ignore_errors=True)
        log.append(f"removed {tmp.relative_to(data_dir)} (TabReD upstream scratch)")
    return log


def _require_vos_base(raw: str) -> str:
    s = raw.strip().rstrip("/")
    if not s.startswith("vos:"):
        raise SystemExit("--vos-base must start with vos: (example: vos:sfabbro/torchregress).")
    return s


def _check_vos_cli() -> None:
    for cmd in ("vcp", "vls", "vmkdir"):
        if shutil.which(cmd) is None:
            raise SystemExit(
                f"{cmd!r} not on PATH. Install the VOS tools, e.g. "
                "`uv sync --extra vos` or `uv pip install vos`."
            )


def _probe_vos_listable(vos_base: str) -> None:
    r = subprocess.run(["vls", vos_base], capture_output=True, text=True, timeout=120)
    if r.returncode == 0:
        return
    tail = (r.stderr or r.stdout or "").strip()
    extra = f"\n{tail}\n" if tail else ""
    raise SystemExit(
        f"vls {vos_base!r} failed (exit {r.returncode}). "
        "Your CADC certificate is probably missing or expired.\n"
        "  cadc-get-cert -u YOUR_CADC_USER\n"
        "Then re-run this script."
        f"{extra}"
    )


def _rel_should_skip_for_vos(rel_posix: str) -> bool:
    """True for paths we do not mirror to VOS (upstream TabReD clone, bytecode)."""
    parts = rel_posix.split("/")
    if ".vendor" in parts:
        return True
    if "__pycache__" in parts:
        return True
    return False


def _write_vcp_specs(repo: Path, data_dir: Path, out: Path) -> int:
    """Emit VCP_SPECS lines (path under repo | same) for every file under data_dir."""
    lines: list[str] = []
    data_dir = data_dir.resolve()
    repo = repo.resolve()
    for p in sorted(data_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(repo).as_posix()
        if _rel_should_skip_for_vos(rel):
            continue
        lines.append(f"{rel}|{rel}")
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="Repo root (default: parent of scripts/).",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Local data root (default: <repo>/data).",
    )
    p.add_argument(
        "--vos-base",
        default=DEFAULT_VOS_BASE,
        metavar="URI",
        help=f"Full VOS URI prefix (default: {DEFAULT_VOS_BASE}).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print cleanup + vmkdir + vcp only; do not run vls/vmkdir/vcp.",
    )
    p.add_argument(
        "--no-clean-local",
        action="store_true",
        help="Do not remove __pycache__, .DS_Store, or tabred preprocessing/tmp under data/.",
    )
    p.add_argument(
        "--skip-auth-check",
        action="store_true",
        help="Skip vls probe (for offline --dry-run only).",
    )
    p.add_argument("--quiet", action="store_true", help="Omit -v on vcp.")
    p.add_argument(
        "--include-tabred-vendor",
        action="store_true",
        help="Upload data/tabred/.vendor too (default: exclude; not needed for runs on VOS).",
    )
    p.add_argument(
        "--vcp-exclude",
        default=None,
        metavar="PATTERN",
        help=(
            "Override vcp --exclude (comma-separated substrings; default is "
            f"{DEFAULT_VCP_EXCLUDE!r}). Pass a single space to disable --exclude."
        ),
    )
    p.add_argument(
        "--write-vcp-specs",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write headless VCP_SPECS rows for every file under data/ (then exit).",
    )
    args = p.parse_args()

    repo = (args.repo or _repo_root()).resolve()
    data_dir = (args.data_dir or repo / "data").resolve()
    vos_base = _require_vos_base(args.vos_base)
    vos_data = f"{vos_base}/data"

    if args.write_vcp_specs is not None:
        if not data_dir.is_dir():
            raise SystemExit(f"Missing data directory: {data_dir}")
        n = _write_vcp_specs(repo, data_dir, args.write_vcp_specs)
        print(f"Wrote {n} VCP_SPECS lines to {args.write_vcp_specs}", flush=True)
        return 0

    if not data_dir.is_dir():
        raise SystemExit(
            f"No local data directory: {data_dir}\n"
            "Create it or pass --data-dir. Large inputs belong under data/ "
            "(see docs/canfar_neurips_batch.md)."
        )

    if not args.no_clean_local:
        removed = clean_local_data_junk(data_dir)
        if removed and not args.quiet:
            print("== local cleanup under data/", flush=True)
            for line in removed:
                print(f"   {line}", flush=True)

    if args.dry_run:
        print("== dry-run (no vls / vmkdir / vcp executed)", flush=True)
    else:
        _check_vos_cli()
        if not args.skip_auth_check:
            print(f"== vls {vos_base!r} (auth check)", flush=True)
            _probe_vos_listable(vos_base)

    mkdir_cmd = ["vmkdir", "-p", vos_data]
    src = str(data_dir)
    if not src.endswith("/"):
        src += "/"
    dst = vos_data if vos_data.endswith("/") else vos_data + "/"
    vcp_cmd = ["vcp"]
    if not args.quiet:
        vcp_cmd.append("-v")
    exclude: str | None
    if args.vcp_exclude is not None:
        raw = args.vcp_exclude
        exclude = None if raw.strip() == "" or raw.strip() == " " else raw.strip()
    elif args.include_tabred_vendor:
        exclude = "__pycache__"
    else:
        exclude = DEFAULT_VCP_EXCLUDE
    if exclude:
        vcp_cmd.extend(["--exclude", exclude])
    vcp_cmd.extend([src, dst])

    print("== " + subprocess.list2cmdline(mkdir_cmd), flush=True)
    print("== " + subprocess.list2cmdline(vcp_cmd), flush=True)

    if args.dry_run:
        print("\n# Re-run without --dry-run to upload.", flush=True)
        return 0

    subprocess.run(mkdir_cmd, cwd=str(repo), check=True)
    subprocess.run(vcp_cmd, cwd=str(repo), check=True)
    print("== done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
