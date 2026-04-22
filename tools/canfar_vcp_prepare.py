#!/usr/bin/env python3
"""Print or run ``vcp`` commands to stage NeurIPS/CANFAR inputs on VOSpace.

Each destination is a **full VOSpace URI** starting with ``vos:`` (never a bare path).
Local sources are paths under ``--repo``; the remote path mirrors that suffix so
``scripts/canfar_headless_job.sh`` ``VCP_SPECS`` lines (``key|key``) stay aligned.

Examples::

    # Destination must be absolute, e.g. vos:sfabbro/torchregress (not sfabbro/... alone)
    uv run python tools/canfar_vcp_prepare.py --vos-base vos:sfabbro/torchregress
    uv run python tools/canfar_vcp_prepare.py --vos-base vos:sfabbro/torchregress --execute
    uv run python tools/canfar_vcp_prepare.py --write-vcp-specs /tmp/vcp_specs.txt
    uv run python tools/canfar_vcp_prepare.py --with-tabred --execute
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
from pathlib import Path


def _load_run_neurips(repo: Path):
    path = repo / "scripts" / "run_neurips_sage_reg_full.py"
    spec = importlib.util.spec_from_file_location("run_neurips_sage_reg_full", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _require_vos_uri(raw: str) -> str:
    """Require a full VOSpace URI (``vos:authority/path``)."""
    s = raw.strip()
    if not s.startswith("vos:"):
        raise SystemExit(
            "--vos-base must be a full VOS URI starting with 'vos:' "
            "(example: vos:sfabbro/torchregress)."
        )
    return s.rstrip("/")


def _rel_under_repo(repo: Path, p: Path) -> str:
    """Path of *p* under *repo* (for mirroring onto VOS under the same suffix)."""
    try:
        return str(p.resolve().relative_to(repo.resolve()))
    except ValueError as exc:
        raise SystemExit(f"Path outside --repo (cannot build VOS key): {p}") from exc


def _collect_core_assets(mod: object, *, with_higgs: bool) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = [
        ("year", Path(mod.YEAR_CACHE_DEFAULT)),
        ("diamonds", Path(mod.OPENML_DIAMONDS_CACHE_DEFAULT)),
        ("tuning", Path(mod.TUNING_CSV_DEFAULT)),
    ]
    if with_higgs:
        out.append(("higgs", Path(mod.HIGGS_PARQUET_DEFAULT)))
    return out


def _iter_tabred_files(mod: object, repo: Path) -> list[Path]:
    """All files under ``data/tabred`` default dataset dirs (for ``VCP_SPECS`` lines)."""
    root = Path(mod.TABRED_ROOT_DEFAULT).resolve()
    out: list[Path] = []
    for name in mod.TABRED_DEFAULT_DATASETS:
        d = root / name
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            parts = p.parts
            if "__pycache__" in parts or p.name == ".DS_Store":
                continue
            try:
                _ = p.resolve().relative_to(repo.resolve())
            except ValueError:
                continue
            out.append(p)
    return out


def _vcp_file(
    repo: Path,
    src: Path,
    vos_uri: str,
    *,
    execute: bool,
    extra_args: list[str],
) -> None:
    if not src.is_file():
        print(f"[skip] missing file: {src}", flush=True)
        return
    cmd = ["vcp", *extra_args, str(src), vos_uri]
    print(subprocess.list2cmdline(cmd), flush=True)
    if execute:
        subprocess.run(cmd, cwd=str(repo), check=True)


def _vcp_recursive_dir(
    repo: Path,
    src_dir: Path,
    vos_dest_dir_uri: str,
    *,
    execute: bool,
    extra_args: list[str],
) -> None:
    """Recursive directory copy: ``vcp local_dir/ vos:…/dir/`` (CADC ``vcp`` is recursive)."""
    if not src_dir.is_dir():
        print(f"[skip] missing directory: {src_dir}", flush=True)
        return
    if not vos_dest_dir_uri.startswith("vos:"):
        raise SystemExit(
            f"Internal error: VOS destination must start with vos:: {vos_dest_dir_uri!r}"
        )
    src_arg = str(src_dir.resolve())
    if not src_arg.endswith("/"):
        src_arg += "/"
    dst_arg = vos_dest_dir_uri if vos_dest_dir_uri.endswith("/") else vos_dest_dir_uri + "/"
    cmd = ["vcp", *extra_args, src_arg, dst_arg]
    print(subprocess.list2cmdline(cmd), flush=True)
    if execute:
        subprocess.run(cmd, cwd=str(repo), check=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=None, help="Repo root (default: parent of tools/).")
    p.add_argument(
        "--vos-base",
        default="vos:sfabbro/torchregress",
        metavar="VOS_URI",
        help="Full VOSpace URI prefix, must start with vos: (e.g. vos:sfabbro/torchregress).",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Run vcp (default: print commands only).",
    )
    p.add_argument("--with-higgs", action="store_true", help="Include Higgs parquet when present.")
    p.add_argument(
        "--with-tabred",
        action="store_true",
        help="One recursive vcp for the whole data/tabred tree; add per-file VCP_SPECS lines.",
    )
    p.add_argument(
        "--tabred-nstreams",
        type=int,
        default=None,
        metavar="N",
        help="If set, pass --nstreams=N to vcp (tool-dependent).",
    )
    p.add_argument(
        "--write-vcp-specs",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write VCP_SPECS lines (year + tuning + diamonds [+ higgs] [+ tabred files if --with-tabred]).",
    )
    args = p.parse_args()

    repo = (args.repo or Path(__file__).resolve().parents[1]).resolve()
    vos = _require_vos_uri(args.vos_base)
    mod = _load_run_neurips(repo)

    vcp_extras: list[str] = []
    if args.tabred_nstreams is not None:
        vcp_extras.append(f"--nstreams={args.tabred_nstreams}")

    for label, path in _collect_core_assets(mod, with_higgs=args.with_higgs):
        if not path.is_file():
            print(f"[skip] missing {label}: {path}", flush=True)
            continue
        rel = _rel_under_repo(repo, path)
        dest = f"{vos}/{rel}"
        _vcp_file(repo, path, dest, execute=args.execute, extra_args=vcp_extras)

    if args.with_tabred:
        tab_root = Path(mod.TABRED_ROOT_DEFAULT).resolve()
        if tab_root.is_dir():
            rel = _rel_under_repo(repo, tab_root)
            vos_tab = f"{vos}/{rel}/"
            _vcp_recursive_dir(
                repo,
                tab_root,
                vos_tab,
                execute=args.execute,
                extra_args=vcp_extras,
            )

    if args.write_vcp_specs is not None:
        lines: list[str] = []
        for _label, path in _collect_core_assets(mod, with_higgs=args.with_higgs):
            if path.is_file():
                rel = _rel_under_repo(repo, path)
                lines.append(f"{rel}|{rel}")
        if args.with_tabred:
            for path in _iter_tabred_files(mod, repo):
                rel = _rel_under_repo(repo, path)
                lines.append(f"{rel}|{rel}")
        text = "\n".join(lines) + ("\n" if lines else "")
        args.write_vcp_specs.write_text(text, encoding="utf-8")
        print(f"Wrote {args.write_vcp_specs} ({len(lines)} rows)", flush=True)

    if not args.execute and not args.write_vcp_specs:
        print(
            "\n# Dry-run only. Re-run with --execute to upload, "
            "or --write-vcp-specs FILE for headless VCP_SPECS.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
