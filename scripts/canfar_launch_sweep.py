#!/usr/bin/env python3
"""Submit CANFAR Skaha headless jobs that run one shard of sage_year_label_fraction_sweep each.

Uses the CANFAR Python API only (no ``canfar`` CLI subprocess). See docs/canfar_batch_sweeps.md.

Prerequisites (local laptop):
  uv pip install 'torchregress[canfar]'   # or: pip install canfar
  canfar auth login

Example:
  python scripts/canfar_launch_sweep.py --run-id y2026_0422_a
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import canfar_common as cf  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-id",
        required=True,
        help="Unique run tag (used in session names and ARC_RUN_ROOT).",
    )
    p.add_argument(
        "--arc-project-root",
        default="/arc/projects/ots/torchregress",
        help="ARC project directory containing runs/ (headless job receives ARC_RUN_ROOT).",
    )
    p.add_argument(
        "--arc-runs-subdir",
        default="runs",
        help="Subdirectory under arc-project-root for this run's outputs.",
    )
    p.add_argument(
        "--arc-run-root",
        default="",
        help="If set, overrides arc-project-root/arc-runs-subdir/run-id for ARC_RUN_ROOT env.",
    )
    p.add_argument(
        "--vos-base",
        default="vos:sfabbro/torchregress",
        help="VOSpace prefix for staging inputs.",
    )
    p.add_argument(
        "--torchregress-repo",
        default="/arc/home/sfabbro/src/torchregress",
        help="ARC path to torchregress clone (headless script path).",
    )
    p.add_argument(
        "--image",
        default="images.canfar.net/skaha/astroml:latest",
        help="Skaha container image.",
    )
    p.add_argument(
        "--shards",
        type=int,
        default=40,
        help="Number of headless sessions (shard_count).",
    )
    p.add_argument("--cores", type=int, default=8, help="CPU cores per session (fixed mode).")
    p.add_argument("--ram", type=int, default=32, help="RAM per session in GB (fixed mode).")
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=40,
        metavar="N",
        help="Max simultaneous session.create calls (throttle for quotas).",
    )
    p.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Write JSON manifest here (default: ./canfar_sweep_<run-id>_manifest.json).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print would-be sessions without calling the CANFAR API.",
    )
    return p.parse_args()


def _arc_run_root(args: argparse.Namespace) -> str:
    return cf.arc_run_root_from_args(
        arc_project_root=args.arc_project_root,
        arc_runs_subdir=args.arc_runs_subdir,
        run_id=args.run_id,
        arc_run_root_override=args.arc_run_root,
    )


def _submit_one(
    *,
    session_factory,
    run_id: str,
    shard_id: int,
    shards: int,
    image: str,
    cores: int,
    ram: int,
    arc_run_root: str,
    vos_base: str,
    torchregress_repo: str,
    dry_run: bool,
) -> dict[str, Any]:
    name = f"torchregress-{run_id}-s{shard_id:02d}"
    env = {
        "ARC_RUN_ROOT": arc_run_root,
        "VOS_BASE": vos_base,
        "SHARD_ID": str(shard_id),
        "SHARD_COUNT": str(shards),
        "TORCHREGRESS_REPO": torchregress_repo,
        "RUN_ID": run_id,
        "CANFAR_JOB_KIND": "year_label_shard",
    }
    rec = cf.submit_headless_session(
        session_factory=session_factory,
        name=name,
        image=image,
        cores=cores,
        ram=ram,
        env=env,
        torchregress_repo=torchregress_repo,
        dry_run=dry_run,
    )
    rec["shard_id"] = shard_id
    return rec


def main() -> int:
    args = _parse_args()
    if not args.dry_run:
        cf.ensure_canfar_import()

    arc_run_root = _arc_run_root(args)
    manifest_path = args.manifest_out or Path(f"canfar_sweep_{args.run_id}_manifest.json")

    def session_factory() -> Any:
        from canfar.sessions import Session  # noqa: PLC0415

        return Session()

    entries: list[dict[str, Any]] = []
    shard_ids = list(range(args.shards))

    if args.max_concurrent < 1:
        print("--max-concurrent must be >= 1", file=sys.stderr)
        return 1

    print(
        f"[canfar_launch_sweep] run_id={args.run_id!r} shards={args.shards} "
        f"arc_run_root={arc_run_root} max_concurrent={args.max_concurrent} dry_run={args.dry_run}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
        futures = {
            pool.submit(
                _submit_one,
                session_factory=session_factory,
                run_id=args.run_id,
                shard_id=sid,
                shards=args.shards,
                image=args.image,
                cores=args.cores,
                ram=args.ram,
                arc_run_root=arc_run_root,
                vos_base=args.vos_base,
                torchregress_repo=args.torchregress_repo,
                dry_run=args.dry_run,
            ): sid
            for sid in shard_ids
        }
        for fut in as_completed(futures):
            sid = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001
                rec = {"shard_id": sid, "status": "error", "error": repr(exc)}
            entries.append(rec)
            shard = rec.get("shard_id", sid)
            sess_id = rec.get("session_id")
            print(f"[canfar_launch_sweep] shard {shard} -> {sess_id}", flush=True)

    entries.sort(key=lambda r: r.get("shard_id", -1))
    manifest = {
        "run_id": args.run_id,
        "arc_run_root": arc_run_root,
        "vos_base": args.vos_base,
        "torchregress_repo": args.torchregress_repo,
        "image": args.image,
        "shards": args.shards,
        "cores": args.cores,
        "ram_gb": args.ram,
        "sessions": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[canfar_launch_sweep] wrote {manifest_path}", flush=True)

    failed = [e for e in entries if not e.get("session_id") and not args.dry_run]
    if failed:
        msg = f"[canfar_launch_sweep] WARNING: {len(failed)} submissions failed or no id"
        print(msg, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
