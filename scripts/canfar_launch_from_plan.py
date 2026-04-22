#!/usr/bin/env python3
"""Submit CANFAR Skaha headless jobs from a YAML/JSON work plan (waves + sharding).

Reads a manifest describing jobs (NeurIPS phase slices, label sweep shards, overnight
extras, aggregate). Submits ``Session.create`` calls wave-by-wave; optionally waits
for each wave to reach a terminal status before starting the next (recommended so
later phases do not race unfinished writers).

See docs/canfar_neurips_batch.md and scripts/canfar/canfar_work_plan.example.yaml.

Example::

    python scripts/canfar_launch_from_plan.py \\
      --plan scripts/canfar/canfar_work_plan.example.yaml \\
      --run-id neurips_2026_0422_a
"""

from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Run as ``python scripts/canfar_launch_from_plan.py`` → import sibling helpers.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import canfar_common as cf  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="Work plan path (.yaml / .yml / .json).",
    )
    p.add_argument(
        "--run-id",
        required=True,
        help="Unique run tag (session name prefix and default ARC_RUN_ROOT leaf).",
    )
    p.add_argument(
        "--arc-project-root",
        default="/arc/projects/ots/torchregress",
        help="ARC project directory containing runs/.",
    )
    p.add_argument(
        "--arc-runs-subdir",
        default="runs",
        help="Subdirectory under arc-project-root for this run's outputs.",
    )
    p.add_argument(
        "--arc-run-root",
        default="",
        help="If set, overrides arc-project-root/arc-runs-subdir/run-id for ARC_RUN_ROOT.",
    )
    p.add_argument(
        "--vos-base",
        default="vos:sfabbro/torchregress",
        help="VOSpace prefix for staging inputs (passed to headless jobs).",
    )
    p.add_argument(
        "--torchregress-repo",
        default="/arc/home/sfabbro/src/torchregress",
        help="ARC path to torchregress clone.",
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=32,
        metavar="N",
        help="Max parallel Session.create calls within a single wave.",
    )
    p.add_argument(
        "--wait-between-waves",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After each wave, poll session.info until jobs finish (default: on).",
    )
    p.add_argument(
        "--wait-poll-seconds",
        type=float,
        default=25.0,
        help="Polling interval when waiting for a wave to complete.",
    )
    p.add_argument(
        "--wait-timeout-seconds",
        type=float,
        default=0.0,
        help="Max seconds to wait per wave (0 = no timeout).",
    )
    p.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Write JSON manifest (default: ./canfar_plan_<run-id>_manifest.json).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved jobs without calling the CANFAR API.",
    )
    return p.parse_args()


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", name.strip()).strip("-")
    return s[:60] if len(s) > 60 else s


def _expand_shard_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for j in jobs:
        sc = int(j.get("shard_count") or 0)
        if sc <= 0:
            out.append(dict(j))
            continue
        base_id = str(j["id"])
        for k in range(sc):
            entry = dict(j)
            entry["id"] = f"{base_id}__s{k:02d}"
            entry["shard_parent_id"] = base_id
            entry["_shard_id"] = k
            entry["_shard_count"] = sc
            del entry["shard_count"]
            out.append(entry)
    return out


def _assign_waves_if_missing(jobs: list[dict[str, Any]]) -> None:
    """Fill missing ``wave`` via longest-path layering from ``needs`` (job ids)."""
    if all("wave" in j and j["wave"] is not None for j in jobs):
        return
    by_id = {str(j["id"]): j for j in jobs}
    for j in jobs:
        for dep in j.get("needs") or []:
            if str(dep) not in by_id:
                raise SystemExit(f"Job {j['id']!r} needs unknown id {dep!r}")

    # Kahn-style relaxation until stable.
    waves: dict[str, int] = {str(j["id"]): 0 for j in jobs}
    changed = True
    while changed:
        changed = False
        for j in jobs:
            jid = str(j["id"])
            base = 0
            for dep in j.get("needs") or []:
                base = max(base, waves[str(dep)] + 1)
            if waves[jid] != base:
                waves[jid] = base
                changed = True
    for j in jobs:
        if j.get("wave") is None:
            j["wave"] = int(waves[str(j["id"])])


def _merge_job_env(
    job: dict[str, Any],
    *,
    run_id: str,
    arc_run_root: str,
    vos_base: str,
    torchregress_repo: str,
) -> dict[str, str]:
    env: dict[str, str] = {
        "ARC_RUN_ROOT": arc_run_root,
        "VOS_BASE": vos_base,
        "TORCHREGRESS_REPO": torchregress_repo,
        "RUN_ID": run_id,
    }
    if "_shard_id" in job:
        env["SHARD_ID"] = str(job["_shard_id"])
        env["SHARD_COUNT"] = str(job["_shard_count"])
    only = job.get("only_phases")
    if only:
        env.setdefault("NEURIPS_ONLY_PHASES", str(only).replace(" ", ""))
    kind = job.get("job_kind") or job.get("kind")
    if kind:
        env.setdefault("CANFAR_JOB_KIND", str(kind))
    for k, v in (job.get("env") or {}).items():
        env[str(k)] = str(v)
    return env


def _submit_wave(
    *,
    wave_idx: int,
    wave_jobs: list[dict[str, Any]],
    args: argparse.Namespace,
    arc_run_root: str,
    session_factory,
    max_concurrent: int,
    dry_run: bool,
) -> list[dict[str, Any]]:
    torchregress_repo = args.torchregress_repo
    records: list[dict[str, Any]] = []

    def submit_one(job: dict[str, Any]) -> dict[str, Any]:
        jid = str(job["id"])
        name = f"torchregress-{args.run_id}-{_slug(jid)}"
        defaults = job.get("_defaults") or {}
        default_image = "images.canfar.net/skaha/astroml:latest"
        image = str(job.get("image") or defaults.get("image", default_image))
        cores = int(job.get("cores") or defaults.get("cores", 8))
        ram = int(job.get("ram_gb") or job.get("ram") or defaults.get("ram_gb", 32))
        env = _merge_job_env(
            job,
            run_id=args.run_id,
            arc_run_root=arc_run_root,
            vos_base=args.vos_base,
            torchregress_repo=torchregress_repo,
        )
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
        rec["plan_job_id"] = jid
        rec["wave"] = wave_idx
        return rec

    if max_concurrent < 1:
        raise SystemExit("--max-concurrent must be >= 1")

    print(
        f"[canfar_launch_from_plan] wave={wave_idx} jobs={len(wave_jobs)} "
        f"max_concurrent={max_concurrent} dry_run={dry_run}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = {pool.submit(submit_one, j): j for j in wave_jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001
                rec = {
                    "plan_job_id": job["id"],
                    "wave": wave_idx,
                    "status": "error",
                    "error": repr(exc),
                }
            records.append(rec)
            print(
                f"[canfar_launch_from_plan] {job['id']} -> {rec.get('session_id')}",
                flush=True,
            )
    records.sort(key=lambda r: str(r.get("plan_job_id", "")))
    return records


def main() -> int:
    args = _parse_args()
    if not args.plan.is_file():
        print(f"Plan not found: {args.plan}", file=sys.stderr)
        return 1
    if not args.dry_run:
        cf.ensure_canfar_import()

    raw = cf.load_yaml_or_json(args.plan)
    version = int(raw.get("version", 1))
    if version != 1:
        print(f"Unsupported plan version: {version}", file=sys.stderr)
        return 1

    defaults = dict(raw.get("defaults") or {})
    jobs_raw = raw.get("jobs")
    if not isinstance(jobs_raw, list) or not jobs_raw:
        raise SystemExit("Plan must contain a non-empty 'jobs' list")

    jobs = _expand_shard_jobs([dict(x) for x in jobs_raw])
    for j in jobs:
        j["_defaults"] = defaults
    _assign_waves_if_missing(jobs)
    for j in jobs:
        j["wave"] = int(j["wave"])

    by_wave: dict[int, list[dict[str, Any]]] = {}
    for j in jobs:
        w = int(j["wave"])
        by_wave.setdefault(w, []).append(j)
    wave_keys = sorted(by_wave)

    arc_run_root = cf.arc_run_root_from_args(
        arc_project_root=args.arc_project_root,
        arc_runs_subdir=args.arc_runs_subdir,
        run_id=args.run_id,
        arc_run_root_override=args.arc_run_root,
    )
    manifest_path = args.manifest_out or Path(f"canfar_plan_{args.run_id}_manifest.json")

    def session_factory() -> Any:
        from canfar.sessions import Session  # noqa: PLC0415

        return Session()

    all_records: list[dict[str, Any]] = []
    for w in wave_keys:
        wave_jobs = by_wave[w]
        recs = _submit_wave(
            wave_idx=w,
            wave_jobs=wave_jobs,
            args=args,
            arc_run_root=arc_run_root,
            session_factory=session_factory,
            max_concurrent=args.max_concurrent,
            dry_run=args.dry_run,
        )
        all_records.extend(recs)

        sids = [str(r["session_id"]) for r in recs if r.get("session_id")]
        if args.wait_between_waves and sids and not args.dry_run:
            timeout = args.wait_timeout_seconds if args.wait_timeout_seconds > 0 else None
            cf.wait_for_session_ids(
                session_factory=session_factory,
                session_ids=sids,
                poll_s=args.wait_poll_seconds,
                timeout_s=timeout,
            )

    manifest = {
        "artifact": "canfar_work_plan_manifest",
        "version": 1,
        "run_id": args.run_id,
        "plan_path": str(args.plan.resolve()),
        "arc_run_root": arc_run_root,
        "vos_base": args.vos_base,
        "torchregress_repo": args.torchregress_repo,
        "wait_between_waves": args.wait_between_waves,
        "waves": wave_keys,
        "sessions": all_records,
    }
    cf.write_manifest(manifest_path, manifest)

    failed = [e for e in all_records if not e.get("session_id") and not args.dry_run]
    if failed:
        print(
            f"[canfar_launch_from_plan] WARNING: {len(failed)} submissions failed or missing id",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
