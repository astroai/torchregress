"""Shared helpers for CANFAR Skaha headless launch scripts."""

from __future__ import annotations

import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Callable


def arc_run_root_from_args(
    *,
    arc_project_root: str,
    arc_runs_subdir: str,
    run_id: str,
    arc_run_root_override: str,
) -> str:
    if arc_run_root_override.strip():
        return arc_run_root_override.rstrip("/")
    return f"{arc_project_root.rstrip('/')}/" f"{arc_runs_subdir.strip('/')}/{run_id}"


def headless_bash_invocation(*, torchregress_repo: str) -> tuple[str, str]:
    """Return (cmd, args) for Skaha: bash -lc <quoted inner>."""
    script_path = f"{torchregress_repo.rstrip('/')}/scripts/canfar_headless_job.sh"
    inner = f"exec bash {shlex.quote(script_path)}"
    return "/bin/bash", "-lc " + shlex.quote(inner)


def submit_headless_session(
    *,
    session_factory: Callable[[], Any],
    name: str,
    image: str,
    cores: int,
    ram: int,
    env: dict[str, str],
    torchregress_repo: str,
    dry_run: bool,
) -> dict[str, Any]:
    cmd, args_str = headless_bash_invocation(torchregress_repo=torchregress_repo)
    record: dict[str, Any] = {
        "name": name,
        "env": env,
        "cmd": cmd,
        "args": args_str,
        "cores": cores,
        "ram": ram,
        "image": image,
    }
    if dry_run:
        record["session_id"] = None
        record["status"] = "dry_run"
        return record

    session = session_factory()
    ids = session.create(
        name=name,
        image=image,
        cores=cores,
        ram=ram,
        kind="headless",
        cmd=cmd,
        args=args_str,
        env=env,
        replicas=1,
    )
    sid = ids[0] if ids else None
    record["session_id"] = sid
    record["status"] = "submitted" if sid else "create_failed"
    return record


_TERMINAL_LOWER = frozenset({"succeeded", "failed", "error", "terminated", "completed", "complete"})


def wait_for_session_ids(
    *,
    session_factory: Callable[[], Any],
    session_ids: list[str],
    poll_s: float = 20.0,
    timeout_s: float | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, str]:
    """Poll ``Session.info`` until all ids reach a terminal status (or timeout).

    Returns mapping id -> final status string (or ``wait_timeout``).
    """
    log = log or (lambda m: print(m, flush=True))
    if not session_ids:
        return {}
    session = session_factory()
    start = time.monotonic()
    pending = set(session_ids)
    out: dict[str, str] = {}

    while pending:
        if timeout_s is not None and (time.monotonic() - start) > timeout_s:
            for sid in pending:
                out[sid] = "wait_timeout"
            break
        for sid in list(pending):
            try:
                rows = session.info(ids=sid)
            except Exception as exc:  # noqa: BLE001
                log(f"[canfar_common] info({sid!r}) error: {exc!r}")
                continue
            if not rows:
                log(f"[canfar_common] info({sid!r}) empty response")
                continue
            st = str(rows[0].get("status", ""))
            if st.strip().lower() in _TERMINAL_LOWER:
                out[sid] = st
                pending.discard(sid)
                log(f"[canfar_common] session {sid} -> {st}")
        if pending:
            time.sleep(poll_s)
    return out


def load_yaml_or_json(path: Path) -> dict[str, Any]:
    """Load a work plan dict from ``.yaml`` / ``.yml`` or ``.json``."""
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise SystemExit(
                "YAML work plans require PyYAML. Install with:\n"
                "  uv pip install 'torchregress[canfar]'\n"
                "or use a .json plan file."
            ) from exc
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise SystemExit(f"Unsupported plan suffix {path.suffix!r} (use .yaml or .json)")
    if not isinstance(data, dict):
        raise SystemExit(f"Work plan root must be a mapping, got {type(data).__name__}")
    return data


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[canfar_common] wrote {path}", flush=True)


def ensure_canfar_import() -> None:
    try:
        import importlib

        importlib.import_module("canfar.sessions")
    except ImportError:
        print(
            "Missing dependency: install with  uv pip install 'torchregress[canfar]'  "
            "or  pip install 'canfar>=1.3'",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
