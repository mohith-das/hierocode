"""Disk-backed plan cache keyed by task + skeleton + model inputs."""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

from hierocode.broker.plan_schema import Plan

DEFAULT_TTL_SECONDS = 86400  # 24 hours
_SCHEMA_VERSION = 1


def cache_key(task: str, skeleton: str, planner_model: str, drafter_model: str) -> str:
    """Deterministic sha256 hex digest of concatenated inputs, prefixed with 'plan-'."""
    joined = "\x00".join([task, skeleton, planner_model, drafter_model])
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return f"plan-{digest}"


def cache_dir() -> Path:
    """Return ~/.cache/hierocode/plans/, creating the directory tree if missing."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    path = base / "hierocode" / "plans"
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_cached_plan(
    key: str,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    cache_root: Optional[Path] = None,
) -> Optional[Plan]:
    """Return cached Plan if it exists, is fresh, and parses cleanly. Else None."""
    path = (cache_root or cache_dir()) / f"{key}.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None
    if data.get("schema_version") != _SCHEMA_VERSION:
        return None
    cached_at = data.get("cached_at", 0.0)
    if time.time() - cached_at >= ttl_seconds:
        return None
    try:
        return Plan.model_validate(data["plan"])
    except Exception:
        return None


def write_cached_plan(
    key: str,
    plan: Plan,
    cache_root: Optional[Path] = None,
) -> None:
    """Write plan to {cache_root or cache_dir()}/{key}.json with a cached_at timestamp."""
    root = cache_root or cache_dir()
    path = root / f"{key}.json"
    tmp = root / f"{key}.json.tmp"
    payload = {
        "cached_at": time.time(),
        "schema_version": _SCHEMA_VERSION,
        "plan": plan.model_dump(mode="json"),
    }
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    os.replace(tmp, path)


def clear_cache(cache_root: Optional[Path] = None) -> int:
    """Delete all cached plans in cache_root (or cache_dir()). Returns count deleted."""
    root = cache_root or cache_dir()
    count = 0
    for p in root.glob("plan-*.json"):
        try:
            p.unlink()
            count += 1
        except Exception:
            pass
    return count
