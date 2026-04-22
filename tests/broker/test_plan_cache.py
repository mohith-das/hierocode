"""Tests for hierocode.broker.plan_cache."""

from __future__ import annotations

import json
import time
from pathlib import Path


from hierocode.broker.plan_cache import (
    DEFAULT_TTL_SECONDS,
    cache_dir,
    cache_key,
    clear_cache,
    read_cached_plan,
    write_cached_plan,
)
from hierocode.broker.plan_schema import Plan, TaskUnit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plan(task: str = "add logging", unit_id: str = "u1") -> Plan:
    """Return a minimal valid Plan."""
    return Plan(
        task=task,
        units=[
            TaskUnit(
                id=unit_id,
                goal="implement feature",
                target_files=["src/foo.py"],
            )
        ],
    )


_TASK = "add logging"
_SKELETON = "src/\n  foo.py (1 KB)"
_PLANNER = "claude-3-opus"
_DRAFTER = "llama3:8b"


# ---------------------------------------------------------------------------
# cache_key
# ---------------------------------------------------------------------------

class TestCacheKey:
    def test_cache_key_deterministic(self):
        """Same inputs produce the same key on repeated calls."""
        k1 = cache_key(_TASK, _SKELETON, _PLANNER, _DRAFTER)
        k2 = cache_key(_TASK, _SKELETON, _PLANNER, _DRAFTER)
        assert k1 == k2

    def test_cache_key_sensitive_to_task(self):
        """Different task produces a different key."""
        k1 = cache_key(_TASK, _SKELETON, _PLANNER, _DRAFTER)
        k2 = cache_key("other task", _SKELETON, _PLANNER, _DRAFTER)
        assert k1 != k2

    def test_cache_key_sensitive_to_skeleton(self):
        """Different skeleton produces a different key."""
        k1 = cache_key(_TASK, _SKELETON, _PLANNER, _DRAFTER)
        k2 = cache_key(_TASK, "src/\n  bar.py (2 KB)", _PLANNER, _DRAFTER)
        assert k1 != k2

    def test_cache_key_sensitive_to_planner_model(self):
        """Different planner_model produces a different key."""
        k1 = cache_key(_TASK, _SKELETON, _PLANNER, _DRAFTER)
        k2 = cache_key(_TASK, _SKELETON, "gpt-4o", _DRAFTER)
        assert k1 != k2

    def test_cache_key_sensitive_to_drafter_model(self):
        """Different drafter_model produces a different key."""
        k1 = cache_key(_TASK, _SKELETON, _PLANNER, _DRAFTER)
        k2 = cache_key(_TASK, _SKELETON, _PLANNER, "mistral:7b")
        assert k1 != k2

    def test_cache_key_format(self):
        """Key starts with 'plan-' followed by 64 lowercase hex characters."""
        k = cache_key(_TASK, _SKELETON, _PLANNER, _DRAFTER)
        assert k.startswith("plan-")
        hex_part = k[len("plan-"):]
        assert len(hex_part) == 64
        assert all(c in "0123456789abcdef" for c in hex_part)


# ---------------------------------------------------------------------------
# write / read roundtrip
# ---------------------------------------------------------------------------

class TestWriteRead:
    def test_write_then_read_roundtrip(self, tmp_path):
        """Write a plan and read it back; both plans compare equal."""
        plan = _make_plan()
        key = cache_key(_TASK, _SKELETON, _PLANNER, _DRAFTER)
        write_cached_plan(key, plan, cache_root=tmp_path)
        result = read_cached_plan(key, cache_root=tmp_path)
        assert result is not None
        assert result.task == plan.task
        assert len(result.units) == len(plan.units)
        assert result.units[0].id == plan.units[0].id

    def test_read_miss_returns_none(self, tmp_path):
        """Non-existent cache file returns None."""
        key = cache_key(_TASK, _SKELETON, _PLANNER, _DRAFTER)
        assert read_cached_plan(key, cache_root=tmp_path) is None

    def test_read_expired_returns_none(self, tmp_path, monkeypatch):
        """A plan older than ttl_seconds is treated as a cache miss."""
        plan = _make_plan()
        key = cache_key(_TASK, _SKELETON, _PLANNER, _DRAFTER)
        write_cached_plan(key, plan, cache_root=tmp_path)

        two_days = DEFAULT_TTL_SECONDS * 2
        real_time = time.time()
        monkeypatch.setattr(
            "hierocode.broker.plan_cache.time.time",
            lambda: real_time + two_days,
        )
        assert read_cached_plan(key, ttl_seconds=DEFAULT_TTL_SECONDS, cache_root=tmp_path) is None

    def test_read_corrupt_returns_none(self, tmp_path):
        """A file with invalid JSON is treated as a cache miss."""
        key = cache_key(_TASK, _SKELETON, _PLANNER, _DRAFTER)
        corrupt = tmp_path / f"{key}.json"
        corrupt.write_text("{bad json", encoding="utf-8")
        assert read_cached_plan(key, cache_root=tmp_path) is None

    def test_read_wrong_schema_version_returns_none(self, tmp_path):
        """A file with an unexpected schema_version is treated as a cache miss."""
        plan = _make_plan()
        key = cache_key(_TASK, _SKELETON, _PLANNER, _DRAFTER)
        payload = {
            "cached_at": time.time(),
            "schema_version": 999,
            "plan": plan.model_dump(mode="json"),
        }
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert read_cached_plan(key, cache_root=tmp_path) is None

    def test_read_invalid_plan_returns_none(self, tmp_path):
        """A file whose plan field fails pydantic validation is treated as a cache miss."""
        key = cache_key(_TASK, _SKELETON, _PLANNER, _DRAFTER)
        # Plan.units must be non-empty — supply an empty list to trigger validation error.
        payload = {
            "cached_at": time.time(),
            "schema_version": 1,
            "plan": {"task": "something", "units": []},
        }
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert read_cached_plan(key, cache_root=tmp_path) is None


# ---------------------------------------------------------------------------
# cache_dir
# ---------------------------------------------------------------------------

class TestCacheDir:
    def test_cache_dir_respects_xdg_cache_home(self, tmp_path, monkeypatch):
        """When XDG_CACHE_HOME is set, cache_dir() is rooted there."""
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        result = cache_dir()
        assert str(result).startswith(str(tmp_path))

    def test_cache_dir_falls_back_to_home_cache(self, tmp_path, monkeypatch):
        """Without XDG_CACHE_HOME, cache_dir() falls back to ~/.cache/hierocode/plans."""
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        result = cache_dir()
        assert result == tmp_path / ".cache" / "hierocode" / "plans"

    def test_cache_dir_creates_missing_tree(self, tmp_path, monkeypatch):
        """cache_dir() creates the directory tree if it does not exist."""
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        result = cache_dir()
        assert result.is_dir()


# ---------------------------------------------------------------------------
# Atomicity
# ---------------------------------------------------------------------------

class TestAtomicWrite:
    def test_write_is_atomic_ish(self, tmp_path):
        """After write_cached_plan, final file exists and no .tmp file remains."""
        plan = _make_plan()
        key = cache_key(_TASK, _SKELETON, _PLANNER, _DRAFTER)
        write_cached_plan(key, plan, cache_root=tmp_path)
        final = tmp_path / f"{key}.json"
        tmp_file = tmp_path / f"{key}.json.tmp"
        assert final.exists()
        assert not tmp_file.exists()


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------

class TestClearCache:
    def test_clear_cache_deletes_all_and_returns_count(self, tmp_path):
        """clear_cache deletes all plan-*.json files and returns the correct count."""
        for i in range(3):
            plan = _make_plan(task=f"task {i}", unit_id=f"u{i}")
            key = cache_key(f"task {i}", _SKELETON, _PLANNER, _DRAFTER)
            write_cached_plan(key, plan, cache_root=tmp_path)

        count = clear_cache(cache_root=tmp_path)
        assert count == 3
        assert list(tmp_path.glob("plan-*.json")) == []

    def test_clear_cache_ignores_non_plan_files(self, tmp_path):
        """clear_cache leaves non-plan files in place and counts only plan-*.json."""
        other = tmp_path / "something.txt"
        other.write_text("keep me", encoding="utf-8")

        for i in range(2):
            plan = _make_plan(task=f"task {i}", unit_id=f"u{i}")
            key = cache_key(f"task {i}", _SKELETON, _PLANNER, _DRAFTER)
            write_cached_plan(key, plan, cache_root=tmp_path)

        count = clear_cache(cache_root=tmp_path)
        assert count == 2
        assert other.exists()
        assert other.read_text(encoding="utf-8") == "keep me"
