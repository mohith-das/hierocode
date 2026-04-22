"""Tests for hierocode.broker.aliases."""

from __future__ import annotations

import pytest
import yaml
from pathlib import Path

from hierocode.broker.aliases import (
    AliasError,
    TaskAlias,
    delete_alias,
    get_alias,
    list_aliases,
    save_alias,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(tmp_path: Path) -> Path:
    """Return a deterministic config path inside tmp_path."""
    return tmp_path / ".hierocode.yaml"


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# list_aliases
# ---------------------------------------------------------------------------


def test_list_aliases_empty_when_no_file(tmp_path: Path):
    """Returns [] when the config file does not exist."""
    result = list_aliases(config_path=_cfg(tmp_path))
    assert result == []


def test_list_aliases_empty_when_no_tasks_key(tmp_path: Path):
    """Returns [] when the YAML exists but has no 'tasks' key."""
    cfg = _cfg(tmp_path)
    _write_yaml(cfg, {"default_provider": "local"})
    result = list_aliases(config_path=cfg)
    assert result == []


# ---------------------------------------------------------------------------
# save_alias
# ---------------------------------------------------------------------------


def test_save_alias_writes_yaml(tmp_path: Path):
    """After saving an alias the YAML file contains it under 'tasks'."""
    cfg = _cfg(tmp_path)
    save_alias("deploy", "push to production", config_path=cfg)
    data = yaml.safe_load(cfg.read_text())
    assert any(t["name"] == "deploy" for t in data["tasks"])
    assert any(t["description"] == "push to production" for t in data["tasks"])


def test_save_alias_returns_task_alias(tmp_path: Path):
    """save_alias returns a TaskAlias with correct fields."""
    cfg = _cfg(tmp_path)
    alias = save_alias("build", "compile the project", config_path=cfg)
    assert isinstance(alias, TaskAlias)
    assert alias.name == "build"
    assert alias.description == "compile the project"


def test_save_alias_rejects_invalid_name(tmp_path: Path):
    """Names containing spaces, slashes, or empty strings raise AliasError."""
    cfg = _cfg(tmp_path)
    for bad in ["has space", "has/slash", "", "!bad"]:
        with pytest.raises(AliasError):
            save_alias(bad, "some description", config_path=cfg)


def test_save_alias_rejects_empty_description(tmp_path: Path):
    """Empty or whitespace-only description raises AliasError."""
    cfg = _cfg(tmp_path)
    for bad_desc in ("", "   "):
        with pytest.raises(AliasError):
            save_alias("valid-name", bad_desc, config_path=cfg)


def test_save_alias_overwrites_existing_name(tmp_path: Path):
    """Saving the same name twice keeps only the latest description."""
    cfg = _cfg(tmp_path)
    save_alias("lint", "run linter v1", config_path=cfg)
    save_alias("lint", "run linter v2", config_path=cfg)
    aliases = list_aliases(config_path=cfg)
    lint_aliases = [a for a in aliases if a.name == "lint"]
    assert len(lint_aliases) == 1
    assert lint_aliases[0].description == "run linter v2"


# ---------------------------------------------------------------------------
# get_alias
# ---------------------------------------------------------------------------


def test_get_alias_found(tmp_path: Path):
    """get_alias returns the correct TaskAlias when it exists."""
    cfg = _cfg(tmp_path)
    save_alias("test", "run test suite", config_path=cfg)
    alias = get_alias("test", config_path=cfg)
    assert alias is not None
    assert alias.name == "test"
    assert alias.description == "run test suite"


def test_get_alias_not_found_returns_none(tmp_path: Path):
    """get_alias returns None when the name is not present."""
    cfg = _cfg(tmp_path)
    assert get_alias("nonexistent", config_path=cfg) is None


# ---------------------------------------------------------------------------
# delete_alias
# ---------------------------------------------------------------------------


def test_delete_alias_true_when_deleted(tmp_path: Path):
    """delete_alias returns True when the alias existed and was removed."""
    cfg = _cfg(tmp_path)
    save_alias("cleanup", "remove temp files", config_path=cfg)
    result = delete_alias("cleanup", config_path=cfg)
    assert result is True
    assert get_alias("cleanup", config_path=cfg) is None


def test_delete_alias_false_when_absent(tmp_path: Path):
    """delete_alias returns False when the alias does not exist."""
    cfg = _cfg(tmp_path)
    result = delete_alias("no-such-alias", config_path=cfg)
    assert result is False


def test_delete_alias_preserves_other_aliases(tmp_path: Path):
    """Deleting one alias leaves sibling aliases untouched."""
    cfg = _cfg(tmp_path)
    save_alias("alpha", "task alpha", config_path=cfg)
    save_alias("beta", "task beta", config_path=cfg)
    delete_alias("alpha", config_path=cfg)
    remaining = list_aliases(config_path=cfg)
    names = [a.name for a in remaining]
    assert "alpha" not in names
    assert "beta" in names


# ---------------------------------------------------------------------------
# Atomic write hygiene
# ---------------------------------------------------------------------------


def test_atomic_write_cleans_up_tmp(tmp_path: Path):
    """After a save there should be no leftover .tmp file in the directory."""
    cfg = _cfg(tmp_path)
    save_alias("check", "run health check", config_path=cfg)
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == [], f"Unexpected .tmp files: {tmp_files}"


# ---------------------------------------------------------------------------
# Preservation of other config fields
# ---------------------------------------------------------------------------


def test_save_preserves_other_config_fields(tmp_path: Path):
    """Saving an alias does not destroy pre-existing config keys."""
    cfg = _cfg(tmp_path)
    _write_yaml(cfg, {
        "default_provider": "anthropic",
        "providers": {
            "anthropic": {"type": "anthropic"},
        },
        "routing": {
            "planner": {"provider": "anthropic", "model": "claude-3-5-sonnet"},
        },
        "tasks": [],
    })

    save_alias("migrate", "run db migrations", config_path=cfg)

    data = yaml.safe_load(cfg.read_text())
    assert data["default_provider"] == "anthropic"
    assert "anthropic" in data["providers"]
    assert data["routing"]["planner"]["model"] == "claude-3-5-sonnet"
    tasks = data["tasks"]
    assert any(t["name"] == "migrate" for t in tasks)
