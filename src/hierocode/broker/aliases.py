"""Named task alias CRUD — round-trips ~/.hierocode.yaml under the `tasks:` key."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import yaml

from hierocode.models.schemas import TaskAlias
from hierocode.utils.paths import get_config_path

# ---------------------------------------------------------------------------
# Name validation
# ---------------------------------------------------------------------------

NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"  # 1–64 chars, letters/digits/underscore/hyphen
_NAME_RE = re.compile(NAME_PATTERN)

__all__ = [
    "TaskAlias",
    "AliasError",
    "NAME_PATTERN",
    "list_aliases",
    "get_alias",
    "save_alias",
    "delete_alias",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AliasError(Exception):
    """Raised on invalid alias operations."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_yaml(config_path: Path) -> dict:
    """Load YAML from *config_path*; return {} when the file is missing or empty."""
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except yaml.YAMLError:
        return {}


def _write_yaml_atomic(data: dict, config_path: Path) -> None:
    """Write *data* to *config_path* atomically via a sibling .tmp file."""
    tmp_path = config_path.with_suffix(config_path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False, allow_unicode=True)
        os.replace(tmp_path, config_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_aliases(config_path: Optional[Path] = None) -> list[TaskAlias]:
    """Return all saved task aliases; returns [] when the file or key is absent."""
    path = config_path or get_config_path()
    data = _load_yaml(path)
    raw_tasks = data.get("tasks")
    if not isinstance(raw_tasks, list):
        return []
    aliases: list[TaskAlias] = []
    for item in raw_tasks:
        if isinstance(item, dict) and "name" in item and "description" in item:
            aliases.append(TaskAlias(name=item["name"], description=item["description"]))
    return aliases


def get_alias(name: str, config_path: Optional[Path] = None) -> Optional[TaskAlias]:
    """Return the alias for *name*, or None if it does not exist."""
    for alias in list_aliases(config_path):
        if alias.name == name:
            return alias
    return None


def save_alias(
    name: str,
    description: str,
    config_path: Optional[Path] = None,
) -> TaskAlias:
    """Persist a task alias, overwriting any existing entry with the same name.

    Raises AliasError for invalid name or empty description.
    """
    if not _NAME_RE.match(name):
        raise AliasError(
            f"Invalid alias name {name!r}. "
            "Names must start with a letter or digit and contain only "
            "letters, digits, underscores, or hyphens (1–64 characters)."
        )
    if not description or not description.strip():
        raise AliasError("Alias description must not be empty.")

    path = config_path or get_config_path()
    data = _load_yaml(path)

    if not isinstance(data.get("tasks"), list):
        data["tasks"] = []

    # Overwrite existing entry with the same name, or append.
    tasks: list[dict] = data["tasks"]
    new_entry = {"name": name, "description": description}
    for i, item in enumerate(tasks):
        if isinstance(item, dict) and item.get("name") == name:
            tasks[i] = new_entry
            break
    else:
        tasks.append(new_entry)

    _write_yaml_atomic(data, path)
    return TaskAlias(name=name, description=description)


def delete_alias(name: str, config_path: Optional[Path] = None) -> bool:
    """Remove the alias named *name*.

    Returns True if deleted, False if the alias did not exist.
    """
    path = config_path or get_config_path()
    data = _load_yaml(path)

    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        return False

    new_tasks = [item for item in tasks if not (isinstance(item, dict) and item.get("name") == name)]
    if len(new_tasks) == len(tasks):
        return False  # nothing removed

    data["tasks"] = new_tasks
    _write_yaml_atomic(data, path)
    return True
