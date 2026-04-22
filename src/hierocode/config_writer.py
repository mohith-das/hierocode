"""Helpers for mutating ~/.hierocode.yaml from the CLI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml

from hierocode.models.schemas import HierocodeConfig
from hierocode.utils.paths import get_config_path

VALID_ROLES = ("planner", "drafter", "reviewer")


class ConfigWriteError(Exception):
    """Raised when a config mutation cannot be applied safely."""


def set_role_model(
    role: str,
    model: str,
    provider: Optional[str] = None,
    config_path: Optional[Path] = None,
) -> HierocodeConfig:
    """Update routing.{role} in the config file. Returns the new validated HierocodeConfig.

    If provider is None, reuses the existing provider for that role (or falls back to
    default_provider if no per-role provider is currently set).
    """
    path = config_path or get_config_path()

    if not path.exists():
        raise ConfigWriteError(
            f"Config not found at {path}. Run `hierocode init` first."
        )

    if role not in VALID_ROLES:
        raise ConfigWriteError(
            f"Invalid role '{role}'. Valid: planner, drafter, reviewer."
        )

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data: dict = yaml.safe_load(fh) or {}
    except yaml.YAMLError as err:
        raise ConfigWriteError(f"Could not parse {path}: {err}") from err

    # Resolve provider
    if provider is not None:
        resolved_provider = provider
    else:
        existing_provider = (
            data.get("routing", {}).get(role, {}) or {}
        ).get("provider")
        if existing_provider:
            resolved_provider = existing_provider
        elif data.get("default_provider"):
            resolved_provider = data["default_provider"]
        else:
            raise ConfigWriteError(
                f"Cannot infer provider for role '{role}'; pass provider explicitly."
            )

    if resolved_provider not in data.get("providers", {}):
        raise ConfigWriteError(f"Provider '{resolved_provider}' is not configured.")

    # Ensure routing dict exists
    if "routing" not in data or not isinstance(data["routing"], dict):
        data["routing"] = {}

    data["routing"][role] = {"provider": resolved_provider, "model": model}

    # Validate before writing
    try:
        validated = HierocodeConfig(**data)
    except Exception as err:
        raise ConfigWriteError(f"Resulting config is invalid: {err}") from err

    # Atomic write
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False, allow_unicode=True)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return validated


def list_roles(config_path: Optional[Path] = None) -> dict:
    """Return {role: {'provider': str, 'model': str, 'source': 'explicit' | 'legacy' | 'default'}}.

    'explicit' = set via routing.planner/drafter/reviewer RoleRouting block.
    'legacy'   = derived from old routing.draft_model / review_model + default_provider.
    'default'  = not configured; would fall back to default_provider + legacy defaults.
    """
    path = config_path or get_config_path()

    data: dict = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except yaml.YAMLError:
            data = {}

    routing = data.get("routing") or {}
    default_provider = data.get("default_provider", "local_default")

    # Pydantic defaults for legacy model names
    _default_draft_model = "model-small"
    _default_review_model = "model-large"

    draft_model = routing.get("draft_model", _default_draft_model)
    review_model = routing.get("review_model", _default_review_model)

    result: dict = {}
    for role in VALID_ROLES:
        role_block = routing.get(role)
        if isinstance(role_block, dict) and role_block.get("provider") and role_block.get("model"):
            result[role] = {
                "provider": role_block["provider"],
                "model": role_block["model"],
                "source": "explicit",
            }
        elif role == "drafter" and routing.get("draft_model"):
            result[role] = {
                "provider": default_provider,
                "model": draft_model,
                "source": "legacy",
            }
        elif role in ("planner", "reviewer") and routing.get("review_model"):
            result[role] = {
                "provider": default_provider,
                "model": review_model,
                "source": "legacy",
            }
        else:
            # Default fallback: use Pydantic field defaults
            if role == "drafter":
                fallback_model = _default_draft_model
            else:
                fallback_model = _default_review_model
            result[role] = {
                "provider": default_provider,
                "model": fallback_model,
                "source": "default",
            }

    return result
