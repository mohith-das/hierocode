"""Pricing and subscription-quota config loader for hierocode.

Reads ``~/.hierocode/pricing.yaml`` (or ``$XDG_CONFIG_HOME/hierocode/pricing.yaml``)
and merges user values over hardcoded defaults.  Missing or malformed files always
fall back to defaults — startup never crashes on a bad pricing file.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubscriptionQuota:
    messages_per_window: int
    window_hours: int


@dataclass
class PricingConfig:
    """Loaded pricing and subscription-quota data. All fields have defaults so
    partial user files merge cleanly over hardcoded values."""

    anthropic_models: dict[str, tuple[float, float]] = field(default_factory=dict)
    openai_models: dict[str, tuple[float, float]] = field(default_factory=dict)
    subscription_quotas: dict[str, SubscriptionQuota] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hardcoded defaults (last-resort fallback). Keep this in sync with reality.
# ---------------------------------------------------------------------------

_DEFAULT_ANTHROPIC_MODELS: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (0.25, 1.25),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-7": (15.0, 75.0),
}

_DEFAULT_OPENAI_MODELS: dict[str, tuple[float, float]] = {
    # Populated later when we add OpenAI API cost tracking; empty for v0.3.2.
}

_DEFAULT_SUBSCRIPTION_QUOTAS: dict[str, SubscriptionQuota] = {
    "claude_pro": SubscriptionQuota(messages_per_window=40, window_hours=5),
    "chatgpt_plus": SubscriptionQuota(messages_per_window=50, window_hours=3),
}


def default_pricing() -> PricingConfig:
    """Return a fresh PricingConfig with hardcoded defaults (copies — safe to mutate)."""
    return PricingConfig(
        anthropic_models=dict(_DEFAULT_ANTHROPIC_MODELS),
        openai_models=dict(_DEFAULT_OPENAI_MODELS),
        subscription_quotas=dict(_DEFAULT_SUBSCRIPTION_QUOTAS),
    )


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def pricing_config_path() -> Path:
    """Resolve the user's pricing config file path.

    Uses ``$XDG_CONFIG_HOME/hierocode/pricing.yaml`` when XDG_CONFIG_HOME is
    set, otherwise ``~/.hierocode/pricing.yaml``.  Does not create the directory.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if xdg:
        return Path(xdg) / "hierocode" / "pricing.yaml"
    return Path.home() / ".hierocode" / "pricing.yaml"


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------


def _is_numeric(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _merge_model_section(
    section: object,
    section_name: str,
    target: dict[str, tuple[float, float]],
) -> None:
    """Validate and merge a user-supplied model pricing section into *target*."""
    if not isinstance(section, dict):
        warnings.warn(
            f"pricing.yaml: '{section_name}' must be a mapping; ignoring section.",
            RuntimeWarning,
            stacklevel=4,
        )
        return
    for key, value in section.items():
        if not isinstance(key, str):
            warnings.warn(
                f"pricing.yaml: '{section_name}' key {key!r} is not a string; skipping.",
                RuntimeWarning,
                stacklevel=4,
            )
            continue
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or not _is_numeric(value[0])
            or not _is_numeric(value[1])
        ):
            warnings.warn(
                f"pricing.yaml: '{section_name}.{key}' must be a list of two numbers; skipping.",
                RuntimeWarning,
                stacklevel=4,
            )
            continue
        target[key] = (float(value[0]), float(value[1]))


def _merge_quota_section(
    section: object,
    target: dict[str, SubscriptionQuota],
) -> None:
    """Validate and merge a user-supplied subscription_quotas section into *target*."""
    if not isinstance(section, dict):
        warnings.warn(
            "pricing.yaml: 'subscription_quotas' must be a mapping; ignoring section.",
            RuntimeWarning,
            stacklevel=4,
        )
        return
    for key, value in section.items():
        if not isinstance(key, str):
            warnings.warn(
                f"pricing.yaml: 'subscription_quotas' key {key!r} is not a string; skipping.",
                RuntimeWarning,
                stacklevel=4,
            )
            continue
        if not isinstance(value, dict):
            warnings.warn(
                f"pricing.yaml: 'subscription_quotas.{key}' must be a mapping; skipping.",
                RuntimeWarning,
                stacklevel=4,
            )
            continue
        mpw = value.get("messages_per_window")
        wh = value.get("window_hours")
        if not isinstance(mpw, int) or not isinstance(wh, int) or isinstance(mpw, bool) or isinstance(wh, bool):
            warnings.warn(
                f"pricing.yaml: 'subscription_quotas.{key}' must have integer "
                "'messages_per_window' and 'window_hours'; skipping.",
                RuntimeWarning,
                stacklevel=4,
            )
            continue
        target[key] = SubscriptionQuota(messages_per_window=mpw, window_hours=wh)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_pricing(override_path: Optional[Path] = None) -> PricingConfig:
    """Load pricing from disk (if present) merged over hardcoded defaults.

    - If file missing → return default_pricing().
    - If file malformed → warn via warnings.warn, return default_pricing().
    - If file partial → user values win per-key, missing keys keep defaults.
    - ``override_path`` (tests) overrides the resolved path.
    """
    path = override_path if override_path is not None else pricing_config_path()

    if not path.exists():
        return default_pricing()

    raw_text = path.read_text(encoding="utf-8")

    if yaml is None:  # pragma: no cover
        warnings.warn(
            "pricing.yaml found but PyYAML is not installed; using defaults.",
            RuntimeWarning,
            stacklevel=2,
        )
        return default_pricing()

    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        warnings.warn(
            f"pricing.yaml is malformed and will be ignored ({exc}); using defaults.",
            RuntimeWarning,
            stacklevel=2,
        )
        return default_pricing()

    if data is None:
        # Empty file
        return default_pricing()

    if not isinstance(data, dict):
        warnings.warn(
            "pricing.yaml top-level must be a mapping; using defaults.",
            RuntimeWarning,
            stacklevel=2,
        )
        return default_pricing()

    cfg = default_pricing()

    if "anthropic_models" in data:
        _merge_model_section(data["anthropic_models"], "anthropic_models", cfg.anthropic_models)

    if "openai_models" in data:
        _merge_model_section(data["openai_models"], "openai_models", cfg.openai_models)

    if "subscription_quotas" in data:
        _merge_quota_section(data["subscription_quotas"], cfg.subscription_quotas)

    return cfg


# ---------------------------------------------------------------------------
# Process-wide cache
# ---------------------------------------------------------------------------

_CACHED: Optional[PricingConfig] = None


def get_pricing(reload: bool = False) -> PricingConfig:
    """Return a process-wide cached PricingConfig. Pass reload=True to re-read disk."""
    global _CACHED
    if reload or _CACHED is None:
        _CACHED = load_pricing()
    return _CACHED
