"""Subscription-quota tracking and rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from hierocode.broker.pricing import PricingConfig, SubscriptionQuota, get_pricing
from hierocode.broker.usage import UsageAccumulator


WarningLevel = Literal["none", "dim", "yellow", "red"]


@dataclass
class QuotaStatus:
    """A snapshot of subscription-quota pressure for a given planner path."""

    provider_type: str        # "claude_code_cli" | "codex_cli" | ...
    subscription_key: str     # "claude_pro" | "chatgpt_plus" | ...
    messages_used: int
    messages_limit: int
    window_hours: int
    percent_used: float       # 0.0 - 1.0+ (can exceed 1.0 if user overshoots)
    warning_level: WarningLevel


# Mapping from provider type (the ProviderConfig.type literal) to the
# subscription_quotas key. This is the only coupling between provider names
# and quota config keys.
_PROVIDER_TO_QUOTA: dict[str, str] = {
    "claude_code_cli": "claude_pro",
    "codex_cli": "chatgpt_plus",
}

_FRIENDLY_NAMES: dict[str, str] = {
    "claude_pro": "Pro",
    "chatgpt_plus": "Plus",
}


def classify_warning(percent_used: float) -> WarningLevel:
    """Threshold-based warning level for a usage fraction."""
    if percent_used < 0.5:
        return "none"
    if percent_used < 0.75:
        return "dim"
    if percent_used < 0.9:
        return "yellow"
    return "red"


def compute_status(
    usage: UsageAccumulator,
    planner_provider_type: str,
    pricing: Optional[PricingConfig] = None,
) -> Optional[QuotaStatus]:
    """Return a QuotaStatus for the planner's subscription path, or None if
    the provider isn't subscription-metered OR no quota is configured for it."""
    subscription_key = _PROVIDER_TO_QUOTA.get(planner_provider_type)
    if subscription_key is None:
        return None

    if pricing is None:
        pricing = get_pricing()

    quota: Optional[SubscriptionQuota] = pricing.subscription_quotas.get(subscription_key)
    if quota is None:
        return None

    messages_used = usage.total_messages()

    if quota.messages_per_window > 0:
        percent_used = messages_used / quota.messages_per_window
    else:
        percent_used = 0.0

    warning_level = classify_warning(percent_used)

    return QuotaStatus(
        provider_type=planner_provider_type,
        subscription_key=subscription_key,
        messages_used=messages_used,
        messages_limit=quota.messages_per_window,
        window_hours=quota.window_hours,
        percent_used=percent_used,
        warning_level=warning_level,
    )


def render_progress_bar(percent: float, width: int = 10) -> str:
    """Unicode block progress bar, filled=▰ / empty=▱. Caps display at 100%.
    Width is the total number of cells. Returns e.g. '▰▰▱▱▱▱▱▱▱▱'."""
    filled_cells = min(width, int(percent * width + 0.5))
    return "▰" * filled_cells + "▱" * (width - filled_cells)


def render_quota_line(status: QuotaStatus) -> str:
    """Return a single-line Rich markup string suitable for console.print(), including
    bar, percent, warning color, and a short hint when warning_level >= yellow.
    Example: '▰▰▱▱▱▱▱▱▱▱  [dim]4 / 40 messages (Pro 5h) — 10% used[/dim]'"""
    bar = render_progress_bar(status.percent_used)
    pct = int(status.percent_used * 100)
    friendly_name = _FRIENDLY_NAMES.get(
        status.subscription_key, status.subscription_key.capitalize()
    )

    level = status.warning_level
    if level == "none":
        color_open, color_close = "", ""
    elif level == "dim":
        color_open, color_close = "[dim]", "[/dim]"
    elif level == "yellow":
        color_open, color_close = "[yellow]", "[/yellow]"
    else:  # red
        color_open, color_close = "[bold red]", "[/bold red]"

    if level == "yellow":
        hint = " — consider /estimate before next /run"
    elif level == "red":
        hint = " — next task may exceed quota"
    else:
        hint = ""

    body = (
        f"{status.messages_used} / {status.messages_limit} messages"
        f" ({friendly_name} {status.window_hours}h) — {pct}% used{hint}"
    )
    return f"{bar}  {color_open}{body}{color_close}"
