"""Tests for hierocode.broker.quota — QuotaStatus computation and rendering."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hierocode.broker.quota import (
    QuotaStatus,
    classify_warning,
    compute_status,
    render_progress_bar,
    render_quota_line,
)
from hierocode.broker.usage import UsageAccumulator, UsageInfo


# ---------------------------------------------------------------------------
# classify_warning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "percent,expected",
    [
        (0.0, "none"),
        (0.49, "none"),
        (0.5, "dim"),
        (0.74, "dim"),
        (0.75, "yellow"),
        (0.89, "yellow"),
        (0.9, "red"),
        (1.2, "red"),
    ],
)
def test_classify_warning_thresholds(percent, expected):
    assert classify_warning(percent) == expected


# ---------------------------------------------------------------------------
# compute_status
# ---------------------------------------------------------------------------


def _make_pricing(subscription_quotas=None):
    """Build a minimal PricingConfig-like mock."""
    from hierocode.broker.pricing import PricingConfig

    quotas = subscription_quotas if subscription_quotas is not None else {}
    return PricingConfig(
        anthropic_models={},
        openai_models={},
        subscription_quotas=quotas,
    )


def _acc_with_messages(n: int, provider_type: str = "claude_code_cli") -> UsageAccumulator:
    acc = UsageAccumulator()
    for _ in range(n):
        acc.record("planner", UsageInfo(messages=1, provider_type=provider_type, model="m"))
    return acc


def test_compute_status_unknown_provider_type_returns_none():
    acc = UsageAccumulator()
    result = compute_status(acc, "ollama", pricing=_make_pricing())
    assert result is None


def test_compute_status_no_quota_configured_returns_none():
    acc = UsageAccumulator()
    result = compute_status(acc, "claude_code_cli", pricing=_make_pricing(subscription_quotas={}))
    assert result is None


def test_compute_status_happy_path():
    from hierocode.broker.pricing import SubscriptionQuota

    pricing = _make_pricing(
        subscription_quotas={"claude_pro": SubscriptionQuota(messages_per_window=40, window_hours=5)}
    )
    acc = _acc_with_messages(10, provider_type="claude_code_cli")
    status = compute_status(acc, "claude_code_cli", pricing=pricing)

    assert status is not None
    assert status.messages_used == 10
    assert status.messages_limit == 40
    assert status.percent_used == pytest.approx(0.25)
    assert status.warning_level == "none"
    assert status.window_hours == 5
    assert status.subscription_key == "claude_pro"
    assert status.provider_type == "claude_code_cli"


def test_compute_status_codex_plus_path():
    from hierocode.broker.pricing import SubscriptionQuota

    pricing = _make_pricing(
        subscription_quotas={"chatgpt_plus": SubscriptionQuota(messages_per_window=50, window_hours=3)}
    )
    acc = _acc_with_messages(10, provider_type="codex_cli")
    status = compute_status(acc, "codex_cli", pricing=pricing)

    assert status is not None
    assert status.messages_limit == 50
    assert status.window_hours == 3
    assert status.subscription_key == "chatgpt_plus"
    assert status.provider_type == "codex_cli"


def test_compute_status_zero_limit_doesnt_crash():
    from hierocode.broker.pricing import SubscriptionQuota

    pricing = _make_pricing(
        subscription_quotas={"claude_pro": SubscriptionQuota(messages_per_window=0, window_hours=5)}
    )
    acc = _acc_with_messages(5, provider_type="claude_code_cli")
    status = compute_status(acc, "claude_code_cli", pricing=pricing)

    assert status is not None
    assert status.percent_used == 0.0


def test_compute_status_overflow_percent():
    from hierocode.broker.pricing import SubscriptionQuota

    pricing = _make_pricing(
        subscription_quotas={"claude_pro": SubscriptionQuota(messages_per_window=40, window_hours=5)}
    )
    acc = _acc_with_messages(50, provider_type="claude_code_cli")
    status = compute_status(acc, "claude_code_cli", pricing=pricing)

    assert status is not None
    assert status.percent_used == pytest.approx(1.25)
    assert status.warning_level == "red"


def test_compute_status_uses_injected_pricing_when_provided():
    from hierocode.broker.pricing import SubscriptionQuota

    pricing = _make_pricing(
        subscription_quotas={"claude_pro": SubscriptionQuota(messages_per_window=40, window_hours=5)}
    )
    acc = _acc_with_messages(4, provider_type="claude_code_cli")

    with patch("hierocode.broker.quota.get_pricing") as mock_get:
        status = compute_status(acc, "claude_code_cli", pricing=pricing)
        mock_get.assert_not_called()

    assert status is not None
    assert status.messages_used == 4


# ---------------------------------------------------------------------------
# render_progress_bar
# ---------------------------------------------------------------------------


def test_render_progress_bar_empty():
    bar = render_progress_bar(0.0, width=10)
    assert bar == "▱▱▱▱▱▱▱▱▱▱"
    assert len(bar) == 10


def test_render_progress_bar_full():
    bar = render_progress_bar(1.0, width=10)
    assert bar == "▰▰▰▰▰▰▰▰▰▰"
    assert len(bar) == 10


def test_render_progress_bar_quarter():
    # 0.2 * 10 = 2.0 → rounds to 2 filled cells
    bar = render_progress_bar(0.2, width=10)
    assert bar == "▰▰▱▱▱▱▱▱▱▱"
    assert len(bar) == 10


def test_render_progress_bar_caps_at_width():
    # Over 100% should not produce more than width cells
    bar = render_progress_bar(2.0, width=10)
    assert bar == "▰▰▰▰▰▰▰▰▰▰"
    assert len(bar) == 10


# ---------------------------------------------------------------------------
# render_quota_line
# ---------------------------------------------------------------------------


def _make_status(
    messages_used: int = 4,
    messages_limit: int = 40,
    window_hours: int = 5,
    subscription_key: str = "claude_pro",
    provider_type: str = "claude_code_cli",
    percent_used: float = 0.1,
    warning_level: str = "none",
) -> QuotaStatus:
    return QuotaStatus(
        provider_type=provider_type,
        subscription_key=subscription_key,
        messages_used=messages_used,
        messages_limit=messages_limit,
        window_hours=window_hours,
        percent_used=percent_used,
        warning_level=warning_level,
    )


def test_render_quota_line_includes_bar_and_limits():
    status = _make_status(messages_used=4, messages_limit=40, window_hours=5, percent_used=0.1)
    line = render_quota_line(status)
    assert "▰" in line
    assert "4 / 40" in line
    assert "Pro" in line
    assert "5h" in line
    assert "10%" in line


def test_render_quota_line_yellow_includes_estimate_hint():
    status = _make_status(
        messages_used=30, messages_limit=40, percent_used=0.75, warning_level="yellow"
    )
    line = render_quota_line(status)
    assert "/estimate" in line
    assert "[yellow]" in line


def test_render_quota_line_red_includes_exceed_hint():
    status = _make_status(
        messages_used=40, messages_limit=40, percent_used=1.0, warning_level="red"
    )
    line = render_quota_line(status)
    assert "exceed quota" in line
    assert "[bold red]" in line


def test_render_quota_line_none_has_no_color_markup():
    status = _make_status(percent_used=0.1, warning_level="none")
    line = render_quota_line(status)
    assert "[dim]" not in line
    assert "[yellow]" not in line
    assert "[red]" not in line


def test_render_quota_line_chatgpt_plus_shows_plus():
    status = _make_status(
        subscription_key="chatgpt_plus",
        provider_type="codex_cli",
        messages_used=5,
        messages_limit=50,
        window_hours=3,
        percent_used=0.1,
        warning_level="none",
    )
    line = render_quota_line(status)
    assert "Plus" in line
    assert "3h" in line
