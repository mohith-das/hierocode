"""Tests for shell_handlers/usage_cmd.py — /usage handler."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
from unittest.mock import MagicMock, patch

from rich.console import Console

from hierocode.broker.pricing import PricingConfig, SubscriptionQuota
from hierocode.broker.usage import UsageAccumulator, UsageInfo


# ---------------------------------------------------------------------------
# Minimal context stubs (mirrors test_broker_cmds.py approach)
# ---------------------------------------------------------------------------


@dataclass
class _SessionState:
    repo_root: Path
    interaction_mode: Literal["prompt", "immediate"] = "prompt"
    last_plan: Optional[object] = None
    last_diff: Optional[str] = None
    task_history: list = field(default_factory=list)
    usage: UsageAccumulator = field(default_factory=UsageAccumulator)


@dataclass
class _HandlerContext:
    args: list
    session: _SessionState
    config: object
    console: Console
    reload_config: object


def _make_console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    con = Console(file=buf, highlight=False, markup=False)
    return con, buf


def _make_ctx(session=None, console=None):
    con = console or _make_console()[0]
    return _HandlerContext(
        args=[],
        session=session or _SessionState(repo_root=Path(".")),
        config=MagicMock(),
        console=con,
        reload_config=lambda: None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_handle_usage_empty_session_prints_no_calls_yet():
    con, buf = _make_console()
    session = _SessionState(repo_root=Path("."))
    ctx = _make_ctx(session=session, console=con)

    from hierocode.shell_handlers.usage_cmd import handle_usage

    result = handle_usage(ctx)

    assert result == "continue"
    out = buf.getvalue()
    assert "No calls yet" in out


def test_handle_usage_prints_each_active_role():
    con, buf = _make_console()
    session = _SessionState(repo_root=Path("."))
    session.usage.record(
        "planner",
        UsageInfo(
            input_tokens=1000,
            output_tokens=500,
            messages=0,
            provider_type="anthropic",
            model="claude-sonnet-4-6",
        ),
    )
    session.usage.record(
        "drafter",
        UsageInfo(
            input_tokens=2000,
            output_tokens=800,
            messages=0,
            provider_type="ollama",
            model="llama3:8b",
        ),
    )
    ctx = _make_ctx(session=session, console=con)

    from hierocode.shell_handlers.usage_cmd import handle_usage

    result = handle_usage(ctx)

    assert result == "continue"
    out = buf.getvalue()
    assert "Planner" in out
    assert "Drafter" in out
    # reviewer had no calls — should not appear
    assert "Reviewer" not in out
    assert "anthropic" in out
    assert "ollama" in out
    assert "1,000" in out or "1000" in out


def test_handle_usage_prints_subscription_total_when_applicable():
    """Subscription-metered roles should each show their messages-billed count.

    The old aggregate 'Total messages billed against subscription' line was
    replaced by the per-planner quota bar in v0.3.2; aggregate messaging is
    covered separately by the quota-bar tests.
    """
    con, buf = _make_console()
    session = _SessionState(repo_root=Path("."))
    session.usage.record(
        "planner",
        UsageInfo(messages=1, provider_type="claude_code_cli", model="claude-sonnet-4-6"),
    )
    session.usage.record(
        "reviewer",
        UsageInfo(messages=1, provider_type="codex_cli", model="gpt-4o"),
    )
    ctx = _make_ctx(session=session, console=con)

    from hierocode.shell_handlers.usage_cmd import handle_usage

    handle_usage(ctx)

    out = buf.getvalue().lower()
    # Both roles should surface their messages_billed line.
    assert out.count("messages billed") == 2


def test_handle_usage_shows_cost_for_anthropic():
    con, buf = _make_console()
    session = _SessionState(repo_root=Path("."))
    session.usage.record(
        "planner",
        UsageInfo(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            provider_type="anthropic",
            model="claude-sonnet-4-6",
        ),
    )
    ctx = _make_ctx(session=session, console=con)

    from hierocode.shell_handlers.usage_cmd import handle_usage

    handle_usage(ctx)

    out = buf.getvalue()
    # Should print approximate cost
    assert "$" in out or "cost" in out.lower()


def _make_config_with_planner(provider_type: str) -> object:
    """Build a minimal config object that resolves the planner provider type."""
    provider_cfg = MagicMock()
    provider_cfg.type = provider_type

    routing = MagicMock()
    routing.planner = MagicMock()
    routing.planner.provider = "planner_provider"

    config = MagicMock()
    config.routing = routing
    config.providers = {"planner_provider": provider_cfg}
    return config


def _make_quota_pricing(
    subscription_key: str,
    messages_per_window: int = 40,
    window_hours: int = 5,
) -> PricingConfig:
    return PricingConfig(
        anthropic_models={},
        openai_models={},
        subscription_quotas={
            subscription_key: SubscriptionQuota(
                messages_per_window=messages_per_window,
                window_hours=window_hours,
            )
        },
    )


def test_handle_usage_shows_quota_bar_for_subscription_planner():
    con, buf = _make_console()
    session = _SessionState(repo_root=Path("."))
    for _ in range(4):
        session.usage.record(
            "planner",
            UsageInfo(messages=1, provider_type="claude_code_cli", model="claude-sonnet-4-6"),
        )
    config = _make_config_with_planner("claude_code_cli")
    ctx = _HandlerContext(
        args=[],
        session=session,
        config=config,
        console=con,
        reload_config=lambda: None,
    )

    pricing = _make_quota_pricing("claude_pro", messages_per_window=40, window_hours=5)
    with patch("hierocode.broker.quota.get_pricing", return_value=pricing):
        from hierocode.shell_handlers.usage_cmd import handle_usage
        handle_usage(ctx)

    out = buf.getvalue()
    assert "▰" in out
    assert "4 / 40" in out


def test_handle_usage_omits_quota_when_planner_not_subscription():
    con, buf = _make_console()
    session = _SessionState(repo_root=Path("."))
    session.usage.record(
        "planner",
        UsageInfo(input_tokens=100, provider_type="ollama", model="llama3:8b"),
    )
    config = _make_config_with_planner("ollama")
    ctx = _HandlerContext(
        args=[],
        session=session,
        config=config,
        console=con,
        reload_config=lambda: None,
    )

    pricing = _make_quota_pricing("claude_pro")
    with patch("hierocode.broker.quota.get_pricing", return_value=pricing):
        from hierocode.shell_handlers.usage_cmd import handle_usage
        handle_usage(ctx)

    out = buf.getvalue()
    # Quota bar uses block characters — should not appear for non-subscription providers
    assert "▰" not in out
    assert "▱" not in out


def test_handle_usage_omits_quota_when_zero_messages():
    con, buf = _make_console()
    session = _SessionState(repo_root=Path("."))
    # Planner call but no messages counted (e.g. API-mode planner)
    session.usage.record(
        "planner",
        UsageInfo(input_tokens=500, messages=0, provider_type="claude_code_cli", model="m"),
    )
    config = _make_config_with_planner("claude_code_cli")
    ctx = _HandlerContext(
        args=[],
        session=session,
        config=config,
        console=con,
        reload_config=lambda: None,
    )

    pricing = _make_quota_pricing("claude_pro")
    with patch("hierocode.broker.quota.get_pricing", return_value=pricing):
        from hierocode.shell_handlers.usage_cmd import handle_usage
        handle_usage(ctx)

    out = buf.getvalue()
    # messages_used == 0, so quota bar should be suppressed
    assert "▰" not in out
    assert "▱" not in out


def test_register_all_registers_usage_command():
    registry = MagicMock()

    from hierocode.shell_handlers.usage_cmd import register_all

    register_all(registry)

    registry.register.assert_called_once()
    name_arg = registry.register.call_args[0][0]
    assert name_arg == "usage"
