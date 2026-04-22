"""Tests for hierocode.broker.usage — UsageInfo, RoleUsage, UsageAccumulator, cost helpers."""

from unittest.mock import patch

from hierocode.broker.pricing import PricingConfig
from hierocode.broker.usage import (
    ANTHROPIC_PRICING,
    RoleUsage,
    UsageAccumulator,
    UsageInfo,
    estimate_api_cost_usd,
)


# ---------------------------------------------------------------------------
# UsageInfo defaults
# ---------------------------------------------------------------------------


def test_usage_info_defaults():
    ui = UsageInfo()
    assert ui.input_tokens == 0
    assert ui.output_tokens == 0
    assert ui.cache_creation_input_tokens == 0
    assert ui.cache_read_input_tokens == 0
    assert ui.messages == 0
    assert ui.provider_type == ""
    assert ui.model == ""


# ---------------------------------------------------------------------------
# RoleUsage accumulation
# ---------------------------------------------------------------------------


def test_role_usage_accumulates_correctly():
    r = RoleUsage()
    r.calls += 2
    r.input_tokens += 100
    r.output_tokens += 50
    assert r.calls == 2
    assert r.input_tokens == 100
    assert r.output_tokens == 50


# ---------------------------------------------------------------------------
# UsageAccumulator.record
# ---------------------------------------------------------------------------


def test_accumulator_record_increments_calls_and_tokens():
    acc = UsageAccumulator()
    ui = UsageInfo(
        input_tokens=200,
        output_tokens=80,
        messages=0,
        provider_type="anthropic",
        model="claude-sonnet-4-6",
    )
    acc.record("planner", ui)
    acc.record("planner", ui)

    assert acc.planner.calls == 2
    assert acc.planner.input_tokens == 400
    assert acc.planner.output_tokens == 160
    assert acc.planner.messages == 0


def test_accumulator_record_captures_latest_model_and_provider():
    acc = UsageAccumulator()
    ui1 = UsageInfo(input_tokens=10, provider_type="anthropic", model="claude-haiku-4-5")
    ui2 = UsageInfo(input_tokens=20, provider_type="anthropic", model="claude-sonnet-4-6")
    acc.record("drafter", ui1)
    acc.record("drafter", ui2)

    # Most recent model wins
    assert acc.drafter.model == "claude-sonnet-4-6"
    assert acc.drafter.provider_type == "anthropic"
    assert acc.drafter.input_tokens == 30


def test_accumulator_record_empty_strings_do_not_overwrite():
    """A UsageInfo with no provider_type/model should not overwrite existing values."""
    acc = UsageAccumulator()
    ui1 = UsageInfo(input_tokens=10, provider_type="ollama", model="llama3")
    ui2 = UsageInfo(input_tokens=5, provider_type="", model="")
    acc.record("reviewer", ui1)
    acc.record("reviewer", ui2)

    assert acc.reviewer.provider_type == "ollama"
    assert acc.reviewer.model == "llama3"


# ---------------------------------------------------------------------------
# total_messages
# ---------------------------------------------------------------------------


def test_total_messages_sums_roles():
    acc = UsageAccumulator()
    acc.record("planner", UsageInfo(messages=1, provider_type="claude_code_cli", model="m"))
    acc.record("drafter", UsageInfo(messages=0, provider_type="ollama", model="m"))
    acc.record("reviewer", UsageInfo(messages=1, provider_type="codex_cli", model="m"))

    assert acc.total_messages() == 2


# ---------------------------------------------------------------------------
# estimate_api_cost_usd
# ---------------------------------------------------------------------------


def test_estimate_api_cost_for_anthropic_sonnet():
    r = RoleUsage(
        calls=1,
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        provider_type="anthropic",
        model="claude-sonnet-4-6",
    )
    cost = estimate_api_cost_usd(r)
    assert cost is not None
    # Sonnet: $3/M input, $15/M output => $18 total
    assert abs(cost - 18.0) < 0.001


def test_estimate_api_cost_none_for_ollama():
    r = RoleUsage(
        calls=1,
        input_tokens=500_000,
        output_tokens=200_000,
        provider_type="ollama",
        model="llama3:8b",
    )
    cost = estimate_api_cost_usd(r)
    assert cost is None


def test_estimate_api_cost_none_for_empty_model():
    r = RoleUsage(
        calls=1,
        input_tokens=100,
        output_tokens=50,
        provider_type="anthropic",
        model="",
    )
    cost = estimate_api_cost_usd(r)
    assert cost is None


def test_estimate_api_cost_handles_unknown_model_as_haiku():
    """Unknown Anthropic model should fall back to haiku pricing."""
    r = RoleUsage(
        calls=1,
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        provider_type="anthropic",
        model="claude-unknown-99",
    )
    cost = estimate_api_cost_usd(r)
    haiku_input, haiku_output = ANTHROPIC_PRICING["claude-haiku-4-5"]
    expected = haiku_input + haiku_output  # per 1M tokens each
    assert cost is not None
    assert abs(cost - expected) < 0.001


def test_cache_read_discount_applied():
    """Cache reads at 10% of input price should make cost lower than full token price."""
    # Full price: 1M input tokens at sonnet rate = $3.00
    r_full = RoleUsage(
        calls=1,
        input_tokens=1_000_000,
        output_tokens=0,
        provider_type="anthropic",
        model="claude-sonnet-4-6",
    )
    # Same token count but all cache reads (10% of input price)
    r_cached = RoleUsage(
        calls=1,
        input_tokens=1_000_000,
        output_tokens=0,
        cache_read_input_tokens=1_000_000,
        provider_type="anthropic",
        model="claude-sonnet-4-6",
    )
    cost_full = estimate_api_cost_usd(r_full)
    cost_cached = estimate_api_cost_usd(r_cached)
    assert cost_full is not None
    assert cost_cached is not None
    assert cost_cached < cost_full


def test_cache_write_premium_applied():
    """Cache writes at 1.25x input price should cost more than regular tokens."""
    r_regular = RoleUsage(
        calls=1,
        input_tokens=1_000_000,
        output_tokens=0,
        provider_type="anthropic",
        model="claude-sonnet-4-6",
    )
    r_write = RoleUsage(
        calls=1,
        input_tokens=1_000_000,
        output_tokens=0,
        cache_creation_input_tokens=1_000_000,
        provider_type="anthropic",
        model="claude-sonnet-4-6",
    )
    cost_regular = estimate_api_cost_usd(r_regular)
    cost_write = estimate_api_cost_usd(r_write)
    assert cost_regular is not None
    assert cost_write is not None
    assert cost_write > cost_regular


def test_estimate_api_cost_uses_loaded_pricing():
    """estimate_api_cost_usd reads from get_pricing(), not a hardcoded dict."""
    fake_config = PricingConfig(
        anthropic_models={"test-model": (1.0, 5.0)},
    )
    with patch("hierocode.broker.pricing.get_pricing", return_value=fake_config):
        r = RoleUsage(
            calls=1,
            input_tokens=1_000_000,
            output_tokens=100_000,
            provider_type="anthropic",
            model="test-model",
        )
        cost = estimate_api_cost_usd(r)
    # input: 1M * $1/M = $1.00; output: 0.1M * $5/M = $0.50 => $1.50
    assert cost is not None
    assert abs(cost - 1.5) < 0.001
