"""Usage tracking — per-call UsageInfo, per-role accumulator, and cost helpers.

This module is the canonical source for actual (post-run) token/message usage.
It complements broker/estimator.py which handles pre-flight projections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Role = Literal["planner", "drafter", "reviewer"]


def __getattr__(name: str) -> object:
    if name == "ANTHROPIC_PRICING":
        from hierocode.broker.pricing import get_pricing

        return get_pricing().anthropic_models
    raise AttributeError(name)


@dataclass
class UsageInfo:
    """Per-call usage snapshot. All token fields are optional — providers fill
    in what they know. ``messages`` is 1 for subscription-mode CLIs, 0 otherwise."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0  # Anthropic cache writes
    cache_read_input_tokens: int = 0      # Anthropic cache reads (cheap)
    messages: int = 0                     # 1 for claude_code_cli / codex_cli, else 0
    provider_type: str = ""               # "ollama" | "anthropic" | ...
    model: str = ""                       # the specific model id


@dataclass
class RoleUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    messages: int = 0
    provider_type: str = ""
    model: str = ""
    # If the user switches model mid-session, we keep only the most-recent model
    # in this field (display-only). Real analytics can come later.


@dataclass
class UsageAccumulator:
    planner: RoleUsage = field(default_factory=RoleUsage)
    drafter: RoleUsage = field(default_factory=RoleUsage)
    reviewer: RoleUsage = field(default_factory=RoleUsage)

    def record(self, role: Role, usage: UsageInfo) -> None:
        """Accumulate a single-call UsageInfo into the per-role totals."""
        target = getattr(self, role)
        target.calls += 1
        target.input_tokens += usage.input_tokens
        target.output_tokens += usage.output_tokens
        target.cache_creation_input_tokens += usage.cache_creation_input_tokens
        target.cache_read_input_tokens += usage.cache_read_input_tokens
        target.messages += usage.messages
        if usage.provider_type:
            target.provider_type = usage.provider_type
        if usage.model:
            target.model = usage.model

    def total_messages(self) -> int:
        """Sum of subscription-mode messages across all roles (subscription quota tally)."""
        return self.planner.messages + self.drafter.messages + self.reviewer.messages


_HAIKU_FALLBACK = "claude-haiku-4-5"


def estimate_api_cost_usd(usage: RoleUsage) -> Optional[float]:
    """Return $ estimate for Anthropic API calls; None for other providers."""
    if usage.provider_type != "anthropic" or not usage.model:
        return None
    from hierocode.broker.pricing import get_pricing

    anthropic_models = get_pricing().anthropic_models
    prices = anthropic_models.get(usage.model, anthropic_models.get(_HAIKU_FALLBACK, (0.25, 1.25)))
    input_price, output_price = prices
    # Cache reads are ~10% of input price; cache writes are ~1.25x input price.
    # Use Anthropic's documented ratios.
    regular_input = max(
        0,
        usage.input_tokens
        - usage.cache_read_input_tokens
        - usage.cache_creation_input_tokens,
    )
    input_cost = (
        regular_input / 1_000_000 * input_price
        + usage.cache_read_input_tokens / 1_000_000 * input_price * 0.1
        + usage.cache_creation_input_tokens / 1_000_000 * input_price * 1.25
    )
    output_cost = usage.output_tokens / 1_000_000 * output_price
    return input_cost + output_cost
