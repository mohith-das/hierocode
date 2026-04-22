"""Handler for the /usage shell command — displays per-role session token/message usage."""

from __future__ import annotations

from typing import Optional

from hierocode.broker.usage import RoleUsage, estimate_api_cost_usd


def _planner_provider_type_from_config(config) -> Optional[str]:
    """Resolve the planner role's ProviderConfig.type, or None if not set."""
    planner_role = getattr(config.routing, "planner", None)
    if planner_role is None:
        return None
    provider_cfg = config.providers.get(planner_role.provider)
    if provider_cfg is None:
        return None
    return provider_cfg.type


def handle_usage(ctx) -> str:
    """Print per-role token/message usage accumulated in this session."""
    u = ctx.session.usage
    ctx.console.print("\n[bold]Session usage[/bold]")
    for role in ("planner", "drafter", "reviewer"):
        r: RoleUsage = getattr(u, role)
        if r.calls == 0:
            continue
        ctx.console.print(
            f"\n[bold]{role.capitalize()}[/bold]  "
            f"([cyan]{r.provider_type}[/cyan] / {r.model})"
        )
        ctx.console.print(f"  calls:            {r.calls}")
        ctx.console.print(f"  input tokens:     {r.input_tokens:,}")
        ctx.console.print(f"  output tokens:    {r.output_tokens:,}")
        if r.cache_read_input_tokens or r.cache_creation_input_tokens:
            ctx.console.print(f"  cache reads:      {r.cache_read_input_tokens:,}")
            ctx.console.print(f"  cache writes:     {r.cache_creation_input_tokens:,}")
        if r.messages:
            ctx.console.print(f"  messages billed:  {r.messages}")
        cost = estimate_api_cost_usd(r)
        if cost is not None:
            ctx.console.print(f"  [green]approx cost:     ${cost:.4f}[/green]")

    # Subscription quota bar — shown when the session has any subscription messages
    # AND a quota is configured for the planner's provider type.
    from hierocode.broker.quota import compute_status, render_quota_line

    planner_provider_type = _planner_provider_type_from_config(ctx.config)
    if planner_provider_type is not None:
        status = compute_status(u, planner_provider_type)
        if status is not None and status.messages_used > 0:
            ctx.console.print("")  # blank line separator
            ctx.console.print(render_quota_line(status))

    if u.planner.calls + u.drafter.calls + u.reviewer.calls == 0:
        ctx.console.print("  [dim]No calls yet this session.[/dim]")
    return "continue"


def register_all(registry) -> None:
    """Register the /usage command."""
    registry.register(
        "usage",
        handle_usage,
        "Show per-role token/message usage for this session.",
    )
