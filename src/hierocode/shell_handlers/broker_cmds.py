"""Broker-wrapping handlers for the hierocode persistent TUI shell (v0.3)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from rich.live import Live

# cli_shell may not exist on disk yet (W21 is writing it in parallel).
# Tests mock everything so collection still works once both modules are present.
from hierocode.cli_shell import HandlerContext, HandlerResult  # noqa: E402
from hierocode.broker.usage import UsageInfo

from hierocode.broker.budget import pack_context
from hierocode.broker.capacity import build_capacity_profile
from hierocode.broker.dispatcher import run_plan
from hierocode.broker.estimator import estimate_task_cost
from hierocode.broker.plan_cache import cache_key, clear_cache, read_cached_plan, write_cached_plan
from hierocode.broker.plan_schema import TaskUnit
from hierocode.broker.planner import generate_plan
from hierocode.broker.progress import ProgressState, UnitPhase, _build_panel
from hierocode.broker.prompts import build_drafter_prompt
from hierocode.broker.router import get_route
from hierocode.broker.skeleton import build_skeleton
from hierocode.config_writer import ConfigWriteError, list_roles, set_role_model
from hierocode.providers import get_provider
from hierocode.repo.diffing import generate_unified_diff
from hierocode.repo.files import read_file_safe
from hierocode.runtime.gpu import probe_gpu
from hierocode.runtime.resources import get_available_ram_gb, get_cpu_count, get_total_ram_gb
from hierocode.utils.paths import get_config_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences from generated text."""
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def _extract_quota_limit(config) -> Optional[int]:
    """Return the messages-per-window limit for subscription-mode planners, else None.

    v0.3.1 hardcoded defaults; v0.3.2 pulls from pricing.yaml.
    """
    planner_role = getattr(config.routing, "planner", None)
    if planner_role is None:
        return None
    provider_name = planner_role.provider
    provider_cfg = config.providers.get(provider_name)
    if provider_cfg is None:
        return None
    if provider_cfg.type == "claude_code_cli":
        return 40
    if provider_cfg.type == "codex_cli":
        return 50
    return None


class _LivePanelReporter:
    """ProgressReporter implementation that pushes a Rich panel into a Live block."""

    def __init__(self, live: Live, state: ProgressState, usage, quota: Optional[int]) -> None:
        self.live = live
        self.state = state
        self.usage = usage
        self.quota = quota

    def _refresh(self) -> None:
        self.live.update(_build_panel(self.state, self.usage, self.quota))

    def seed(self, task: str, units: list[tuple[str, str]]) -> None:
        """Initialise state from the initial unit list."""
        self.state.task = task
        for uid, goal in units:
            self.state.add_unit(uid, goal)
        self._refresh()

    def enqueue(self, unit_id: str, goal: str) -> None:
        """Add a sub-unit that was created by a split verdict."""
        self.state.add_unit(unit_id, goal)
        self._refresh()

    def phase(self, unit_id: str, phase: UnitPhase) -> None:
        """Transition a unit to a new phase and redraw."""
        self.state.set_phase(unit_id, phase)
        self._refresh()

    def revision(self, unit_id: str) -> None:
        """Bump the revision counter and redraw."""
        self.state.bump_revision(unit_id)
        self._refresh()

    def finished(self) -> None:
        """No-op — the Live context manager handles teardown."""


def _exploration_for_role(config, role: str) -> tuple[str, Optional[list[str]]]:
    """Return (exploration, allowed_tools) for a role, defaulting to passive."""
    routing = config.routing
    role_cfg = getattr(routing, role, None)
    if role_cfg is None:
        return ("passive", None)
    return (role_cfg.exploration, role_cfg.allowed_tools)


def _resolve_providers(ctx: HandlerContext):
    """Return (planner_prov, planner_m, drafter_prov, drafter_m) from config routing."""
    planner_p, planner_m = get_route(ctx.config, "planner")
    drafter_p, drafter_m = get_route(ctx.config, "drafter")
    planner_prov = get_provider(planner_p, ctx.config.providers[planner_p])
    drafter_prov = get_provider(drafter_p, ctx.config.providers[drafter_p])
    return planner_prov, planner_m, drafter_prov, drafter_m


# ---------------------------------------------------------------------------
# /run
# ---------------------------------------------------------------------------

def handle_run(ctx: HandlerContext) -> HandlerResult:
    """Full pipeline: plan -> draft -> QA. Usage: /run <task>"""
    if not ctx.args:
        ctx.console.print("Usage: /run <task>")
        return "continue"

    task = " ".join(ctx.args)
    planner_prov, planner_m, drafter_prov, drafter_m = _resolve_providers(ctx)

    profile = build_capacity_profile(drafter_prov, drafter_m)
    skeleton = build_skeleton(ctx.session.repo_root)
    key = cache_key(task, skeleton, planner_m, drafter_m)

    p_exploration, p_tools = _exploration_for_role(ctx.config, "planner")
    r_exploration, r_tools = _exploration_for_role(ctx.config, "reviewer")

    planned = read_cached_plan(key)
    if planned is not None:
        ctx.console.print(f"Plan cache [green]HIT[/green] (key={key[:16]}...)")
    else:
        ctx.console.print(f"Plan cache MISS — planning with {planner_m}...")
        planned = generate_plan(
            task, skeleton, profile, planner_prov, planner_m,
            exploration=p_exploration, allowed_tools=p_tools,
        )
        if isinstance(planner_prov.last_usage, UsageInfo):
            ctx.session.usage.record("planner", planner_prov.last_usage)
        write_cached_plan(key, planned)

    ctx.console.print(f"Plan has {len(planned.units)} units. Dispatching...")

    progress_state = ProgressState()
    quota = _extract_quota_limit(ctx.config)
    initial_panel = _build_panel(progress_state, ctx.session.usage, quota)

    escalation_cb = None
    if getattr(ctx.config.policy, "warn_before_escalation", False):
        from hierocode.shell_handlers._prompts import (
            EscalationChoice,
            prompt_escalation_approval,
        )

        def _ask_escalation(unit, planner_model, revisions_done=0):
            choice = prompt_escalation_approval(
                unit.id, unit.goal, revisions_done, planner_model,
            )
            return choice == EscalationChoice.APPROVE

        escalation_cb = _ask_escalation

    with Live(initial_panel, console=ctx.console, refresh_per_second=4, transient=False) as live:
        reporter = _LivePanelReporter(live, progress_state, ctx.session.usage, quota)
        result = run_plan(
            planned,
            profile,
            planner_prov,
            planner_m,
            drafter_prov,
            drafter_m,
            ctx.session.repo_root,
            max_revisions_per_unit=ctx.config.policy.max_revisions_per_unit,
            max_escalations_per_task=ctx.config.policy.max_escalations_per_task,
            usage_accumulator=ctx.session.usage,
            progress_reporter=reporter,
            reviewer_exploration=r_exploration,
            reviewer_allowed_tools=r_tools,
            escalation_confirm=escalation_cb,
        )
        # Final panel refresh to show all units in terminal state
        live.update(_build_panel(progress_state, ctx.session.usage, quota))

    ctx.console.print(
        f"\n[bold]Task complete.[/bold] "
        f"revisions={result.total_revisions} escalations={result.total_escalations}"
    )
    for u in result.units:
        status_color = {
            "completed": "green",
            "escalated": "yellow",
            "failed": "red",
            "revised": "cyan",
        }.get(u.status, "white")
        ctx.console.print(
            f"\n[bold][{status_color}]{u.unit_id}[/{status_color}][/bold] "
            f"({u.status}, revisions={u.revision_count})"
        )
        if u.reason:
            ctx.console.print(f"  [dim]{u.reason}[/dim]")
        if u.diff:
            ctx.console.print(u.diff)

    # Collect all diffs
    all_diffs = [u.diff for u in result.units if u.diff]
    ctx.session.last_plan = planned
    ctx.session.last_diff = "\n".join(all_diffs) if all_diffs else None
    ctx.session.task_history.append(task)

    return "continue"


# ---------------------------------------------------------------------------
# /plan  (and /plan show)
# ---------------------------------------------------------------------------

def handle_plan(ctx: HandlerContext) -> HandlerResult:
    """Plan only. Usage: /plan <task> OR /plan show"""
    if ctx.args and ctx.args[0] == "show":
        if ctx.session.last_plan is None:
            ctx.console.print("No plan yet.")
        else:
            plan = ctx.session.last_plan
            ctx.console.print(f"\n[bold]Plan ({len(plan.units)} units):[/bold]")
            for u in plan.units:
                ctx.console.print(f"  [cyan]{u.id}[/cyan]: {u.goal}")
                if u.target_files:
                    ctx.console.print(f"    target: {', '.join(u.target_files)}")
                if u.acceptance:
                    ctx.console.print(f"    [dim]accept: {u.acceptance}[/dim]")
        return "continue"

    if not ctx.args:
        ctx.console.print("Usage: /plan <task> OR /plan show")
        return "continue"

    task = " ".join(ctx.args)
    planner_prov, planner_m, drafter_prov, drafter_m = _resolve_providers(ctx)

    profile = build_capacity_profile(drafter_prov, drafter_m)
    skeleton = build_skeleton(ctx.session.repo_root)
    key = cache_key(task, skeleton, planner_m, drafter_m)

    p_exploration, p_tools = _exploration_for_role(ctx.config, "planner")

    planned = read_cached_plan(key)
    if planned is not None:
        ctx.console.print(f"Plan cache [green]HIT[/green] (key={key[:16]}...)")
    else:
        ctx.console.print(f"Planning with {planner_m}...")
        planned = generate_plan(
            task, skeleton, profile, planner_prov, planner_m,
            exploration=p_exploration, allowed_tools=p_tools,
        )
        if isinstance(planner_prov.last_usage, UsageInfo):
            ctx.session.usage.record("planner", planner_prov.last_usage)
        write_cached_plan(key, planned)

    ctx.console.print(f"\n[bold]Plan ({len(planned.units)} units):[/bold]")
    for u in planned.units:
        ctx.console.print(f"  [cyan]{u.id}[/cyan]: {u.goal}")
        if u.target_files:
            ctx.console.print(f"    target: {', '.join(u.target_files)}")
        if u.acceptance:
            ctx.console.print(f"    [dim]accept: {u.acceptance}[/dim]")

    ctx.session.last_plan = planned
    return "continue"


# ---------------------------------------------------------------------------
# /estimate
# ---------------------------------------------------------------------------

def handle_estimate(ctx: HandlerContext) -> HandlerResult:
    """Predict cost/quota for a task. Usage: /estimate <task>"""
    if not ctx.args:
        ctx.console.print("Usage: /estimate <task>")
        return "continue"

    task = " ".join(ctx.args)
    planner_p, planner_m = get_route(ctx.config, "planner")
    drafter_p, drafter_m = get_route(ctx.config, "drafter")
    drafter_prov = get_provider(drafter_p, ctx.config.providers[drafter_p])

    profile = build_capacity_profile(drafter_prov, drafter_m)
    skeleton = build_skeleton(ctx.session.repo_root)

    est = estimate_task_cost(
        task=task,
        skeleton=skeleton,
        profile=profile,
        planner_provider_config=ctx.config.providers[planner_p],
        planner_model=planner_m,
        max_revisions_per_unit=ctx.config.policy.max_revisions_per_unit,
    )

    ctx.console.print("\n[bold]Cost estimate[/bold]")
    ctx.console.print(f"Planner: [magenta]{est.planner_kind}[/magenta] ({planner_m})")
    ctx.console.print(f"Expected plan units: {est.expected_plan_units}")
    ctx.console.print(f"Expected QA calls: {est.expected_qa_calls}")
    ctx.console.print(
        f"Planner tokens (in/out): {est.planner_input_tokens} / {est.planner_output_tokens}"
    )
    if est.approximate_cost_usd is not None:
        ctx.console.print(f"[green]Approx. cost: ${est.approximate_cost_usd:.4f}[/green]")
    if est.approximate_message_count is not None:
        ctx.console.print(
            f"[yellow]Approx. messages against quota: {est.approximate_message_count}[/yellow]"
        )
    for note in est.notes:
        ctx.console.print(f"  [dim]* {note}[/dim]")

    return "continue"


# ---------------------------------------------------------------------------
# /draft
# ---------------------------------------------------------------------------

def handle_draft(ctx: HandlerContext) -> HandlerResult:
    """Single-file draft. Usage: /draft <task> <file>"""
    if len(ctx.args) < 2:
        ctx.console.print("Usage: /draft <task> <file>  (last arg is the target file)")
        return "continue"

    filepath = ctx.args[-1]
    task = " ".join(ctx.args[:-1])

    drafter_p, drafter_m = get_route(ctx.config, "drafter")
    drafter_prov = get_provider(drafter_p, ctx.config.providers[drafter_p])

    profile = build_capacity_profile(drafter_prov, drafter_m)
    unit = TaskUnit(id="draft", goal=task, target_files=[filepath], acceptance="")
    packed = pack_context(unit, profile, ctx.session.repo_root)
    prompt = build_drafter_prompt(unit, packed.content)

    ctx.console.print(f"Drafting {filepath} with {drafter_m}...")
    drafted = drafter_prov.generate(prompt=prompt, model=drafter_m)
    if isinstance(drafter_prov.last_usage, UsageInfo):
        ctx.session.usage.record("drafter", drafter_prov.last_usage)
    cleaned = _strip_code_fences(drafted)

    original = read_file_safe(Path(ctx.session.repo_root) / filepath)
    diff = generate_unified_diff(original, cleaned, filepath)

    ctx.console.print("\n[bold]Proposed Patch:[/bold]")
    ctx.console.print(diff if diff else "No changes proposed.")

    ctx.session.last_diff = diff or None
    return "continue"


# ---------------------------------------------------------------------------
# /review
# ---------------------------------------------------------------------------

def handle_review(ctx: HandlerContext) -> HandlerResult:
    """Single-file review. Usage: /review <task> <file>"""
    if len(ctx.args) < 2:
        ctx.console.print("Usage: /review <task> <file>  (last arg is the target file)")
        return "continue"

    filepath = ctx.args[-1]
    task = " ".join(ctx.args[:-1])

    reviewer_p, reviewer_m = get_route(ctx.config, "reviewer")
    reviewer_prov = get_provider(reviewer_p, ctx.config.providers[reviewer_p])
    r_exploration, r_tools = _exploration_for_role(ctx.config, "reviewer")

    content = read_file_safe(Path(ctx.session.repo_root) / filepath)
    prompt = (
        f"You are a strict code reviewer.\n\n"
        f"Task: {task}\n"
        f"File: {filepath}\n\n"
        f"Current content:\n```\n{content}\n```\n\n"
        f"Identify correctness issues relative to the task. Be concise and specific."
    )

    ctx.console.print(f"Reviewing {filepath} with {reviewer_m}...")
    output = reviewer_prov.generate(
        prompt=prompt, model=reviewer_m,
        exploration=r_exploration, allowed_tools=r_tools,
    )
    if isinstance(reviewer_prov.last_usage, UsageInfo):
        ctx.session.usage.record("reviewer", reviewer_prov.last_usage)

    ctx.console.print("\n[bold]Review Output:[/bold]")
    ctx.console.print(output)

    return "continue"


# ---------------------------------------------------------------------------
# /diff
# ---------------------------------------------------------------------------

def handle_diff(ctx: HandlerContext) -> HandlerResult:
    """Re-display last diff."""
    if ctx.session.last_diff is None:
        ctx.console.print("No diff yet.")
    else:
        ctx.console.print(ctx.session.last_diff)
    return "continue"


# ---------------------------------------------------------------------------
# /models
# ---------------------------------------------------------------------------

def handle_models(ctx: HandlerContext) -> HandlerResult:
    """Show planner/drafter/reviewer bindings."""
    for role, info in list_roles().items():
        source_color = {"explicit": "green", "legacy": "yellow", "default": "dim"}.get(
            info["source"], "white"
        )
        ctx.console.print(
            f"[bold]{role:8}[/bold]  [cyan]{info['provider']}[/cyan] / {info['model']}"
            f"  [{source_color}]({info['source']})[/{source_color}]"
        )
    return "continue"


# ---------------------------------------------------------------------------
# /models set
# ---------------------------------------------------------------------------

def handle_models_set(ctx: HandlerContext) -> HandlerResult:
    """Set provider+model for a role. Usage: /models set <role> <model> [--provider P]"""
    # Parse --provider P from args
    args = list(ctx.args)
    provider: str | None = None
    if "--provider" in args:
        idx = args.index("--provider")
        if idx + 1 < len(args):
            provider = args[idx + 1]
            args = args[:idx] + args[idx + 2 :]
        else:
            ctx.console.print("Error: --provider requires a value.")
            return "continue"

    if len(args) < 2:
        ctx.console.print("Usage: /models set <role> <model> [--provider P]")
        return "continue"

    role = args[0]
    model = args[1]

    try:
        set_role_model(role, model, provider=provider)
        provider_display = provider or "(inferred)"
        ctx.console.print(f"Updated {role} -> {provider_display}:{model}")
    except ConfigWriteError as exc:
        ctx.console.print(f"Error: {exc}")
        return "continue"

    return "reload_config"


# ---------------------------------------------------------------------------
# /cache clear
# ---------------------------------------------------------------------------

def handle_cache_clear(ctx: HandlerContext) -> HandlerResult:
    """Wipe plan cache."""
    count = clear_cache()
    ctx.console.print(f"Cleared {count} cached plan{'s' if count != 1 else ''}.")
    return "continue"


# ---------------------------------------------------------------------------
# /doctor
# ---------------------------------------------------------------------------

def handle_doctor(ctx: HandlerContext) -> HandlerResult:
    """Check provider health."""
    all_ok = True
    for p_name, p_conf in ctx.config.providers.items():
        prov = get_provider(p_name, p_conf)
        if prov.healthcheck():
            ctx.console.print(f"[green]OK[/green]  Provider '{p_name}' ({p_conf.type}) is reachable.")
        else:
            ctx.console.print(
                f"[yellow]WARN[/yellow]  Provider '{p_name}' ({p_conf.type}) is unreachable."
            )
            all_ok = False

    if all_ok:
        ctx.console.print("[green]All system checks passed.[/green]")
    else:
        ctx.console.print(
            "Some providers not reachable. Ensure they are running and config URLs are correct."
        )
    return "continue"


# ---------------------------------------------------------------------------
# /resources
# ---------------------------------------------------------------------------

def handle_resources(ctx: HandlerContext) -> HandlerResult:
    """Print CPU/RAM/GPU."""
    cpus = get_cpu_count()
    total_ram = get_total_ram_gb()
    avail_ram = get_available_ram_gb()
    gpu = probe_gpu()

    ctx.console.print("[bold]System Resources[/bold]")
    ctx.console.print(f"CPUs: [magenta]{cpus}[/magenta]")
    ctx.console.print(f"Total RAM: [magenta]{total_ram:.1f} GB[/magenta]")
    ctx.console.print(f"Available RAM: [magenta]{avail_ram:.1f} GB[/magenta]")
    ctx.console.print(
        f"GPU: [magenta]{gpu.gpu_name or 'none'} "
        f"({'%.1f' % gpu.vram_gb} GB VRAM, backend={gpu.backend})[/magenta]"
    )
    return "continue"


# ---------------------------------------------------------------------------
# /config edit
# ---------------------------------------------------------------------------

def handle_config_edit(ctx: HandlerContext) -> HandlerResult:
    """Open config in $EDITOR."""
    editor = os.environ.get("EDITOR", "")
    if not editor:
        ctx.console.print("Set $EDITOR to use /config edit.")
        return "continue"

    path = get_config_path()
    subprocess.call([editor, str(path)])
    return "reload_config"


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------

def register_all(registry) -> None:
    """Register every handler in this module with the shell registry."""
    registry.register("run", handle_run, "Full pipeline: plan -> draft -> QA. Usage: /run <task>")
    registry.register("plan", handle_plan, "Plan only. Usage: /plan <task> OR /plan show")
    registry.register("estimate", handle_estimate, "Predict cost/quota for a task.")
    registry.register("draft", handle_draft, "Single-file draft. Usage: /draft <task> <file>")
    registry.register("review", handle_review, "Single-file review. Usage: /review <task> <file>")
    registry.register("diff", handle_diff, "Re-display last diff.")
    registry.register("models", handle_models, "Show planner/drafter/reviewer bindings.")
    registry.register(
        "models set",
        handle_models_set,
        "Set provider+model for a role. Usage: /models set <role> <model> [--provider P]",
    )
    registry.register("cache clear", handle_cache_clear, "Wipe plan cache.")
    registry.register("doctor", handle_doctor, "Check provider health.")
    registry.register("resources", handle_resources, "Print CPU/RAM/GPU.")
    registry.register("config edit", handle_config_edit, "Open config in $EDITOR.")
