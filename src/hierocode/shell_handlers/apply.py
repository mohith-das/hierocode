"""Shell handler for the /apply command."""

from __future__ import annotations

from hierocode.broker.patcher import PatchParseError, apply_patch, parse_diff
from hierocode.cli_shell import HandlerContext, HandlerResult
from hierocode.shell_handlers._prompts import (
    ApplyChoice,
    BatchApplyChoice,
    prompt_apply_batch,
    prompt_apply_choice,
)


def handle_apply(ctx: HandlerContext) -> HandlerResult:
    """Parse the last diff and write confirmed files to disk."""
    if not ctx.session.last_diff:
        ctx.console.print("No diff to apply. Run /run or /draft first.")
        return "continue"

    try:
        patches = parse_diff(ctx.session.last_diff)
    except PatchParseError as exc:
        ctx.console.print(f"[red]Diff parse error:[/red] {exc}")
        return "continue"

    if not patches:
        ctx.console.print("Diff is empty.")
        return "continue"

    # Auto-apply gate: session-sticky OR policy-level
    session_sticky = getattr(ctx.session, "auto_apply_session", False)
    policy_auto = getattr(ctx.config.policy, "auto_apply", False)

    if session_sticky or policy_auto:
        _apply_all(ctx, patches, reason="auto-apply")
        return "continue"

    # Default path: batch prompt
    result = prompt_apply_batch(patches)
    if result.choice is BatchApplyChoice.ABORT:
        ctx.console.print("[yellow]Aborted.[/yellow]")
        return "continue"
    if result.choice is BatchApplyChoice.YES_ALL:
        if result.make_sticky:
            ctx.session.auto_apply_session = True
            ctx.console.print("[dim]Auto-apply enabled for this session.[/dim]")
        _apply_all(ctx, patches, reason="batch yes")
        return "continue"
    # REVIEW → per-file fallback (today's flow)
    _apply_per_file(ctx, patches)
    return "continue"


def _apply_all(ctx: HandlerContext, patches: list, reason: str) -> None:
    """Apply every patch without prompting. Print a summary."""
    applied = errors = 0
    for p in patches:
        r = apply_patch(p, ctx.session.repo_root)
        if r.status == "applied":
            ctx.console.print(f"  [green]wrote[/green] {p.path}")
            applied += 1
        else:
            ctx.console.print(f"  [red]error[/red] {p.path}: {r.message}")
            errors += 1
    ctx.console.print(
        f"\n[bold]Summary ({reason}):[/bold] applied={applied} errors={errors}"
    )


def _apply_per_file(ctx: HandlerContext, patches: list) -> None:
    """Per-file confirmation loop — reused unchanged for the REVIEW branch."""
    skip_confirmation = False
    applied = 0
    skipped = 0
    errors = 0

    for p in patches:
        if skip_confirmation:
            choice = ApplyChoice.YES
        else:
            choice = prompt_apply_choice(
                p.path, p.line_count_added, p.line_count_removed, p.action.value
            )

        if choice == ApplyChoice.ABORT:
            ctx.console.print("[yellow]Aborted.[/yellow]")
            break

        if choice == ApplyChoice.SKIP:
            skipped += 1
            continue

        if choice == ApplyChoice.YES_ALL:
            skip_confirmation = True

        result = apply_patch(p, ctx.session.repo_root)
        if result.status == "applied":
            ctx.console.print(f"  [green]wrote[/green] {p.path}")
            applied += 1
        else:
            ctx.console.print(f"  [red]error[/red] {p.path}: {result.message}")
            errors += 1

    ctx.console.print(
        f"\n[bold]Summary:[/bold] applied={applied} skipped={skipped} errors={errors}"
    )


def _confirm(console, patch) -> str:
    """Legacy thin wrapper kept for backward compatibility.

    Delegates to prompt_apply_choice and maps ApplyChoice back to the old
    single-character codes used by pre-W27 tests that still patch this function.
    New code should call prompt_apply_choice directly.
    """
    choice = prompt_apply_choice(
        patch.path, patch.line_count_added, patch.line_count_removed, patch.action.value
    )
    if choice == ApplyChoice.YES:
        return "y"
    if choice == ApplyChoice.YES_ALL:
        return "y"
    if choice == ApplyChoice.ABORT:
        return "q"
    return "n"  # SKIP


def register_all(registry) -> None:
    """Register the /apply handler."""
    registry.register(
        "apply",
        handle_apply,
        "Apply the last diff to disk — batch prompt, then write confirmed files.",
    )
