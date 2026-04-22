"""Shell handler for the /apply command (W23)."""

from __future__ import annotations

from hierocode.broker.patcher import PatchParseError, apply_patch, parse_diff
from hierocode.cli_shell import HandlerContext, HandlerResult
from hierocode.shell_handlers._prompts import ApplyChoice, prompt_apply_choice


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

    ctx.console.print(f"\n[bold]Last diff touches {len(patches)} file(s):[/bold]")
    for p in patches:
        ctx.console.print(
            f"  - {p.path}  ({p.action.value}, "
            f"+{p.line_count_added} / -{p.line_count_removed})"
        )

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

        # YES or YES_ALL: apply the patch.
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
    return "continue"


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
        "Apply the last diff to disk with per-file confirmation.",
    )
