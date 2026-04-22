"""Shell handler for the /apply command (W23)."""

from __future__ import annotations

from hierocode.broker.patcher import PatchParseError, apply_patch, parse_diff
from hierocode.cli_shell import HandlerContext, HandlerResult


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

    applied = 0
    skipped = 0
    errors = 0

    for p in patches:
        choice = _confirm(ctx.console, p)
        if choice == "q":
            ctx.console.print("[yellow]Aborted remaining patches.[/yellow]")
            break
        if choice in ("n", "s"):
            skipped += 1
            continue
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
    """Prompt the user for per-file apply confirmation.

    Returns one of 'y', 'n', 's' (skip), or 'q' (quit all remaining).
    Default on blank/unknown input is 'n'.
    """
    try:
        raw = input(f"Apply to {patch.path}? [y/N/s(kip)/q(uit)]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "n"
    if raw in ("y",):
        return "y"
    if raw in ("s", "skip"):
        return "s"
    if raw in ("q", "quit"):
        return "q"
    return "n"


def register_all(registry) -> None:
    """Register the /apply handler."""
    registry.register(
        "apply",
        handle_apply,
        "Apply the last diff to disk with per-file confirmation.",
    )
