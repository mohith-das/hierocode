"""Shell handlers for /task — named task alias management."""

from __future__ import annotations

import shlex
from dataclasses import replace

from hierocode.broker.aliases import AliasError, delete_alias, get_alias, list_aliases, save_alias
from hierocode.cli_shell import HandlerContext, HandlerResult


# ---------------------------------------------------------------------------
# /task dispatcher
# ---------------------------------------------------------------------------


def handle_task(ctx: HandlerContext) -> HandlerResult:
    """Dispatches sub-commands: /task save | list | delete | <name>."""
    if not ctx.args:
        ctx.console.print("Usage: /task [save|list|delete|<name>] ...")
        return "continue"

    sub = ctx.args[0]

    # ------------------------------------------------------------------
    # /task list
    # ------------------------------------------------------------------
    if sub == "list":
        aliases = list_aliases()
        if not aliases:
            ctx.console.print("No task aliases saved.")
            return "continue"
        for a in aliases:
            ctx.console.print(f"  [cyan]{a.name}[/cyan]: {a.description}")
        return "continue"

    # ------------------------------------------------------------------
    # /task save <name> <description...>
    # ------------------------------------------------------------------
    if sub == "save":
        if len(ctx.args) < 3:
            ctx.console.print("Usage: /task save <name> <description...>")
            return "continue"
        name = ctx.args[1]
        description = " ".join(ctx.args[2:])
        try:
            save_alias(name, description)
            ctx.console.print(f"[green]Saved:[/green] /task {name}")
        except AliasError as e:
            ctx.console.print(f"[red]{e}[/red]")
        return "continue"

    # ------------------------------------------------------------------
    # /task delete <name>
    # ------------------------------------------------------------------
    if sub == "delete":
        if len(ctx.args) != 2:
            ctx.console.print("Usage: /task delete <name>")
            return "continue"
        if delete_alias(ctx.args[1]):
            ctx.console.print(f"[green]Deleted:[/green] {ctx.args[1]}")
        else:
            ctx.console.print(f"[yellow]No such alias:[/yellow] {ctx.args[1]}")
        return "continue"

    # ------------------------------------------------------------------
    # /task <name>  →  resolve alias → dispatch /run
    # ------------------------------------------------------------------
    alias = get_alias(sub)
    if alias is None:
        ctx.console.print(f"Unknown task alias: {sub}. Try /task list.")
        return "continue"

    from hierocode.shell_handlers.broker_cmds import handle_run  # avoid circular at module level

    run_args = shlex.split(alias.description)
    new_ctx = replace(ctx, args=run_args)
    return handle_run(new_ctx)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_all(registry) -> None:
    """Register /task handler into *registry*."""
    registry.register("task", handle_task,
                      "Named task aliases. Subcommands: save / list / delete / <name>.")
