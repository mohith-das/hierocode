"""Reusable interactive prompt helpers for shell handlers.

These wrap prompt_toolkit shortcuts so individual handlers don't each reinvent
the yes/no/skip/abort UX. Designed so tests can patch the helper rather than
prompt_toolkit internals.
"""

from __future__ import annotations

from enum import Enum

from prompt_toolkit import PromptSession
from rich.console import Console
from rich.panel import Panel


class ApplyChoice(str, Enum):
    """Possible responses to a per-file apply prompt."""

    YES = "yes"
    YES_ALL = "yes_all"  # apply this and don't ask for remaining files this session
    SKIP = "skip"  # skip this file, keep asking for the rest
    ABORT = "abort"  # stop processing all remaining files


class EscalationChoice(str, Enum):
    """Possible responses to an escalation approval prompt."""

    APPROVE = "approve"
    ABORT = "abort"


# Module-level console for rendering panels (tests can replace this).
_console = Console(stderr=True)


def prompt_apply_choice(
    file_path: str,
    added_lines: int,
    removed_lines: int,
    action: str,  # "modify" | "create" | "delete" — matches PatchAction.value
) -> ApplyChoice:
    """Prompt the user to confirm applying a single file patch.

    Renders a Rich panel with the file details then reads a single character via
    prompt_toolkit. Returns ApplyChoice.ABORT on Ctrl-C or EOFError.
    """
    body = (
        f"File: [bold]{file_path}[/bold]  ({action})\n"
        f"[green]+{added_lines}[/green] [red]-{removed_lines}[/red]\n"
        "\n"
        "  [bold]y[/bold]   yes\n"
        "  [bold]a[/bold]   yes, apply all remaining without asking\n"
        "  [bold]n[/bold]   skip\n"
        "  [bold]q[/bold]   abort"
    )
    _console.print(Panel(body, title="Apply patch", border_style="blue"))

    session: PromptSession = PromptSession()
    try:
        raw = session.prompt("> ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return ApplyChoice.ABORT

    if raw == "y":
        return ApplyChoice.YES
    if raw == "a":
        return ApplyChoice.YES_ALL
    if raw == "q":
        return ApplyChoice.ABORT
    # "n", blank, or anything else → safe default
    return ApplyChoice.SKIP


def prompt_escalation_approval(
    unit_id: str,
    unit_goal: str,
    revisions_done: int,
    planner_model: str,
) -> EscalationChoice:
    """Ask whether to escalate a unit to the planner after the drafter has
    exhausted its revision budget.

    Cost note is surfaced: escalation uses the planner tier (counts against
    subscription quota or costs API $). Returns EscalationChoice.ABORT on Ctrl-C.
    """
    body = (
        f"Unit [bold]{unit_id}[/bold] has exhausted its revision budget "
        f"({revisions_done} revision(s) done).\n"
        "\n"
        "Escalating will ask the [bold]planner[/bold] tier to draft directly.\n"
        f"Model: [magenta]{planner_model}[/magenta]  "
        "[dim](counts against your subscription quota or API cost)[/dim]\n"
        "\n"
        f"Goal: {unit_goal}\n"
        "\n"
        "  [bold]y[/bold]   approve escalation\n"
        "  [bold]n[/bold]   abort (mark unit failed)"
    )
    _console.print(Panel(body, title="Escalation approval", border_style="yellow"))

    session: PromptSession = PromptSession()
    try:
        raw = session.prompt("> ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return EscalationChoice.ABORT

    if raw == "y":
        return EscalationChoice.APPROVE
    return EscalationChoice.ABORT
