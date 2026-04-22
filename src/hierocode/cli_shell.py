"""REPL framework for the hierocode TUI (v0.3).

Other workers hand their handlers to a HandlerRegistry; run_shell loops forever
until /exit or Ctrl-D.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from rich.console import Console

from hierocode.broker.plan_schema import Plan
from hierocode.broker.usage import UsageAccumulator
from hierocode.models.schemas import HierocodeConfig

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

HandlerResult = Literal["continue", "reload_config", "exit"]
Handler = Callable[["HandlerContext"], HandlerResult]


@dataclass
class SessionState:
    """Mutable per-session state threaded through every handler call."""

    repo_root: Path
    interaction_mode: Literal["prompt", "immediate"] = "prompt"
    last_plan: Optional[Plan] = None
    last_diff: Optional[str] = None  # raw unified diff string
    task_history: list[str] = field(default_factory=list)
    usage: UsageAccumulator = field(default_factory=UsageAccumulator)


@dataclass
class HandlerContext:
    """Immutable (by convention) bundle passed to every handler."""

    args: list[str]
    session: SessionState
    config: HierocodeConfig
    console: Console
    reload_config: Callable[[], HierocodeConfig]


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------


class HandlerRegistry:
    """Maps slash command strings to handler callables.

    Supports multi-word commands via longest-match (e.g. 'models set').
    """

    def __init__(self) -> None:
        # key: tuple of command words  value: (handler, help_text)
        self._table: dict[tuple[str, ...], tuple[Handler, str]] = {}

    def register(self, command: str, handler: Handler, help_text: str = "") -> None:
        """Register command (may contain spaces) with an optional help string."""
        key = tuple(command.split())
        self._table[key] = (handler, help_text)

    def resolve(self, tokens: list[str]) -> tuple[Optional[Handler], list[str]]:
        """Return (handler, remaining_args). None if no match.

        Tries longest prefix first so 'models set' beats 'models'.
        """
        for length in range(len(tokens), 0, -1):
            key = tuple(tokens[:length])
            if key in self._table:
                handler, _ = self._table[key]
                return handler, list(tokens[length:])
        return None, tokens

    def commands(self) -> list[str]:
        """Sorted list of registered command strings, each prefixed with /."""
        return sorted("/" + " ".join(key) for key in self._table)

    def set_help(self, command: str, text: str) -> None:
        """Update the help text for an already-registered command."""
        key = tuple(command.split())
        if key in self._table:
            handler, _ = self._table[key]
            self._table[key] = (handler, text)

    def get_help(self, command: str) -> Optional[str]:
        """Return help text for command, or None if not registered."""
        key = tuple(command.split())
        entry = self._table.get(key)
        return entry[1] if entry is not None else None


# ---------------------------------------------------------------------------
# Meta handlers (owned by W21)
# ---------------------------------------------------------------------------


def _handle_help(ctx: HandlerContext) -> HandlerResult:
    """Print all commands or the help text for a specific command."""
    if ctx.args:
        # Strip leading slash if user typed "/help /models"
        name = ctx.args[0].lstrip("/")
        # We stash the registry on the console as a side-channel (set by run_shell).
        registry: Optional[HandlerRegistry] = getattr(ctx.console, "_hiero_registry", None)
        if registry is not None:
            text = registry.get_help(name)
            if text:
                ctx.console.print(f"[bold]/{name}[/bold]  {text}")
            else:
                ctx.console.print(f"No help text for /{name}.")
        else:
            ctx.console.print(f"No help text for /{name}.")
    else:
        registry = getattr(ctx.console, "_hiero_registry", None)
        if registry is not None:
            cmds = registry.commands()
            if cmds:
                for cmd in cmds:
                    help_text = registry.get_help(cmd.lstrip("/")) or ""
                    if help_text:
                        ctx.console.print(f"  {cmd:<24}  {help_text}")
                    else:
                        ctx.console.print(f"  {cmd}")
            else:
                ctx.console.print("No commands registered.")
        else:
            ctx.console.print("No commands registered.")
    return "continue"


def _handle_exit(ctx: HandlerContext) -> HandlerResult:
    """Exit the REPL."""
    return "exit"


def _handle_clear(ctx: HandlerContext) -> HandlerResult:
    """Clear the terminal screen."""
    ctx.console.clear()
    return "continue"


def _handle_history(ctx: HandlerContext) -> HandlerResult:
    """Print the task history for this session."""
    if not ctx.session.task_history:
        ctx.console.print("[dim](no history yet)[/dim]")
    else:
        for i, entry in enumerate(ctx.session.task_history, 1):
            ctx.console.print(f"  {i}. {entry}")
    return "continue"


def _handle_repo(ctx: HandlerContext) -> HandlerResult:
    """Show or update the current repo root."""
    if not ctx.args:
        ctx.console.print(str(ctx.session.repo_root))
        return "continue"
    candidate = Path(ctx.args[0]).expanduser().resolve()
    if not candidate.exists():
        ctx.console.print(f"[red]Path does not exist:[/red] {candidate}")
        return "continue"
    if not candidate.is_dir():
        ctx.console.print(f"[red]Path is not a directory:[/red] {candidate}")
        return "continue"
    ctx.session.repo_root = candidate
    ctx.console.print(f"[green]repo_root updated to[/green] {candidate}")
    return "continue"


def _register_meta_handlers(registry: HandlerRegistry) -> None:
    """Register the built-in meta commands owned by W21."""
    registry.register("help", _handle_help, "Show all commands or help for a specific command.")
    registry.register("?", _handle_help, "Alias for /help.")
    registry.register("exit", _handle_exit, "Exit the hierocode REPL.")
    registry.register("quit", _handle_exit, "Exit the hierocode REPL.")
    registry.register("clear", _handle_clear, "Clear the terminal screen.")
    registry.register("history", _handle_history, "Show task history for this session.")
    registry.register("repo", _handle_repo, "Show or update the current repo root.")


# ---------------------------------------------------------------------------
# Plain-text confirmation helper
# ---------------------------------------------------------------------------


def _confirm_run(text: str, console: Console) -> Literal["y", "n", "e"]:
    """Prompt the user for a run-confirmation choice.

    Returns one of 'y', 'n', or 'e' (edit).
    """
    try:
        answer = input("Run as task? [y/N/e(dit)]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if answer in ("y",):
        return "y"
    if answer in ("e",):
        return "e"
    return "n"


# ---------------------------------------------------------------------------
# History path resolution
# ---------------------------------------------------------------------------


def _default_history_path() -> Path:
    """Return the persistent REPL history file, respecting XDG_CACHE_HOME."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    p = base / "hierocode" / "repl_history"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_shell(
    config: HierocodeConfig,
    registry: HandlerRegistry,
    console: Optional[Console] = None,
    history_path: Optional[Path] = None,
) -> None:
    """Start the REPL. Blocks until /exit or Ctrl-D."""
    if console is None:
        console = Console()

    if history_path is None:
        history_path = _default_history_path()

    # Register the meta commands defined in this file.
    _register_meta_handlers(registry)

    # Stash registry on the console so _handle_help can access it.
    console._hiero_registry = registry  # type: ignore[attr-defined]

    # Build session state from config defaults.
    interaction_mode: Literal["prompt", "immediate"] = "prompt"
    tui_cfg = getattr(config, "tui", None)
    if tui_cfg is not None:
        interaction_mode = getattr(tui_cfg, "interaction_mode", "prompt")

    session_state = SessionState(
        repo_root=Path(".").resolve(),
        interaction_mode=interaction_mode,
    )

    def _reload_config() -> HierocodeConfig:
        """Re-read ~/.hierocode.yaml and return fresh config."""
        from hierocode.config import load_config  # noqa: PLC0415

        return load_config()

    # Rebuild the tab-completer lazily before each prompt so newly registered
    # commands are picked up without restarting.
    def _make_completer() -> WordCompleter:
        return WordCompleter(registry.commands(), sentence=True)

    hist = FileHistory(str(history_path))
    pt_session: PromptSession = PromptSession(history=hist)

    current_config = config

    while True:
        # Refresh completer to capture any newly registered commands.
        completer = _make_completer()
        try:
            raw: str = pt_session.prompt(">  ", completer=completer)
        except KeyboardInterrupt:
            # Ctrl-C cancels the current line; keep looping.
            continue
        except EOFError:
            # Ctrl-D — exit cleanly.
            break

        raw = raw.strip()
        if not raw:
            continue

        # Split the raw input.
        try:
            tokens = shlex.split(raw, posix=True)
        except ValueError as exc:
            console.print(f"[red]Parse error:[/red] {exc}")
            continue

        if not tokens:
            continue

        first = tokens[0]

        if first.startswith("/"):
            # Slash command path.
            cmd_tokens = [first[1:]] + tokens[1:]
            handler, remaining = registry.resolve(cmd_tokens)
            if handler is None:
                console.print(f"Unknown command: {first}. Try /help.")
                continue
            ctx = HandlerContext(
                args=remaining,
                session=session_state,
                config=current_config,
                console=console,
                reload_config=_reload_config,
            )
            try:
                result: HandlerResult = handler(ctx)
            except Exception:
                console.print_exception(show_locals=False)
                continue
            if result == "exit":
                break
            if result == "reload_config":
                current_config = _reload_config()
            # "continue" → keep looping
        else:
            # Plain-text path.
            if session_state.interaction_mode == "immediate":
                run_handler, _ = registry.resolve(["run"])
                if run_handler is None:
                    console.print("Unknown input. Register a /run handler to enable immediate mode.")
                    continue
                ctx = HandlerContext(
                    args=[raw],
                    session=session_state,
                    config=current_config,
                    console=console,
                    reload_config=_reload_config,
                )
                try:
                    result = run_handler(ctx)
                except Exception:
                    console.print_exception(show_locals=False)
                    continue
                if result == "exit":
                    break
                if result == "reload_config":
                    current_config = _reload_config()
            else:
                # "prompt" mode — ask the user.
                choice = _confirm_run(raw, console)
                if choice == "n":
                    continue
                if choice == "e":
                    try:
                        raw = input("Edit: ").strip() or raw  # noqa: F841
                    except (EOFError, KeyboardInterrupt):
                        continue
                # y or edited text → dispatch /run
                run_handler, _ = registry.resolve(["run"])
                if run_handler is None:
                    console.print("Unknown input. Register a /run handler to use this feature.")
                    continue
                ctx = HandlerContext(
                    args=[raw],
                    session=session_state,
                    config=current_config,
                    console=console,
                    reload_config=_reload_config,
                )
                try:
                    result = run_handler(ctx)
                except Exception:
                    console.print_exception(show_locals=False)
                    continue
                if result == "exit":
                    break
                if result == "reload_config":
                    current_config = _reload_config()
