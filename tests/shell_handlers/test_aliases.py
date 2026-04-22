"""Tests for hierocode.shell_handlers.aliases."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch


from hierocode.broker.aliases import AliasError, TaskAlias
from hierocode.shell_handlers.aliases import handle_task, register_all


# ---------------------------------------------------------------------------
# Fake HandlerContext
# ---------------------------------------------------------------------------


@dataclass
class _FakeSessionState:
    repo_root: Path = field(default_factory=lambda: Path("/tmp/repo"))
    interaction_mode: str = "prompt"
    last_plan: object = None
    last_diff: object = None
    task_history: list = field(default_factory=list)


def _make_ctx(args: list[str]):
    """Build a real HandlerContext dataclass so dataclasses.replace works in handlers."""
    from hierocode.cli_shell import HandlerContext

    return HandlerContext(
        args=args,
        session=_FakeSessionState(),
        config=MagicMock(),
        console=MagicMock(),
        reload_config=MagicMock(),
    )


# ---------------------------------------------------------------------------
# Patch target helpers
# ---------------------------------------------------------------------------

_MOD = "hierocode.shell_handlers.aliases"


# ---------------------------------------------------------------------------
# /task (no args)
# ---------------------------------------------------------------------------


def test_no_args_prints_usage():
    ctx = _make_ctx([])
    result = handle_task(ctx)
    ctx.console.print.assert_called_once()
    printed = ctx.console.print.call_args[0][0]
    assert "Usage" in printed
    assert result == "continue"


# ---------------------------------------------------------------------------
# /task list
# ---------------------------------------------------------------------------


def test_list_empty():
    ctx = _make_ctx(["list"])
    with patch(f"{_MOD}.list_aliases", return_value=[]):
        result = handle_task(ctx)
    ctx.console.print.assert_called_once()
    printed = ctx.console.print.call_args[0][0]
    assert "No task aliases" in printed
    assert result == "continue"


def test_list_multiple_aliases():
    aliases = [
        TaskAlias(name="build", description="compile"),
        TaskAlias(name="test", description="run tests"),
    ]
    ctx = _make_ctx(["list"])
    with patch(f"{_MOD}.list_aliases", return_value=aliases):
        result = handle_task(ctx)
    assert ctx.console.print.call_count == 2
    assert result == "continue"


# ---------------------------------------------------------------------------
# /task save
# ---------------------------------------------------------------------------


def test_save_valid():
    ctx = _make_ctx(["save", "lint", "run", "linter"])
    with patch(f"{_MOD}.save_alias", return_value=TaskAlias(name="lint", description="run linter")):
        result = handle_task(ctx)
    ctx.console.print.assert_called_once()
    printed = ctx.console.print.call_args[0][0]
    assert "Saved" in printed
    assert result == "continue"


def test_save_too_few_args():
    ctx = _make_ctx(["save", "lint"])  # missing description
    result = handle_task(ctx)
    ctx.console.print.assert_called_once()
    printed = ctx.console.print.call_args[0][0]
    assert "Usage" in printed
    assert result == "continue"


def test_save_alias_error_printed():
    ctx = _make_ctx(["save", "bad name", "desc"])
    with patch(f"{_MOD}.save_alias", side_effect=AliasError("bad name")):
        result = handle_task(ctx)
    ctx.console.print.assert_called_once()
    printed = ctx.console.print.call_args[0][0]
    assert "bad name" in printed
    assert result == "continue"


# ---------------------------------------------------------------------------
# /task delete
# ---------------------------------------------------------------------------


def test_delete_existing():
    ctx = _make_ctx(["delete", "old-task"])
    with patch(f"{_MOD}.delete_alias", return_value=True):
        result = handle_task(ctx)
    ctx.console.print.assert_called_once()
    printed = ctx.console.print.call_args[0][0]
    assert "Deleted" in printed
    assert result == "continue"


def test_delete_missing_prints_hint():
    ctx = _make_ctx(["delete", "ghost"])
    with patch(f"{_MOD}.delete_alias", return_value=False):
        result = handle_task(ctx)
    ctx.console.print.assert_called_once()
    printed = ctx.console.print.call_args[0][0]
    assert "No such alias" in printed
    assert result == "continue"


# ---------------------------------------------------------------------------
# /task <name> — alias resolution
# ---------------------------------------------------------------------------


def test_unknown_alias_prints_hint():
    ctx = _make_ctx(["unknown-alias"])
    with patch(f"{_MOD}.get_alias", return_value=None):
        result = handle_task(ctx)
    ctx.console.print.assert_called_once()
    printed = ctx.console.print.call_args[0][0]
    assert "Unknown task alias" in printed
    assert result == "continue"


def test_known_alias_dispatches_run():
    """When a valid alias is resolved, handle_run is called with shlex-split args."""
    alias = TaskAlias(name="ci", description="run tests --fast")
    ctx = _make_ctx(["ci"])

    fake_result: str = "continue"
    mock_handle_run = MagicMock(return_value=fake_result)

    with (
        patch(f"{_MOD}.get_alias", return_value=alias),
        # The lazy import inside handle_task pulls from broker_cmds at call time.
        patch("hierocode.shell_handlers.broker_cmds.handle_run", mock_handle_run),
    ):
        result = handle_task(ctx)

    assert result == fake_result
    mock_handle_run.assert_called_once()
    called_ctx = mock_handle_run.call_args[0][0]
    assert called_ctx.args == ["run", "tests", "--fast"]


# ---------------------------------------------------------------------------
# register_all
# ---------------------------------------------------------------------------


def test_register_all_registers_task():
    """register_all calls registry.register with 'task' as the command name."""
    registry = MagicMock()
    register_all(registry)
    registry.register.assert_called_once()
    call_args = registry.register.call_args[0]
    assert call_args[0] == "task"
