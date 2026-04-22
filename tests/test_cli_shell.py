"""Tests for hierocode.cli_shell — REPL framework (W21)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


from hierocode.cli_shell import (
    Handler,
    HandlerContext,
    HandlerRegistry,
    HandlerResult,
    SessionState,
    run_shell,
)
from hierocode.models.schemas import HierocodeConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config() -> HierocodeConfig:
    """Return a minimal HierocodeConfig."""
    return HierocodeConfig()


def _make_session(tmp_path: Path, mode: str = "prompt") -> SessionState:
    return SessionState(repo_root=tmp_path, interaction_mode=mode)  # type: ignore[arg-type]


def _make_console() -> MagicMock:
    c = MagicMock()
    c._hiero_registry = None
    return c


def _exit_handler(ctx: HandlerContext) -> HandlerResult:
    return "exit"


def _continue_handler(ctx: HandlerContext) -> HandlerResult:
    return "continue"


def _make_run_shell_kwargs(
    inputs: list[str],
    tmp_path: Path,
    extra_register: list[tuple[str, Handler, str]] | None = None,
    mode: str = "prompt",
    history_path: Path | None = None,
):
    """Build everything needed to call run_shell with a mock PromptSession."""
    config = _make_config()
    registry = HandlerRegistry()
    console = _make_console()
    if extra_register:
        for cmd, handler, help_text in extra_register:
            registry.register(cmd, handler, help_text)

    if history_path is None:
        history_path = tmp_path / "repl_history"

    return config, registry, console, history_path, inputs


# ---------------------------------------------------------------------------
# HandlerRegistry unit tests
# ---------------------------------------------------------------------------


class TestHandlerRegistryRegisterAndResolveSingleWord:
    def test_registry_register_and_resolve_single_word(self):
        """A single-word command registers and resolves correctly."""
        reg = HandlerRegistry()
        handler = _continue_handler
        reg.register("run", handler, "Run a task.")
        resolved, remaining = reg.resolve(["run", "do", "something"])
        assert resolved is handler
        assert remaining == ["do", "something"]

    def test_registry_help_text_stored(self):
        """Help text is stored and retrievable."""
        reg = HandlerRegistry()
        reg.register("run", _continue_handler, "Run it.")
        assert reg.get_help("run") == "Run it."

    def test_registry_set_help_updates_text(self):
        """set_help can update the help string after initial registration."""
        reg = HandlerRegistry()
        reg.register("run", _continue_handler, "old help")
        reg.set_help("run", "new help")
        assert reg.get_help("run") == "new help"

    def test_registry_commands_sorted_with_slash(self):
        """commands() returns sorted slash-prefixed strings."""
        reg = HandlerRegistry()
        reg.register("zzz", _continue_handler)
        reg.register("aaa", _continue_handler)
        cmds = reg.commands()
        assert cmds == ["/aaa", "/zzz"]


class TestRegistryResolvesMultiWordLongestFirst:
    def test_registry_resolves_multi_word_longest_first(self):
        """Multi-word lookup picks the longest matching prefix."""
        reg = HandlerRegistry()
        short_handler = MagicMock(return_value="continue")
        long_handler = MagicMock(return_value="continue")
        reg.register("models", short_handler)
        reg.register("models set", long_handler)

        resolved, remaining = reg.resolve(["models", "set", "x"])
        assert resolved is long_handler
        assert remaining == ["x"]

    def test_registry_falls_back_to_shorter_match(self):
        """When the longer key doesn't match, the shorter one is used."""
        reg = HandlerRegistry()
        short_handler = MagicMock(return_value="continue")
        reg.register("models", short_handler)
        reg.register("models set", MagicMock(return_value="continue"))

        resolved, remaining = reg.resolve(["models", "list"])
        assert resolved is short_handler
        assert remaining == ["list"]


class TestRegistryUnknownReturnsNone:
    def test_registry_unknown_returns_none(self):
        """Unregistered command returns (None, original_tokens)."""
        reg = HandlerRegistry()
        handler, remaining = reg.resolve(["unknown"])
        assert handler is None
        assert remaining == ["unknown"]

    def test_registry_empty_tokens_returns_none(self):
        """Empty token list returns (None, [])."""
        reg = HandlerRegistry()
        handler, remaining = reg.resolve([])
        assert handler is None
        assert remaining == []


# ---------------------------------------------------------------------------
# run_shell integration tests (mocking PromptSession)
# ---------------------------------------------------------------------------


def _make_pt_session(inputs: list[str | type]) -> MagicMock:
    """Build a mock PromptSession whose .prompt() yields inputs in order.

    A value of KeyboardInterrupt (the class) causes a raise on that call.
    A value of EOFError (the class) causes a raise on that call.
    """
    mock_session = MagicMock()
    side_effects: list = []
    for item in inputs:
        if item is KeyboardInterrupt:
            side_effects.append(KeyboardInterrupt())
        elif item is EOFError:
            side_effects.append(EOFError())
        else:
            side_effects.append(item)
    mock_session.prompt.side_effect = side_effects
    return mock_session


class TestExitReturnsCleanly:
    def test_exit_returns_cleanly(self, tmp_path):
        """The REPL exits without error when /exit is typed."""
        config, registry, console, history_path, inputs = _make_run_shell_kwargs(
            ["/exit"], tmp_path
        )
        pt_mock = _make_pt_session(inputs)
        with patch("hierocode.cli_shell.PromptSession", return_value=pt_mock):
            run_shell(config, registry, console=console, history_path=history_path)
        # Verify prompt was called at least once.
        assert pt_mock.prompt.call_count >= 1


class TestShlelxUnbalancedQuotesReprompts:
    def test_shlex_unbalanced_quotes_reprompts(self, tmp_path):
        """/run with unbalanced quotes prints error and continues; /exit terminates."""
        config, registry, console, history_path, inputs = _make_run_shell_kwargs(
            ['/run "unclosed', "/exit"], tmp_path
        )
        pt_mock = _make_pt_session(inputs)
        with patch("hierocode.cli_shell.PromptSession", return_value=pt_mock):
            run_shell(config, registry, console=console, history_path=history_path)
        # console.print should have been called with a parse error message.
        print_calls = [str(c) for c in console.print.call_args_list]
        assert any("Parse error" in c or "parse error" in c.lower() for c in print_calls)


class TestDispatchHandlerExceptionKeepsLoop:
    def test_dispatch_handler_exception_keeps_loop(self, tmp_path):
        """A handler that raises does not crash the REPL; loop exits on /exit."""
        def boom_handler(ctx: HandlerContext) -> HandlerResult:
            raise ValueError("explosion")

        config, registry, console, history_path, _ = _make_run_shell_kwargs([], tmp_path)
        registry.register("boom", boom_handler)

        pt_mock = _make_pt_session(["/boom", "/exit"])
        with patch("hierocode.cli_shell.PromptSession", return_value=pt_mock):
            run_shell(config, registry, console=console, history_path=history_path)

        # print_exception must have been called.
        console.print_exception.assert_called_once()


class TestReloadConfigCallsReload:
    def test_reload_config_calls_reload(self, tmp_path):
        """A handler returning 'reload_config' triggers the reload callable."""
        reload_mock = MagicMock(return_value=_make_config())

        def reload_handler(ctx: HandlerContext) -> HandlerResult:
            return "reload_config"

        config, registry, console, history_path, _ = _make_run_shell_kwargs([], tmp_path)
        registry.register("rld", reload_handler)

        pt_mock = _make_pt_session(["/rld", "/exit"])
        with patch("hierocode.cli_shell.PromptSession", return_value=pt_mock):
            with patch("hierocode.config.load_config", reload_mock):
                run_shell(config, registry, console=console, history_path=history_path)

        reload_mock.assert_called()


class TestPlainTextImmediateModeDispatchesRun:
    def test_plain_text_immediate_mode_dispatches_run_direct(self, tmp_path):
        """In immediate mode (patched via tui config attribute), /run is called."""
        run_mock = MagicMock(return_value="continue")

        config = _make_config()
        # Attach a tui object with interaction_mode="immediate".
        tui = MagicMock()
        tui.interaction_mode = "immediate"
        config.tui = tui  # type: ignore[attr-defined]

        registry = HandlerRegistry()
        registry.register("run", run_mock)
        console = _make_console()
        history_path = tmp_path / "history"

        pt_mock = _make_pt_session(["do something cool", "/exit"])
        with patch("hierocode.cli_shell.PromptSession", return_value=pt_mock):
            run_shell(config, registry, console=console, history_path=history_path)

        run_mock.assert_called_once()
        ctx_arg: HandlerContext = run_mock.call_args[0][0]
        assert ctx_arg.args == ["do something cool"]


class TestPlainTextPromptModeYesDispatches:
    def test_plain_text_prompt_mode_yes_dispatches(self, tmp_path):
        """In prompt mode, answering 'y' dispatches to /run."""
        run_mock = MagicMock(return_value="continue")
        config, registry, console, history_path, _ = _make_run_shell_kwargs([], tmp_path)
        registry.register("run", run_mock)

        pt_mock = _make_pt_session(["plain input text", "/exit"])
        with patch("hierocode.cli_shell.PromptSession", return_value=pt_mock):
            with patch("builtins.input", return_value="y"):
                run_shell(config, registry, console=console, history_path=history_path)

        run_mock.assert_called_once()
        ctx_arg: HandlerContext = run_mock.call_args[0][0]
        assert ctx_arg.args == ["plain input text"]


class TestPlainTextPromptModeNoSkips:
    def test_plain_text_prompt_mode_no_skips(self, tmp_path):
        """In prompt mode, answering 'n' skips dispatch to /run."""
        run_mock = MagicMock(return_value="continue")
        config, registry, console, history_path, _ = _make_run_shell_kwargs([], tmp_path)
        registry.register("run", run_mock)

        pt_mock = _make_pt_session(["plain input text", "/exit"])
        with patch("hierocode.cli_shell.PromptSession", return_value=pt_mock):
            with patch("builtins.input", return_value="n"):
                run_shell(config, registry, console=console, history_path=history_path)

        run_mock.assert_not_called()


class TestEmptyInputNoop:
    def test_empty_input_noop(self, tmp_path):
        """Empty lines do not dispatch or error; REPL continues to next prompt."""
        config, registry, console, history_path, _ = _make_run_shell_kwargs([], tmp_path)
        pt_mock = _make_pt_session(["", "   ", "/exit"])
        with patch("hierocode.cli_shell.PromptSession", return_value=pt_mock):
            run_shell(config, registry, console=console, history_path=history_path)
        # No print calls from dispatch errors.
        assert pt_mock.prompt.call_count == 3


class TestCtrlCDoesntExit:
    def test_ctrl_c_doesnt_exit(self, tmp_path):
        """KeyboardInterrupt on prompt cancels the line; REPL keeps going."""
        config, registry, console, history_path, _ = _make_run_shell_kwargs([], tmp_path)
        pt_mock = _make_pt_session([KeyboardInterrupt, KeyboardInterrupt, "/exit"])
        with patch("hierocode.cli_shell.PromptSession", return_value=pt_mock):
            run_shell(config, registry, console=console, history_path=history_path)
        assert pt_mock.prompt.call_count == 3


class TestCtrlDExits:
    def test_ctrl_d_exits(self, tmp_path):
        """EOFError (Ctrl-D) exits the REPL cleanly."""
        config, registry, console, history_path, _ = _make_run_shell_kwargs([], tmp_path)
        pt_mock = _make_pt_session([EOFError])
        with patch("hierocode.cli_shell.PromptSession", return_value=pt_mock):
            run_shell(config, registry, console=console, history_path=history_path)
        assert pt_mock.prompt.call_count == 1


# ---------------------------------------------------------------------------
# Meta-handler tests
# ---------------------------------------------------------------------------


class TestHelpNoArgsListsCommands:
    def test_help_no_args_lists_commands(self, tmp_path):
        """/help with no arguments prints the registered command list."""
        config, registry, console, history_path, _ = _make_run_shell_kwargs([], tmp_path)
        pt_mock = _make_pt_session(["/help", "/exit"])
        with patch("hierocode.cli_shell.PromptSession", return_value=pt_mock):
            run_shell(config, registry, console=console, history_path=history_path)
        # At least one console.print call should include "/exit" or "/help".
        print_calls = " ".join(str(c) for c in console.print.call_args_list)
        assert "/help" in print_calls or "/exit" in print_calls


class TestHelpWithCommandShowsHelpText:
    def test_help_with_command_shows_help_text(self, tmp_path):
        """/help <cmd> prints the help text for that command."""
        config, registry, console, history_path, _ = _make_run_shell_kwargs([], tmp_path)
        registry.register("mycommand", _continue_handler, "Does something useful.")

        pt_mock = _make_pt_session(["/help mycommand", "/exit"])
        with patch("hierocode.cli_shell.PromptSession", return_value=pt_mock):
            run_shell(config, registry, console=console, history_path=history_path)

        print_calls = " ".join(str(c) for c in console.print.call_args_list)
        assert "Does something useful" in print_calls


class TestClearCallsConsoleClear:
    def test_clear_calls_console_clear(self, tmp_path):
        """/clear calls console.clear()."""
        config, registry, console, history_path, _ = _make_run_shell_kwargs([], tmp_path)
        pt_mock = _make_pt_session(["/clear", "/exit"])
        with patch("hierocode.cli_shell.PromptSession", return_value=pt_mock):
            run_shell(config, registry, console=console, history_path=history_path)
        console.clear.assert_called_once()


class TestRepoWithNoArgsPrintsCurrent:
    def test_repo_with_no_args_prints_current(self, tmp_path):
        """/repo with no args prints the current repo_root."""
        config, registry, console, history_path, _ = _make_run_shell_kwargs([], tmp_path)
        pt_mock = _make_pt_session(["/repo", "/exit"])
        with patch("hierocode.cli_shell.PromptSession", return_value=pt_mock):
            run_shell(config, registry, console=console, history_path=history_path)
        print_calls = " ".join(str(c) for c in console.print.call_args_list)
        # The printed text should contain the resolved cwd path.
        assert str(Path(".").resolve()) in print_calls


class TestRepoWithInvalidPathErrors:
    def test_repo_with_invalid_path_errors(self, tmp_path):
        """/repo <nonexistent_path> prints an error."""
        config, registry, console, history_path, _ = _make_run_shell_kwargs([], tmp_path)
        pt_mock = _make_pt_session(["/repo /does/not/exist/at/all", "/exit"])
        with patch("hierocode.cli_shell.PromptSession", return_value=pt_mock):
            run_shell(config, registry, console=console, history_path=history_path)
        print_calls = " ".join(str(c) for c in console.print.call_args_list)
        assert "does not exist" in print_calls.lower() or "not exist" in print_calls.lower()


class TestRepoWithValidPathUpdatesSession:
    def test_repo_with_valid_path_updates_session(self, tmp_path):
        """/repo <valid_dir> updates session.repo_root."""
        new_dir = tmp_path / "myrepo"
        new_dir.mkdir()

        config, registry, console, history_path, _ = _make_run_shell_kwargs([], tmp_path)

        captured_session: list[SessionState] = []

        def capture_handler(ctx: HandlerContext) -> HandlerResult:
            captured_session.append(ctx.session)
            return "exit"

        registry.register("peek", capture_handler)
        pt_mock = _make_pt_session([f"/repo {new_dir}", "/peek"])
        with patch("hierocode.cli_shell.PromptSession", return_value=pt_mock):
            run_shell(config, registry, console=console, history_path=history_path)

        assert len(captured_session) == 1
        assert captured_session[0].repo_root == new_dir.resolve()


class TestHistoryPrintsTaskHistory:
    def test_history_prints_task_history(self, tmp_path):
        """/history prints the entries in session.task_history."""
        config, registry, console, history_path, _ = _make_run_shell_kwargs([], tmp_path)

        # Inject history via a custom handler that mutates session.
        def add_history(ctx: HandlerContext) -> HandlerResult:
            ctx.session.task_history.append("task one")
            ctx.session.task_history.append("task two")
            return "continue"

        registry.register("addhist", add_history)
        pt_mock = _make_pt_session(["/addhist", "/history", "/exit"])
        with patch("hierocode.cli_shell.PromptSession", return_value=pt_mock):
            run_shell(config, registry, console=console, history_path=history_path)

        print_calls = " ".join(str(c) for c in console.print.call_args_list)
        assert "task one" in print_calls
        assert "task two" in print_calls


class TestUnknownCommandPrintsMessage:
    def test_unknown_command_prints_message(self, tmp_path):
        """Unknown slash command prints a helpful error, loop continues."""
        config, registry, console, history_path, _ = _make_run_shell_kwargs([], tmp_path)
        pt_mock = _make_pt_session(["/notacommand", "/exit"])
        with patch("hierocode.cli_shell.PromptSession", return_value=pt_mock):
            run_shell(config, registry, console=console, history_path=history_path)
        print_calls = " ".join(str(c) for c in console.print.call_args_list)
        assert "Unknown command" in print_calls


class TestQuitAliasWorks:
    def test_quit_alias_works(self, tmp_path):
        """/quit exits the REPL just like /exit."""
        config, registry, console, history_path, _ = _make_run_shell_kwargs([], tmp_path)
        pt_mock = _make_pt_session(["/quit"])
        with patch("hierocode.cli_shell.PromptSession", return_value=pt_mock):
            run_shell(config, registry, console=console, history_path=history_path)
        assert pt_mock.prompt.call_count == 1


class TestQuestionMarkAliasForHelp:
    def test_question_mark_alias_for_help(self, tmp_path):
        """'/?' is an alias for /help and doesn't crash."""
        config, registry, console, history_path, _ = _make_run_shell_kwargs([], tmp_path)
        pt_mock = _make_pt_session(["/?", "/exit"])
        with patch("hierocode.cli_shell.PromptSession", return_value=pt_mock):
            run_shell(config, registry, console=console, history_path=history_path)
        # Just assert it didn't crash and continued.
        assert pt_mock.prompt.call_count == 2
