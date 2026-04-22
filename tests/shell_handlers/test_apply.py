"""Tests for hierocode.shell_handlers.apply."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


from hierocode.broker.patcher import ApplyResult, FilePatch, PatchAction, PatchParseError
from hierocode.cli_shell import HandlerContext, HandlerRegistry, SessionState
from hierocode.models.schemas import HierocodeConfig
from hierocode.shell_handlers.apply import _confirm, handle_apply, register_all


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_session(tmp_path: Path, last_diff: str | None = None) -> SessionState:
    """Build a minimal SessionState."""
    s = SessionState(repo_root=tmp_path)
    s.last_diff = last_diff
    return s


def _make_ctx(
    tmp_path: Path,
    last_diff: str | None = None,
    console: MagicMock | None = None,
) -> HandlerContext:
    """Build a HandlerContext with optional last_diff and mock console."""
    if console is None:
        console = MagicMock()
    return HandlerContext(
        args=[],
        session=_make_session(tmp_path, last_diff),
        config=HierocodeConfig(),
        console=console,
        reload_config=lambda: HierocodeConfig(),
    )


def _make_patch(path: str = "foo.py", action: PatchAction = PatchAction.MODIFY) -> FilePatch:
    return FilePatch(
        path=path,
        action=action,
        line_count_added=2,
        line_count_removed=1,
        raw_diff="--- foo.py\n+++ foo.py\n@@ -1 +1 @@\n-old\n+new\n",
    )


# ---------------------------------------------------------------------------
# handle_apply tests
# ---------------------------------------------------------------------------


class TestApplyWithNoDiff:
    def test_apply_with_no_diff_prints_message(self, tmp_path):
        """When session.last_diff is None, a helpful message is printed."""
        console = MagicMock()
        ctx = _make_ctx(tmp_path, last_diff=None, console=console)

        result = handle_apply(ctx)

        assert result == "continue"
        print_calls = " ".join(str(c) for c in console.print.call_args_list)
        assert "No diff" in print_calls or "no diff" in print_calls.lower()


class TestApplyWithParseError:
    def test_apply_with_parse_error_prints_error_returns_continue(self, tmp_path):
        """A PatchParseError is caught and printed; handler returns 'continue'."""
        console = MagicMock()
        ctx = _make_ctx(tmp_path, last_diff="not a real diff\n", console=console)

        with patch(
            "hierocode.shell_handlers.apply.parse_diff",
            side_effect=PatchParseError("bad diff"),
        ):
            result = handle_apply(ctx)

        assert result == "continue"
        print_calls = " ".join(str(c) for c in console.print.call_args_list)
        assert "bad diff" in print_calls


class TestApplyEmptyParsedDiff:
    def test_apply_empty_parsed_diff_prints_empty_message(self, tmp_path):
        """When parse_diff returns [], an 'empty' message is shown."""
        console = MagicMock()
        ctx = _make_ctx(tmp_path, last_diff="   \n", console=console)

        with patch("hierocode.shell_handlers.apply.parse_diff", return_value=[]):
            result = handle_apply(ctx)

        assert result == "continue"
        print_calls = " ".join(str(c) for c in console.print.call_args_list)
        assert "empty" in print_calls.lower()


class TestApplyWalksThroughEachFile:
    def test_apply_walks_through_each_file(self, tmp_path):
        """With 2 patches and 'y' for each, apply_patch is called twice and both applied."""
        patches = [_make_patch("a.py"), _make_patch("b.py")]
        apply_ok = ApplyResult(path="x", status="applied")
        console = MagicMock()
        ctx = _make_ctx(tmp_path, last_diff="dummy", console=console)

        with patch("hierocode.shell_handlers.apply.parse_diff", return_value=patches):
            with patch(
                "hierocode.shell_handlers.apply.apply_patch", return_value=apply_ok
            ) as mock_apply:
                with patch(
                    "hierocode.shell_handlers.apply._confirm", return_value="y"
                ):
                    result = handle_apply(ctx)

        assert result == "continue"
        assert mock_apply.call_count == 2


class TestApplySkipPerFile:
    def test_apply_skip_per_file(self, tmp_path):
        """When _confirm returns 'n', apply_patch is NOT called for that patch."""
        patches = [_make_patch("skip_me.py")]
        console = MagicMock()
        ctx = _make_ctx(tmp_path, last_diff="dummy", console=console)

        with patch("hierocode.shell_handlers.apply.parse_diff", return_value=patches):
            with patch(
                "hierocode.shell_handlers.apply.apply_patch"
            ) as mock_apply:
                with patch(
                    "hierocode.shell_handlers.apply._confirm", return_value="n"
                ):
                    result = handle_apply(ctx)

        assert result == "continue"
        mock_apply.assert_not_called()

    def test_apply_skip_via_s_response(self, tmp_path):
        """When _confirm returns 's', apply_patch is NOT called."""
        patches = [_make_patch("also_skip.py")]
        console = MagicMock()
        ctx = _make_ctx(tmp_path, last_diff="dummy", console=console)

        with patch("hierocode.shell_handlers.apply.parse_diff", return_value=patches):
            with patch(
                "hierocode.shell_handlers.apply.apply_patch"
            ) as mock_apply:
                with patch(
                    "hierocode.shell_handlers.apply._confirm", return_value="s"
                ):
                    handle_apply(ctx)

        mock_apply.assert_not_called()


class TestApplyQuitAbortsRemaining:
    def test_apply_quit_aborts_remaining(self, tmp_path):
        """'q' on the second patch leaves the third patch untouched."""
        patches = [_make_patch("a.py"), _make_patch("b.py"), _make_patch("c.py")]
        apply_ok = ApplyResult(path="a.py", status="applied")
        console = MagicMock()
        ctx = _make_ctx(tmp_path, last_diff="dummy", console=console)

        confirm_responses = ["y", "q", "y"]
        confirm_iter = iter(confirm_responses)

        with patch("hierocode.shell_handlers.apply.parse_diff", return_value=patches):
            with patch(
                "hierocode.shell_handlers.apply.apply_patch", return_value=apply_ok
            ) as mock_apply:
                with patch(
                    "hierocode.shell_handlers.apply._confirm",
                    side_effect=confirm_iter,
                ):
                    result = handle_apply(ctx)

        assert result == "continue"
        # Only the first patch was applied; 'q' stopped before b.py and c.py.
        assert mock_apply.call_count == 1
        print_calls = " ".join(str(c) for c in console.print.call_args_list)
        assert "bort" in print_calls.lower() or "Aborted" in print_calls


class TestApplySummaryPrinted:
    def test_apply_summary_printed(self, tmp_path):
        """Summary line is always printed at the end showing applied/skipped/errors."""
        patches = [_make_patch("x.py"), _make_patch("y.py")]
        apply_ok = ApplyResult(path="x.py", status="applied")
        apply_err = ApplyResult(path="y.py", status="error", message="oops")
        console = MagicMock()
        ctx = _make_ctx(tmp_path, last_diff="dummy", console=console)

        responses = ["y", "y"]
        resp_iter = iter(responses)
        apply_results = [apply_ok, apply_err]
        apply_iter = iter(apply_results)

        with patch("hierocode.shell_handlers.apply.parse_diff", return_value=patches):
            with patch(
                "hierocode.shell_handlers.apply.apply_patch",
                side_effect=apply_iter,
            ):
                with patch(
                    "hierocode.shell_handlers.apply._confirm",
                    side_effect=resp_iter,
                ):
                    handle_apply(ctx)

        print_calls = " ".join(str(c) for c in console.print.call_args_list)
        assert "Summary" in print_calls
        assert "applied=1" in print_calls
        assert "errors=1" in print_calls


class TestApplyErrorFromApplyPatch:
    def test_apply_error_from_apply_patch_printed(self, tmp_path):
        """When apply_patch returns status='error', the error message is printed."""
        patches = [_make_patch("bad.py")]
        apply_err = ApplyResult(path="bad.py", status="error", message="git exploded")
        console = MagicMock()
        ctx = _make_ctx(tmp_path, last_diff="dummy", console=console)

        with patch("hierocode.shell_handlers.apply.parse_diff", return_value=patches):
            with patch(
                "hierocode.shell_handlers.apply.apply_patch", return_value=apply_err
            ):
                with patch(
                    "hierocode.shell_handlers.apply._confirm", return_value="y"
                ):
                    handle_apply(ctx)

        print_calls = " ".join(str(c) for c in console.print.call_args_list)
        assert "git exploded" in print_calls


# ---------------------------------------------------------------------------
# register_all test
# ---------------------------------------------------------------------------


class TestRegisterAll:
    def test_register_all_registers_apply(self):
        """register_all adds 'apply' to the registry."""
        reg = HandlerRegistry()
        register_all(reg)
        handler, remaining = reg.resolve(["apply"])
        assert handler is not None


# ---------------------------------------------------------------------------
# _confirm unit tests (using patched input())
# ---------------------------------------------------------------------------


class TestConfirmFunction:
    def _fake_patch(self, path: str = "test.py") -> FilePatch:
        return _make_patch(path)

    def test_confirm_y_returns_y(self):
        patch_obj = self._fake_patch()
        with patch("builtins.input", return_value="y"):
            assert _confirm(MagicMock(), patch_obj) == "y"

    def test_confirm_blank_returns_n(self):
        patch_obj = self._fake_patch()
        with patch("builtins.input", return_value=""):
            assert _confirm(MagicMock(), patch_obj) == "n"

    def test_confirm_s_returns_s(self):
        patch_obj = self._fake_patch()
        with patch("builtins.input", return_value="s"):
            assert _confirm(MagicMock(), patch_obj) == "s"

    def test_confirm_q_returns_q(self):
        patch_obj = self._fake_patch()
        with patch("builtins.input", return_value="q"):
            assert _confirm(MagicMock(), patch_obj) == "q"

    def test_confirm_eof_returns_n(self):
        patch_obj = self._fake_patch()
        with patch("builtins.input", side_effect=EOFError):
            assert _confirm(MagicMock(), patch_obj) == "n"

    def test_confirm_keyboard_interrupt_returns_n(self):
        patch_obj = self._fake_patch()
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            assert _confirm(MagicMock(), patch_obj) == "n"
