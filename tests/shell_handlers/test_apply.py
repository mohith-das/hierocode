"""Tests for hierocode.shell_handlers.apply."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from hierocode.broker.patcher import ApplyResult, FilePatch, PatchAction, PatchParseError
from hierocode.cli_shell import HandlerContext, HandlerRegistry, SessionState
from hierocode.models.schemas import HierocodeConfig, PolicyConfig
from hierocode.shell_handlers._prompts import ApplyChoice, BatchApplyChoice, BatchApplyResult
from hierocode.shell_handlers.apply import _apply_per_file, _confirm, handle_apply, register_all

# Patch targets
_PROMPT_APPLY = "hierocode.shell_handlers.apply.prompt_apply_choice"
_PROMPT_BATCH = "hierocode.shell_handlers.apply.prompt_apply_batch"
_APPLY_PATCH = "hierocode.shell_handlers.apply.apply_patch"
_PARSE_DIFF = "hierocode.shell_handlers.apply.parse_diff"


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
    config: HierocodeConfig | None = None,
) -> HandlerContext:
    """Build a HandlerContext with optional last_diff and mock console."""
    if console is None:
        console = MagicMock()
    if config is None:
        config = HierocodeConfig()
    return HandlerContext(
        args=[],
        session=_make_session(tmp_path, last_diff),
        config=config,
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
# handle_apply — guard tests
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

        with patch(_PARSE_DIFF, side_effect=PatchParseError("bad diff")):
            result = handle_apply(ctx)

        assert result == "continue"
        print_calls = " ".join(str(c) for c in console.print.call_args_list)
        assert "bad diff" in print_calls


class TestApplyEmptyParsedDiff:
    def test_apply_empty_parsed_diff_prints_empty_message(self, tmp_path):
        """When parse_diff returns [], an 'empty' message is shown."""
        console = MagicMock()
        ctx = _make_ctx(tmp_path, last_diff="   \n", console=console)

        with patch(_PARSE_DIFF, return_value=[]):
            result = handle_apply(ctx)

        assert result == "continue"
        print_calls = " ".join(str(c) for c in console.print.call_args_list)
        assert "empty" in print_calls.lower()


# ---------------------------------------------------------------------------
# handle_apply — auto-apply gates
# ---------------------------------------------------------------------------


class TestApplyAutoSessionStickySkipsPrompts:
    def test_session_sticky_skips_batch_prompt(self, tmp_path):
        """session.auto_apply_session=True → prompt_apply_batch NEVER called; all patches applied."""
        patches = [_make_patch("a.py"), _make_patch("b.py")]
        apply_ok = ApplyResult(path="x", status="applied")
        console = MagicMock()
        ctx = _make_ctx(tmp_path, last_diff="dummy", console=console)
        ctx.session.auto_apply_session = True

        with patch(_PARSE_DIFF, return_value=patches):
            with patch(_APPLY_PATCH, return_value=apply_ok) as mock_apply:
                with patch(_PROMPT_BATCH) as mock_batch:
                    result = handle_apply(ctx)

        assert result == "continue"
        mock_batch.assert_not_called()
        assert mock_apply.call_count == 2


class TestApplyPolicyAutoApplySkipsPrompts:
    def test_policy_auto_apply_skips_batch_prompt(self, tmp_path):
        """config.policy.auto_apply=True → prompt_apply_batch NEVER called."""
        patches = [_make_patch("a.py")]
        apply_ok = ApplyResult(path="a.py", status="applied")
        console = MagicMock()
        config = HierocodeConfig(policy=PolicyConfig(auto_apply=True))
        ctx = _make_ctx(tmp_path, last_diff="dummy", console=console, config=config)

        with patch(_PARSE_DIFF, return_value=patches):
            with patch(_APPLY_PATCH, return_value=apply_ok) as mock_apply:
                with patch(_PROMPT_BATCH) as mock_batch:
                    result = handle_apply(ctx)

        assert result == "continue"
        mock_batch.assert_not_called()
        assert mock_apply.call_count == 1


# ---------------------------------------------------------------------------
# handle_apply — batch prompt path
# ---------------------------------------------------------------------------


class TestApplyBatchYesAllNoSticky:
    def test_patches_applied_session_flag_stays_false(self, tmp_path):
        """Batch YES_ALL no-sticky → patches applied, session.auto_apply_session stays False."""
        patches = [_make_patch("a.py"), _make_patch("b.py")]
        apply_ok = ApplyResult(path="x", status="applied")
        console = MagicMock()
        ctx = _make_ctx(tmp_path, last_diff="dummy", console=console)

        batch_result = BatchApplyResult(choice=BatchApplyChoice.YES_ALL, make_sticky=False)

        with patch(_PARSE_DIFF, return_value=patches):
            with patch(_APPLY_PATCH, return_value=apply_ok) as mock_apply:
                with patch(_PROMPT_BATCH, return_value=batch_result):
                    result = handle_apply(ctx)

        assert result == "continue"
        assert mock_apply.call_count == 2
        assert ctx.session.auto_apply_session is False


class TestApplyBatchYesAllWithSticky:
    def test_patches_applied_session_flag_set_true(self, tmp_path):
        """Batch YES_ALL sticky=True → patches applied AND session.auto_apply_session=True."""
        patches = [_make_patch("a.py")]
        apply_ok = ApplyResult(path="a.py", status="applied")
        console = MagicMock()
        ctx = _make_ctx(tmp_path, last_diff="dummy", console=console)

        batch_result = BatchApplyResult(choice=BatchApplyChoice.YES_ALL, make_sticky=True)

        with patch(_PARSE_DIFF, return_value=patches):
            with patch(_APPLY_PATCH, return_value=apply_ok):
                with patch(_PROMPT_BATCH, return_value=batch_result):
                    result = handle_apply(ctx)

        assert result == "continue"
        assert ctx.session.auto_apply_session is True
        print_calls = " ".join(str(c) for c in console.print.call_args_list)
        assert "Auto-apply enabled" in print_calls


class TestApplyBatchReviewDropsIntoPerFile:
    def test_review_calls_per_file_prompt(self, tmp_path):
        """Batch REVIEW → prompt_apply_choice called once per patch."""
        patches = [_make_patch("a.py"), _make_patch("b.py")]
        apply_ok = ApplyResult(path="x", status="applied")
        console = MagicMock()
        ctx = _make_ctx(tmp_path, last_diff="dummy", console=console)

        batch_result = BatchApplyResult(choice=BatchApplyChoice.REVIEW)

        with patch(_PARSE_DIFF, return_value=patches):
            with patch(_APPLY_PATCH, return_value=apply_ok):
                with patch(_PROMPT_BATCH, return_value=batch_result):
                    with patch(_PROMPT_APPLY, return_value=ApplyChoice.YES) as mock_per_file:
                        result = handle_apply(ctx)

        assert result == "continue"
        assert mock_per_file.call_count == 2


class TestApplyBatchAbortWritesNothing:
    def test_abort_does_not_call_apply_patch(self, tmp_path):
        """Batch ABORT → apply_patch NEVER called."""
        patches = [_make_patch("a.py")]
        console = MagicMock()
        ctx = _make_ctx(tmp_path, last_diff="dummy", console=console)

        batch_result = BatchApplyResult(choice=BatchApplyChoice.ABORT)

        with patch(_PARSE_DIFF, return_value=patches):
            with patch(_APPLY_PATCH) as mock_apply:
                with patch(_PROMPT_BATCH, return_value=batch_result):
                    result = handle_apply(ctx)

        assert result == "continue"
        mock_apply.assert_not_called()
        print_calls = " ".join(str(c) for c in console.print.call_args_list)
        assert "bort" in print_calls.lower() or "Aborted" in print_calls


# ---------------------------------------------------------------------------
# _apply_per_file — REVIEW branch regression tests
# ---------------------------------------------------------------------------


class TestApplyPerFileWalksThroughEachFile:
    def test_apply_per_file_yes_applies_all(self, tmp_path):
        """With 2 patches and YES for each, apply_patch is called twice."""
        patches = [_make_patch("a.py"), _make_patch("b.py")]
        apply_ok = ApplyResult(path="x", status="applied")
        console = MagicMock()
        ctx = _make_ctx(tmp_path, last_diff="dummy", console=console)

        with patch(_APPLY_PATCH, return_value=apply_ok) as mock_apply:
            with patch(_PROMPT_APPLY, return_value=ApplyChoice.YES):
                _apply_per_file(ctx, patches)

        assert mock_apply.call_count == 2


class TestApplyPerFileSkipPerFile:
    def test_skip_does_not_call_apply_patch(self, tmp_path):
        """SKIP leaves apply_patch uncalled."""
        patches = [_make_patch("skip_me.py")]
        console = MagicMock()
        ctx = _make_ctx(tmp_path, last_diff="dummy", console=console)

        with patch(_APPLY_PATCH) as mock_apply:
            with patch(_PROMPT_APPLY, return_value=ApplyChoice.SKIP):
                _apply_per_file(ctx, patches)

        mock_apply.assert_not_called()


class TestApplyPerFileQuitAbortsRemaining:
    def test_abort_on_second_patch_leaves_third_untouched(self, tmp_path):
        """ABORT on the second patch leaves the third patch untouched."""
        patches = [_make_patch("a.py"), _make_patch("b.py"), _make_patch("c.py")]
        apply_ok = ApplyResult(path="a.py", status="applied")
        console = MagicMock()
        ctx = _make_ctx(tmp_path, last_diff="dummy", console=console)

        choices = iter([ApplyChoice.YES, ApplyChoice.ABORT, ApplyChoice.YES])

        with patch(_APPLY_PATCH, return_value=apply_ok) as mock_apply:
            with patch(_PROMPT_APPLY, side_effect=choices):
                _apply_per_file(ctx, patches)

        assert mock_apply.call_count == 1
        print_calls = " ".join(str(c) for c in console.print.call_args_list)
        assert "bort" in print_calls.lower() or "Aborted" in print_calls


class TestApplyPerFileYesAllSkipsSubsequentPrompts:
    def test_yes_all_prompt_called_once_all_applied(self, tmp_path):
        """YES_ALL on first patch: prompt_apply_choice called once, all 3 patches applied."""
        patches = [_make_patch("a.py"), _make_patch("b.py"), _make_patch("c.py")]
        apply_ok = ApplyResult(path="x", status="applied")
        console = MagicMock()
        ctx = _make_ctx(tmp_path, last_diff="dummy", console=console)

        with patch(_APPLY_PATCH, return_value=apply_ok) as mock_apply:
            with patch(_PROMPT_APPLY, return_value=ApplyChoice.YES_ALL) as mock_prompt:
                _apply_per_file(ctx, patches)

        assert mock_prompt.call_count == 1
        assert mock_apply.call_count == 3


class TestApplyPerFileSummaryPrinted:
    def test_summary_shows_applied_skipped_errors(self, tmp_path):
        """Summary line shows applied/skipped/errors."""
        patches = [_make_patch("x.py"), _make_patch("y.py")]
        apply_ok = ApplyResult(path="x.py", status="applied")
        apply_err = ApplyResult(path="y.py", status="error", message="oops")
        console = MagicMock()
        ctx = _make_ctx(tmp_path, last_diff="dummy", console=console)

        choices = iter([ApplyChoice.YES, ApplyChoice.YES])
        apply_results = iter([apply_ok, apply_err])

        with patch(_APPLY_PATCH, side_effect=apply_results):
            with patch(_PROMPT_APPLY, side_effect=choices):
                _apply_per_file(ctx, patches)

        print_calls = " ".join(str(c) for c in console.print.call_args_list)
        assert "Summary" in print_calls
        assert "applied=1" in print_calls
        assert "errors=1" in print_calls


class TestApplyPerFileReviewRespectSkipAndAbort:
    def test_via_review_branch_skip_and_abort(self, tmp_path):
        """Per-file flow reached via REVIEW branch handles skip and abort correctly."""
        patches = [_make_patch("a.py"), _make_patch("b.py"), _make_patch("c.py")]
        apply_ok = ApplyResult(path="a.py", status="applied")
        console = MagicMock()
        ctx = _make_ctx(tmp_path, last_diff="dummy", console=console)

        batch_result = BatchApplyResult(choice=BatchApplyChoice.REVIEW)
        # a.py → skip, b.py → abort, c.py never reached
        per_file_choices = iter([ApplyChoice.SKIP, ApplyChoice.ABORT])

        with patch(_PARSE_DIFF, return_value=patches):
            with patch(_APPLY_PATCH, return_value=apply_ok) as mock_apply:
                with patch(_PROMPT_BATCH, return_value=batch_result):
                    with patch(_PROMPT_APPLY, side_effect=per_file_choices):
                        handle_apply(ctx)

        mock_apply.assert_not_called()


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
# _confirm unit tests (legacy wrapper)
# ---------------------------------------------------------------------------


class TestConfirmFunction:
    _PROMPT_PATH = "hierocode.shell_handlers.apply.prompt_apply_choice"

    def _fake_patch_obj(self, path: str = "test.py") -> FilePatch:
        return _make_patch(path)

    def test_confirm_y_returns_y(self):
        p = self._fake_patch_obj()
        with patch(self._PROMPT_PATH, return_value=ApplyChoice.YES):
            assert _confirm(MagicMock(), p) == "y"

    def test_confirm_yes_all_returns_y(self):
        """YES_ALL maps to 'y' in the legacy adapter."""
        p = self._fake_patch_obj()
        with patch(self._PROMPT_PATH, return_value=ApplyChoice.YES_ALL):
            assert _confirm(MagicMock(), p) == "y"

    def test_confirm_skip_returns_n(self):
        p = self._fake_patch_obj()
        with patch(self._PROMPT_PATH, return_value=ApplyChoice.SKIP):
            assert _confirm(MagicMock(), p) == "n"

    def test_confirm_abort_returns_q(self):
        p = self._fake_patch_obj()
        with patch(self._PROMPT_PATH, return_value=ApplyChoice.ABORT):
            assert _confirm(MagicMock(), p) == "q"
