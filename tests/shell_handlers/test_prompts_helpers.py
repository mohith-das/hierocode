"""Tests for hierocode.shell_handlers._prompts."""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock, patch

from rich.console import Console

from hierocode.shell_handlers._prompts import (
    ApplyChoice,
    BatchApplyChoice,
    EscalationChoice,
    prompt_apply_batch,
    prompt_apply_choice,
    prompt_escalation_approval,
)


def _capturing_console() -> Console:
    """Real Rich Console that writes to a StringIO and records, for output inspection."""
    return Console(file=StringIO(), width=80, record=True, force_terminal=False)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROMPT_SESSION_PATH = "hierocode.shell_handlers._prompts.PromptSession"
_CONSOLE_PATH = "hierocode.shell_handlers._prompts._console"


def _patched_prompt(return_value: str):
    """Return a context-manager patch that makes PromptSession().prompt() return return_value."""
    mock_session = MagicMock()
    mock_session.return_value.prompt.return_value = return_value
    return patch(_PROMPT_SESSION_PATH, mock_session)


def _patched_prompt_raises(exc):
    """Return a context-manager patch that makes PromptSession().prompt() raise exc."""
    mock_session = MagicMock()
    mock_session.return_value.prompt.side_effect = exc
    return patch(_PROMPT_SESSION_PATH, mock_session)


# ---------------------------------------------------------------------------
# prompt_apply_choice — choice mapping
# ---------------------------------------------------------------------------


class TestPromptApplyChoiceMapping:
    def test_returns_yes_on_y(self):
        with _patched_prompt("y"), patch(_CONSOLE_PATH):
            result = prompt_apply_choice("src/api.py", 12, 3, "modify")
        assert result == ApplyChoice.YES

    def test_returns_yes_all_on_a(self):
        with _patched_prompt("a"), patch(_CONSOLE_PATH):
            result = prompt_apply_choice("src/api.py", 5, 1, "modify")
        assert result == ApplyChoice.YES_ALL

    def test_returns_skip_on_n(self):
        with _patched_prompt("n"), patch(_CONSOLE_PATH):
            result = prompt_apply_choice("src/api.py", 0, 0, "create")
        assert result == ApplyChoice.SKIP

    def test_returns_abort_on_q(self):
        with _patched_prompt("q"), patch(_CONSOLE_PATH):
            result = prompt_apply_choice("src/api.py", 0, 5, "delete")
        assert result == ApplyChoice.ABORT

    def test_returns_abort_on_keyboard_interrupt(self):
        with _patched_prompt_raises(KeyboardInterrupt), patch(_CONSOLE_PATH):
            result = prompt_apply_choice("src/api.py", 2, 1, "modify")
        assert result == ApplyChoice.ABORT

    def test_returns_abort_on_eof_error(self):
        with _patched_prompt_raises(EOFError), patch(_CONSOLE_PATH):
            result = prompt_apply_choice("src/api.py", 2, 1, "modify")
        assert result == ApplyChoice.ABORT

    def test_returns_skip_on_unknown_input(self):
        with _patched_prompt("z"), patch(_CONSOLE_PATH):
            result = prompt_apply_choice("src/api.py", 3, 0, "modify")
        assert result == ApplyChoice.SKIP

    def test_returns_skip_on_empty_input(self):
        with _patched_prompt(""), patch(_CONSOLE_PATH):
            result = prompt_apply_choice("src/api.py", 0, 0, "modify")
        assert result == ApplyChoice.SKIP

    def test_input_is_case_insensitive(self):
        with _patched_prompt("Y"), patch(_CONSOLE_PATH):
            result = prompt_apply_choice("src/api.py", 1, 0, "modify")
        assert result == ApplyChoice.YES


# ---------------------------------------------------------------------------
# prompt_apply_choice — display content
# ---------------------------------------------------------------------------


class TestPromptApplyChoiceDisplay:
    def test_displays_file_and_line_counts(self):
        """The panel rendered to the console must mention the file path, +N -M, and action."""
        real_console = _capturing_console()
        with _patched_prompt("n"), patch(_CONSOLE_PATH, real_console):
            prompt_apply_choice("src/api.py", 12, 3, "modify")

        rendered = real_console.export_text()
        assert "src/api.py" in rendered
        assert "12" in rendered
        assert "3" in rendered
        assert "modify" in rendered

    def test_displays_all_four_options(self):
        """Panel body must advertise y, a, n, q options."""
        real_console = _capturing_console()
        with _patched_prompt("n"), patch(_CONSOLE_PATH, real_console):
            prompt_apply_choice("foo.py", 0, 0, "create")

        rendered = real_console.export_text()
        for key in ("y", "a", "n", "q"):
            assert key in rendered


# ---------------------------------------------------------------------------
# prompt_escalation_approval — choice mapping
# ---------------------------------------------------------------------------


class TestPromptEscalationApprovalMapping:
    def test_returns_approve_on_y(self):
        with _patched_prompt("y"), patch(_CONSOLE_PATH):
            result = prompt_escalation_approval("u1", "add retry logic", 2, "claude-sonnet-4-6")
        assert result == EscalationChoice.APPROVE

    def test_returns_abort_on_n(self):
        with _patched_prompt("n"), patch(_CONSOLE_PATH):
            result = prompt_escalation_approval("u1", "add retry logic", 2, "claude-sonnet-4-6")
        assert result == EscalationChoice.ABORT

    def test_returns_abort_on_keyboard_interrupt(self):
        with _patched_prompt_raises(KeyboardInterrupt), patch(_CONSOLE_PATH):
            result = prompt_escalation_approval("u1", "add retry logic", 2, "claude-sonnet-4-6")
        assert result == EscalationChoice.ABORT

    def test_returns_abort_on_eof(self):
        with _patched_prompt_raises(EOFError), patch(_CONSOLE_PATH):
            result = prompt_escalation_approval("u1", "goal", 1, "model")
        assert result == EscalationChoice.ABORT

    def test_returns_abort_on_unknown_input(self):
        with _patched_prompt("x"), patch(_CONSOLE_PATH):
            result = prompt_escalation_approval("u1", "goal", 1, "model")
        assert result == EscalationChoice.ABORT


# ---------------------------------------------------------------------------
# prompt_escalation_approval — display content
# ---------------------------------------------------------------------------


class TestPromptEscalationApprovalDisplay:
    def test_mentions_planner_model_and_cost_hint(self):
        """Displayed panel must include the planner_model name and a cost/quota hint."""
        real_console = _capturing_console()
        with _patched_prompt("n"), patch(_CONSOLE_PATH, real_console):
            prompt_escalation_approval("u42", "implement caching", 3, "claude-opus-4-7")

        rendered = real_console.export_text()
        assert "claude-opus-4-7" in rendered
        # Cost/quota hint — the panel body must reference quota, cost, or subscription
        hint_found = any(
            word in rendered.lower() for word in ("quota", "cost", "subscription", "api")
        )
        assert hint_found, f"No cost/quota hint found in rendered panel: {rendered!r}"

    def test_mentions_unit_id_and_goal(self):
        """Panel must echo back the unit id and goal."""
        real_console = _capturing_console()
        with _patched_prompt("n"), patch(_CONSOLE_PATH, real_console):
            prompt_escalation_approval("unit-99", "write the tests", 1, "model-x")

        rendered = real_console.export_text()
        assert "unit-99" in rendered
        assert "write the tests" in rendered


# ---------------------------------------------------------------------------
# Helpers for prompt_apply_batch
# ---------------------------------------------------------------------------


def _make_file_patch(path: str = "src/foo.py", added: int = 5, removed: int = 2):
    """Build a minimal FilePatch-compatible object without importing patcher."""
    from hierocode.broker.patcher import FilePatch, PatchAction

    return FilePatch(
        path=path,
        action=PatchAction.MODIFY,
        line_count_added=added,
        line_count_removed=removed,
        raw_diff="--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new\n",
    )


def _patched_prompt_sequence(*return_values):
    """Patch PromptSession so successive .prompt() calls return each value in order."""
    mock_session = MagicMock()
    mock_session.return_value.prompt.side_effect = list(return_values)
    return patch(_PROMPT_SESSION_PATH, mock_session)


# ---------------------------------------------------------------------------
# prompt_apply_batch — choice mapping
# ---------------------------------------------------------------------------


class TestPromptApplyBatch:
    def test_yes_returns_yes_all_no_sticky(self):
        """'y' then 'n' → YES_ALL with make_sticky=False."""
        patches = [_make_file_patch()]
        with _patched_prompt_sequence("y", "n"), patch(_CONSOLE_PATH):
            result = prompt_apply_batch(patches)
        assert result.choice == BatchApplyChoice.YES_ALL
        assert result.make_sticky is False

    def test_yes_with_sticky(self):
        """'y' then 'y' → YES_ALL with make_sticky=True."""
        patches = [_make_file_patch()]
        with _patched_prompt_sequence("y", "y"), patch(_CONSOLE_PATH):
            result = prompt_apply_batch(patches)
        assert result.choice == BatchApplyChoice.YES_ALL
        assert result.make_sticky is True

    def test_review_returns_review(self):
        """'r' → REVIEW, make_sticky=False (no sticky prompt for review)."""
        patches = [_make_file_patch()]
        with _patched_prompt_sequence("r"), patch(_CONSOLE_PATH):
            result = prompt_apply_batch(patches)
        assert result.choice == BatchApplyChoice.REVIEW
        assert result.make_sticky is False

    def test_no_returns_abort(self):
        """'n' → ABORT."""
        patches = [_make_file_patch()]
        with _patched_prompt_sequence("n"), patch(_CONSOLE_PATH):
            result = prompt_apply_batch(patches)
        assert result.choice == BatchApplyChoice.ABORT

    def test_empty_returns_abort(self):
        """Empty input → ABORT (safer default)."""
        patches = [_make_file_patch()]
        with _patched_prompt_sequence(""), patch(_CONSOLE_PATH):
            result = prompt_apply_batch(patches)
        assert result.choice == BatchApplyChoice.ABORT

    def test_unknown_returns_abort(self):
        """'x' → ABORT."""
        patches = [_make_file_patch()]
        with _patched_prompt_sequence("x"), patch(_CONSOLE_PATH):
            result = prompt_apply_batch(patches)
        assert result.choice == BatchApplyChoice.ABORT

    def test_ctrl_c_returns_abort(self):
        """KeyboardInterrupt → ABORT."""
        patches = [_make_file_patch()]
        with _patched_prompt_raises(KeyboardInterrupt), patch(_CONSOLE_PATH):
            result = prompt_apply_batch(patches)
        assert result.choice == BatchApplyChoice.ABORT
        assert result.make_sticky is False

    def test_eof_returns_abort(self):
        """EOFError → ABORT."""
        patches = [_make_file_patch()]
        with _patched_prompt_raises(EOFError), patch(_CONSOLE_PATH):
            result = prompt_apply_batch(patches)
        assert result.choice == BatchApplyChoice.ABORT


# ---------------------------------------------------------------------------
# prompt_apply_batch — display content
# ---------------------------------------------------------------------------


class TestPromptApplyBatchDisplay:
    def test_renders_all_files(self):
        """Panel must contain each of the 3 file paths."""
        patches = [
            _make_file_patch("src/a.py"),
            _make_file_patch("src/b.py"),
            _make_file_patch("src/c.py"),
        ]
        real_console = _capturing_console()
        with _patched_prompt_sequence("n"), patch(_CONSOLE_PATH, real_console):
            prompt_apply_batch(patches)

        rendered = real_console.export_text()
        assert "src/a.py" in rendered
        assert "src/b.py" in rendered
        assert "src/c.py" in rendered

    def test_renders_line_counts(self):
        """Panel must mention +N -M per file."""
        patches = [_make_file_patch("x.py", added=10, removed=3)]
        real_console = _capturing_console()
        with _patched_prompt_sequence("n"), patch(_CONSOLE_PATH, real_console):
            prompt_apply_batch(patches)

        rendered = real_console.export_text()
        assert "10" in rendered
        assert "3" in rendered
