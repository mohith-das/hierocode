"""Tests for hierocode.broker.progress."""

from __future__ import annotations

import io
from unittest.mock import MagicMock

from rich.console import Console

from hierocode.broker.progress import (
    NULL_REPORTER,
    UnitPhase,
    ProgressState,
    _build_panel,
    make_panel_renderer,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _render(panel) -> str:
    """Render a Rich renderable to a plain string (no ANSI)."""
    buf = io.StringIO()
    console = Console(file=buf, highlight=False, markup=False, no_color=True, width=120)
    console.print(panel)
    return buf.getvalue()


def _fake_usage(
    planner_in: int = 0,
    planner_out: int = 0,
    planner_model: str = "",
    drafter_in: int = 0,
    drafter_out: int = 0,
    drafter_model: str = "",
    reviewer_in: int = 0,
    reviewer_out: int = 0,
    total_messages: int = 0,
):
    """Return a mock UsageAccumulator with the given token counts."""
    acc = MagicMock()
    acc.planner.calls = 1 if (planner_in or planner_out) else 0
    acc.planner.input_tokens = planner_in
    acc.planner.output_tokens = planner_out
    acc.planner.model = planner_model

    acc.drafter.calls = 1 if (drafter_in or drafter_out) else 0
    acc.drafter.input_tokens = drafter_in
    acc.drafter.output_tokens = drafter_out
    acc.drafter.model = drafter_model

    acc.reviewer.calls = 1 if (reviewer_in or reviewer_out) else 0
    acc.reviewer.input_tokens = reviewer_in
    acc.reviewer.output_tokens = reviewer_out
    acc.reviewer.model = ""

    acc.total_messages.return_value = total_messages
    return acc


# ---------------------------------------------------------------------------
# UnitPhase glyph + color
# ---------------------------------------------------------------------------

def test_unit_phase_glyph_and_color():
    """Each phase maps to an expected glyph and a non-empty color string."""
    from hierocode.broker.progress import _PHASE_GLYPH, _PHASE_COLOR

    for phase in UnitPhase:
        assert phase in _PHASE_GLYPH, f"Missing glyph for {phase}"
        assert phase in _PHASE_COLOR, f"Missing color for {phase}"

    # Spot-check a few specific values
    assert _PHASE_GLYPH[UnitPhase.QUEUED] == "○"
    assert _PHASE_GLYPH[UnitPhase.FAILED] == "✗"
    assert _PHASE_GLYPH[UnitPhase.COMPLETED] == "●"

    assert _PHASE_COLOR[UnitPhase.COMPLETED] == "green"
    assert _PHASE_COLOR[UnitPhase.DRAFTING] == "cyan"
    assert _PHASE_COLOR[UnitPhase.REVISING] == "yellow"
    assert _PHASE_COLOR[UnitPhase.FAILED] == "red"
    assert _PHASE_COLOR[UnitPhase.QUEUED] == "dim"
    assert _PHASE_COLOR[UnitPhase.ESCALATING] == "magenta"


# ---------------------------------------------------------------------------
# ProgressState mutations
# ---------------------------------------------------------------------------

def test_progress_state_add_and_set_phase():
    """add_unit appends; set_phase changes the right unit only."""
    state = ProgressState()
    state.add_unit("u1", "first goal")
    state.add_unit("u2", "second goal")

    assert len(state.units) == 2
    assert state.units[0].phase == UnitPhase.QUEUED
    assert state.units[1].phase == UnitPhase.QUEUED

    state.set_phase("u1", UnitPhase.DRAFTING)
    assert state.units[0].phase == UnitPhase.DRAFTING
    assert state.units[1].phase == UnitPhase.QUEUED  # unchanged


def test_progress_state_bump_revision():
    """bump_revision increments the correct unit's counter."""
    state = ProgressState()
    state.add_unit("u1", "goal")
    state.add_unit("u2", "goal2")

    state.bump_revision("u1")
    state.bump_revision("u1")

    assert state.units[0].revision_count == 2
    assert state.units[1].revision_count == 0


# ---------------------------------------------------------------------------
# NullReporter
# ---------------------------------------------------------------------------

def test_null_reporter_all_methods_noop():
    """All 5 protocol methods on NULL_REPORTER are callable without exception."""
    NULL_REPORTER.seed("task", [("u1", "goal")])
    NULL_REPORTER.enqueue("u2", "another goal")
    NULL_REPORTER.phase("u1", UnitPhase.DRAFTING)
    NULL_REPORTER.revision("u1")
    NULL_REPORTER.finished()


# ---------------------------------------------------------------------------
# Panel renders (pure renderer tests — no Live terminal needed)
# ---------------------------------------------------------------------------

def test_panel_renders_with_no_usage():
    """Panel with 1 queued unit and no usage acc renders without error, contains unit id."""
    state = ProgressState(task="test task")
    state.add_unit("u1", "do something")
    panel = _build_panel(state, usage_accumulator=None)
    text = _render(panel)
    assert "u1" in text
    assert "queued" in text


def test_panel_renders_with_usage():
    """Panel with planner usage shows 'planner', token counts, and model name."""
    state = ProgressState(task="add retry")
    state.add_unit("u1", "add tenacity import")
    usage = _fake_usage(
        planner_in=3412, planner_out=380, planner_model="claude-sonnet-4-6"
    )
    panel = _build_panel(state, usage_accumulator=usage)
    text = _render(panel)
    assert "planner" in text
    assert "3,412" in text
    assert "claude-sonnet-4-6" in text


def test_panel_shows_messages_line_when_quota_set():
    """Messages line appears when quota_messages_max is set and total_messages > 0."""
    state = ProgressState(task="task")
    state.add_unit("u1", "goal")
    usage = _fake_usage(planner_in=100, planner_out=50, total_messages=4)
    panel = _build_panel(state, usage_accumulator=usage, quota_messages_max=40)
    text = _render(panel)
    assert "4" in text
    assert "40" in text
    # Both numbers appear together in the messages line
    assert "4" in text and "40" in text


def test_panel_hides_messages_line_when_no_quota():
    """No quota line appears when quota_messages_max is None."""
    state = ProgressState(task="task")
    state.add_unit("u1", "goal")
    usage = _fake_usage(planner_in=100, planner_out=50, total_messages=4)
    panel = _build_panel(state, usage_accumulator=usage, quota_messages_max=None)
    text = _render(panel)
    # "/ 40" or similar quota fraction should not be present
    assert "/ 40" not in text
    assert "Messages" not in text


def test_panel_elapsed_formatted_minutes_seconds():
    """elapsed=65 seconds formats as '01:05'."""
    state = ProgressState(task="t")
    # Monkey-patch elapsed_seconds for determinism
    state.elapsed_seconds = lambda: 65.0
    panel = _build_panel(state)
    text = _render(panel)
    assert "01:05" in text


def test_panel_elapsed_formatted_hours_minutes_seconds():
    """elapsed=3700 seconds formats as '01:01:40'."""
    state = ProgressState(task="t")
    state.elapsed_seconds = lambda: 3700.0
    panel = _build_panel(state)
    text = _render(panel)
    assert "01:01:40" in text


def test_panel_shows_revision_count():
    """A unit with revision_count=2 shows 'r2' in its row."""
    state = ProgressState(task="t")
    state.add_unit("u1", "goal")
    state.units[0].revision_count = 2
    state.units[0].phase = UnitPhase.DRAFTING
    panel = _build_panel(state)
    text = _render(panel)
    assert "r2" in text


# ---------------------------------------------------------------------------
# make_panel_renderer returns a callable producing fresh panels
# ---------------------------------------------------------------------------

def test_make_panel_renderer_returns_callable():
    """make_panel_renderer returns a zero-argument callable producing a Panel."""
    from rich.panel import Panel

    state = ProgressState(task="t")
    state.add_unit("u1", "goal")
    renderer = make_panel_renderer(state)
    panel = renderer()
    assert isinstance(panel, Panel)


def test_make_panel_renderer_reflects_state_changes():
    """Calling the renderer after a phase change reflects the updated phase."""
    state = ProgressState(task="t")
    state.add_unit("u1", "goal")
    renderer = make_panel_renderer(state)

    text_before = _render(renderer())
    assert "queued" in text_before

    state.set_phase("u1", UnitPhase.COMPLETED)
    text_after = _render(renderer())
    assert "completed" in text_after
