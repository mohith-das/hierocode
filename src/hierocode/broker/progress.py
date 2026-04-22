"""Live progress reporting for `hierocode run`."""

from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import Optional, Protocol

from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class UnitPhase(str, Enum):
    QUEUED = "queued"
    DRAFTING = "drafting"
    REVIEWING = "reviewing"
    REVISING = "revising"
    ESCALATING = "escalating"
    COMPLETED = "completed"
    FAILED = "failed"
    SPLIT = "split"       # when QA splits the unit; kept in history
    ESCALATED = "escalated"  # when a planner draft replaces the drafter's


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------

@dataclass
class UnitProgress:
    unit_id: str
    goal: str
    phase: UnitPhase = UnitPhase.QUEUED
    revision_count: int = 0


@dataclass
class ProgressState:
    """Shared state updated by the dispatcher, read by the panel renderer."""

    task: str = ""
    units: list[UnitProgress] = field(default_factory=list)
    started_at: float = field(default_factory=monotonic)

    def elapsed_seconds(self) -> float:
        """Return wall-clock seconds since this state was created."""
        return monotonic() - self.started_at

    def add_unit(self, unit_id: str, goal: str) -> None:
        """Append a new unit in QUEUED state."""
        self.units.append(UnitProgress(unit_id=unit_id, goal=goal))

    def set_phase(self, unit_id: str, phase: UnitPhase) -> None:
        """Transition the named unit to a new phase."""
        for u in self.units:
            if u.unit_id == unit_id:
                u.phase = phase
                return

    def bump_revision(self, unit_id: str) -> None:
        """Increment the revision counter for the named unit."""
        for u in self.units:
            if u.unit_id == unit_id:
                u.revision_count += 1
                return


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class ProgressReporter(Protocol):
    """Minimal interface the dispatcher calls; the TUI implements it."""

    def seed(self, task: str, units: list[tuple[str, str]]) -> None:
        """Initialise with task name and list of (unit_id, goal) pairs."""
        ...

    def enqueue(self, unit_id: str, goal: str) -> None:
        """Append a newly-created sub-unit (e.g. after a split verdict)."""
        ...

    def phase(self, unit_id: str, phase: UnitPhase) -> None:
        """Transition a unit to a new phase."""
        ...

    def revision(self, unit_id: str) -> None:
        """Record that the drafter is being retried for this unit."""
        ...

    def finished(self) -> None:
        """Called once when run_plan completes."""
        ...


class _NullReporter:
    """No-op implementation for when progress reporting is disabled."""

    def seed(self, task, units): ...  # noqa: E704
    def enqueue(self, unit_id, goal): ...  # noqa: E704
    def phase(self, unit_id, phase): ...  # noqa: E704
    def revision(self, unit_id): ...  # noqa: E704
    def finished(self): ...  # noqa: E704


NULL_REPORTER: ProgressReporter = _NullReporter()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

_PHASE_GLYPH: dict[UnitPhase, str] = {
    UnitPhase.QUEUED: "○",
    UnitPhase.DRAFTING: "●",
    UnitPhase.REVIEWING: "●",
    UnitPhase.REVISING: "●",
    UnitPhase.ESCALATING: "●",
    UnitPhase.COMPLETED: "●",
    UnitPhase.FAILED: "✗",
    UnitPhase.SPLIT: "●",
    UnitPhase.ESCALATED: "●",
}

_PHASE_COLOR: dict[UnitPhase, str] = {
    UnitPhase.QUEUED: "dim",
    UnitPhase.DRAFTING: "cyan",
    UnitPhase.REVIEWING: "cyan",
    UnitPhase.REVISING: "yellow",
    UnitPhase.ESCALATING: "magenta",
    UnitPhase.COMPLETED: "green",
    UnitPhase.FAILED: "red",
    UnitPhase.SPLIT: "green",
    UnitPhase.ESCALATED: "green",
}


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds as MM:SS or HH:MM:SS."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def _build_panel(
    state: ProgressState,
    usage_accumulator=None,
    quota_messages_max: Optional[int] = None,
) -> Panel:
    """Build and return a Rich Panel from current ProgressState + usage."""
    table = Table.grid(padding=(0, 1))
    table.add_column(width=2)   # glyph
    table.add_column(width=6)   # unit id
    table.add_column()          # goal (truncated)
    table.add_column(width=22)  # phase + revision badge

    # ----- Units section -----
    for up in state.units:
        glyph = _PHASE_GLYPH.get(up.phase, "●")
        color = _PHASE_COLOR.get(up.phase, "white")
        phase_label = up.phase.value
        if up.phase in (UnitPhase.DRAFTING, UnitPhase.REVIEWING):
            phase_label += "…"
        badge = f" r{up.revision_count}" if up.revision_count else ""
        goal_display = up.goal[:42] + "…" if len(up.goal) > 43 else up.goal
        table.add_row(
            Text(glyph, style=color),
            Text(up.unit_id, style="bold"),
            Text(goal_display),
            Text(f"[{color}]{phase_label}[/{color}]{badge}"),
        )

    lines: list[object] = [table]

    # ----- Model names (from usage_accumulator) -----
    if usage_accumulator is not None:
        drafter_model = getattr(usage_accumulator.drafter, "model", "") or ""
        planner_model = getattr(usage_accumulator.planner, "model", "") or ""
        if drafter_model or planner_model:
            lines.append(Text(""))
            if drafter_model:
                lines.append(Text(f"  Drafter:   {drafter_model}", style="dim"))
            if planner_model:
                lines.append(Text(f"  Planner:   {planner_model}", style="dim"))

        # ----- Tokens block -----
        roles_to_show = []
        for role_name in ("planner", "drafter", "reviewer"):
            ru = getattr(usage_accumulator, role_name)
            if ru.calls > 0:
                roles_to_show.append((role_name, ru))

        if roles_to_show:
            lines.append(Text(""))
            tok_table = Table.grid(padding=(0, 1))
            tok_table.add_column(width=10)   # "Tokens" / blank
            tok_table.add_column(width=10)   # role name
            tok_table.add_column()           # in/out values
            first = True
            for role_name, ru in roles_to_show:
                label = "Tokens" if first else ""
                first = False
                tok_table.add_row(
                    Text(label, style="bold"),
                    Text(role_name),
                    Text(f"in: {ru.input_tokens:>6,}  out: {ru.output_tokens:>5,}"),
                )
            lines.append(tok_table)

        # ----- Messages / quota line -----
        if quota_messages_max is not None:
            total_msgs = usage_accumulator.total_messages()
            if total_msgs > 0:
                lines.append(Text(""))
                lines.append(
                    Text(f"  Messages:  {total_msgs} / {quota_messages_max}")
                )

    # ----- Elapsed -----
    lines.append(Text(""))
    lines.append(Text(f"  Elapsed:   {_fmt_elapsed(state.elapsed_seconds())}"))

    # Combine into a renderable group
    from rich.console import Group  # local import avoids circular at module level
    body = Group(*lines)

    title = f"hierocode run: {state.task}" if state.task else "hierocode run"
    return Panel(body, title=title, border_style="blue", expand=False)


def make_panel_renderer(
    state: ProgressState,
    usage_accumulator=None,
    quota_messages_max: Optional[int] = None,
):
    """Return a zero-argument callable that produces a fresh Rich Panel on each call.

    Suitable for ``rich.live.Live(get_renderable=make_panel_renderer(...))``.
    """
    def _render():
        return _build_panel(state, usage_accumulator, quota_messages_max)

    return _render
