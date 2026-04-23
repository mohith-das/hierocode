"""Tests for hierocode.broker.dispatcher."""

from unittest.mock import MagicMock, patch


from hierocode.broker.dispatcher import (
    DispatchResult,
    _normalize_target,
    _strip_code_fences,
    run_plan,
)
from hierocode.broker.plan_schema import CapacityProfile, Plan, QAVerdict, TaskUnit
from hierocode.broker.progress import ProgressReporter


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def minimal_profile() -> CapacityProfile:
    """Return a valid CapacityProfile using the 'narrow' tier."""
    return CapacityProfile(
        drafter_model="test-drafter:3b",
        context_window=2048,
        host_ram_gb=8.0,
        host_cpu_cores=4,
        tier="narrow",
        max_input_tokens=1024,
        max_output_tokens=512,
        max_files_per_unit=5,
    )


def unit(uid: str, target_files: list[str] | None = None) -> TaskUnit:
    """Return a minimal valid TaskUnit."""
    tf = target_files if target_files is not None else ["a.py"]
    # Ensure at least one file (plan_schema validator)
    cf: list[str] = [] if tf else ["b.py"]
    return TaskUnit(id=uid, goal=f"Goal for {uid}", target_files=tf, context_files=cf)


def plan_of(units: list[TaskUnit]) -> Plan:
    """Return a Plan wrapping the given units."""
    return Plan(task="test task", units=units)


def _accept_verdict() -> QAVerdict:
    return QAVerdict(action="accept")


def _revise_verdict(feedback: str = "needs work") -> QAVerdict:
    return QAVerdict(action="revise", feedback=feedback)


def _escalate_verdict() -> QAVerdict:
    return QAVerdict(action="escalate")


def _split_verdict(sub_units: list[TaskUnit]) -> QAVerdict:
    return QAVerdict(action="split", sub_units=sub_units)


# ---------------------------------------------------------------------------
# Common mock setup
# ---------------------------------------------------------------------------

_PACK_PATH = "hierocode.broker.dispatcher.pack_context"
_BUILD_PROMPT_PATH = "hierocode.broker.dispatcher.build_drafter_prompt"
_BUILD_REVISION_PATH = "hierocode.broker.dispatcher.build_drafter_revision_prompt"
_REVIEW_PATH = "hierocode.broker.dispatcher.review_draft"
_READ_FILE_PATH = "hierocode.broker.dispatcher.read_file_safe"
_UNIFIED_DIFF_PATH = "hierocode.broker.dispatcher.generate_unified_diff"


def _make_packed(content: str = "packed_content") -> MagicMock:
    packed = MagicMock()
    packed.content = content
    return packed


def _mock_providers(drafter_output: str = "drafted code") -> tuple[MagicMock, MagicMock]:
    planner = MagicMock()
    drafter = MagicMock()
    drafter.generate.return_value = drafter_output
    return planner, drafter


# ---------------------------------------------------------------------------
# test_happy_path_accept
# ---------------------------------------------------------------------------

@patch(_REVIEW_PATH)
@patch(_BUILD_PROMPT_PATH, return_value="prompt")
@patch(_PACK_PATH)
@patch(_READ_FILE_PATH, return_value="original content")
@patch(_UNIFIED_DIFF_PATH, return_value="--- diff ---")
def test_happy_path_accept(mock_diff, mock_read, mock_pack, mock_prompt, mock_review):
    """Single unit; QA returns accept immediately; result has status='completed'."""
    mock_pack.return_value = _make_packed()
    mock_review.return_value = _accept_verdict()

    planner, drafter = _mock_providers()
    p = plan_of([unit("u1")])
    result = run_plan(p, minimal_profile(), planner, "planner-m", drafter, "drafter-m", "/repo")

    assert isinstance(result, DispatchResult)
    assert len(result.units) == 1
    ur = result.units[0]
    assert ur.unit_id == "u1"
    assert ur.status == "completed"
    assert ur.revision_count == 0
    assert result.total_revisions == 0
    assert result.total_escalations == 0


# ---------------------------------------------------------------------------
# test_revise_then_accept
# ---------------------------------------------------------------------------

@patch(_REVIEW_PATH)
@patch(_BUILD_REVISION_PATH, return_value="revision prompt")
@patch(_BUILD_PROMPT_PATH, return_value="initial prompt")
@patch(_PACK_PATH)
@patch(_READ_FILE_PATH, return_value="")
@patch(_UNIFIED_DIFF_PATH, return_value="--- diff ---")
def test_revise_then_accept(mock_diff, mock_read, mock_pack, mock_prompt, mock_revision, mock_review):
    """First review returns revise; second returns accept; revision_count=1."""
    mock_pack.return_value = _make_packed()
    mock_review.side_effect = [_revise_verdict(), _accept_verdict()]

    planner, drafter = _mock_providers()
    p = plan_of([unit("u1")])
    result = run_plan(
        p, minimal_profile(), planner, "planner-m", drafter, "drafter-m", "/repo",
        max_revisions_per_unit=2,
    )

    assert len(result.units) == 1
    ur = result.units[0]
    assert ur.status == "completed"
    assert ur.revision_count == 1
    assert result.total_revisions == 1
    # Revision prompt should have been built once
    mock_revision.assert_called_once()


# ---------------------------------------------------------------------------
# test_revise_caps_then_escalates
# ---------------------------------------------------------------------------

@patch(_REVIEW_PATH)
@patch(_BUILD_REVISION_PATH, return_value="revision prompt")
@patch(_BUILD_PROMPT_PATH, return_value="initial prompt")
@patch(_PACK_PATH)
@patch(_READ_FILE_PATH, return_value="")
@patch(_UNIFIED_DIFF_PATH, return_value="--- diff ---")
def test_revise_caps_then_escalates(mock_diff, mock_read, mock_pack, mock_prompt, mock_revision, mock_review):
    """Always revise; after cap, dispatcher escalates using planner; status='escalated'."""
    mock_pack.return_value = _make_packed()
    # Always return revise — even after escalation attempt the verdict is cached
    mock_review.return_value = _revise_verdict()

    planner, drafter = _mock_providers()
    planner.generate.return_value = "escalated output"

    p = plan_of([unit("u1")])
    result = run_plan(
        p, minimal_profile(), planner, "planner-m", drafter, "drafter-m", "/repo",
        max_revisions_per_unit=1,
        max_escalations_per_task=3,
    )

    assert len(result.units) == 1
    ur = result.units[0]
    assert ur.status == "escalated"
    assert ur.escalated is True
    assert result.total_escalations == 1
    # Planner was called for escalation
    planner.generate.assert_called()


# ---------------------------------------------------------------------------
# test_escalate_direct
# ---------------------------------------------------------------------------

@patch(_REVIEW_PATH)
@patch(_BUILD_PROMPT_PATH, return_value="prompt")
@patch(_PACK_PATH)
@patch(_READ_FILE_PATH, return_value="")
@patch(_UNIFIED_DIFF_PATH, return_value="--- diff ---")
def test_escalate_direct(mock_diff, mock_read, mock_pack, mock_prompt, mock_review):
    """QA immediately returns escalate; planner drafts; status='escalated'."""
    mock_pack.return_value = _make_packed()
    mock_review.return_value = _escalate_verdict()

    planner, drafter = _mock_providers()
    planner.generate.return_value = "planner drafted"

    p = plan_of([unit("u1")])
    result = run_plan(
        p, minimal_profile(), planner, "planner-m", drafter, "drafter-m", "/repo",
    )

    assert len(result.units) == 1
    ur = result.units[0]
    assert ur.status == "escalated"
    assert ur.escalated is True
    assert result.total_escalations == 1
    planner.generate.assert_called()


# ---------------------------------------------------------------------------
# test_escalation_cap_enforced
# ---------------------------------------------------------------------------

@patch(_REVIEW_PATH)
@patch(_BUILD_PROMPT_PATH, return_value="prompt")
@patch(_PACK_PATH)
@patch(_READ_FILE_PATH, return_value="")
@patch(_UNIFIED_DIFF_PATH, return_value="--- diff ---")
def test_escalation_cap_enforced(mock_diff, mock_read, mock_pack, mock_prompt, mock_review):
    """4 units all requesting escalate; cap=3; 4th unit becomes 'failed'."""
    mock_pack.return_value = _make_packed()
    mock_review.return_value = _escalate_verdict()

    planner, drafter = _mock_providers()
    planner.generate.return_value = "escalated output"

    units = [unit(f"u{i}") for i in range(4)]
    p = plan_of(units)
    result = run_plan(
        p, minimal_profile(), planner, "planner-m", drafter, "drafter-m", "/repo",
        max_escalations_per_task=3,
    )

    statuses = [ur.status for ur in result.units]
    assert statuses.count("escalated") == 3
    assert statuses.count("failed") == 1
    failed = next(ur for ur in result.units if ur.status == "failed")
    assert failed.reason is not None
    assert "cap" in failed.reason
    assert result.total_escalations == 3


# ---------------------------------------------------------------------------
# test_split_enqueues_sub_units
# ---------------------------------------------------------------------------

@patch(_REVIEW_PATH)
@patch(_BUILD_PROMPT_PATH, return_value="prompt")
@patch(_PACK_PATH)
@patch(_READ_FILE_PATH, return_value="")
@patch(_UNIFIED_DIFF_PATH, return_value="--- diff ---")
def test_split_enqueues_sub_units(mock_diff, mock_read, mock_pack, mock_prompt, mock_review):
    """Split verdict with 2 sub_units; dispatcher processes all 3 total; original is 'revised'."""
    mock_pack.return_value = _make_packed()

    sub1 = unit("sub1", target_files=["sub1.py"])
    sub2 = unit("sub2", target_files=["sub2.py"])

    # Original unit splits; sub-units accept
    mock_review.side_effect = [
        _split_verdict([sub1, sub2]),
        _accept_verdict(),
        _accept_verdict(),
    ]

    planner, drafter = _mock_providers()
    p = plan_of([unit("u1")])
    result = run_plan(
        p, minimal_profile(), planner, "planner-m", drafter, "drafter-m", "/repo",
    )

    assert len(result.units) == 3
    original = result.units[0]
    assert original.unit_id == "u1"
    assert original.status == "revised"
    assert original.reason is not None
    assert "split" in original.reason

    sub_ids = {ur.unit_id for ur in result.units[1:]}
    assert sub_ids == {"sub1", "sub2"}
    for ur in result.units[1:]:
        assert ur.status == "completed"


# ---------------------------------------------------------------------------
# test_pack_context_called_per_unit
# ---------------------------------------------------------------------------

@patch(_REVIEW_PATH)
@patch(_BUILD_PROMPT_PATH, return_value="prompt")
@patch(_PACK_PATH)
@patch(_READ_FILE_PATH, return_value="")
@patch(_UNIFIED_DIFF_PATH, return_value="--- diff ---")
def test_pack_context_called_per_unit(mock_diff, mock_read, mock_pack, mock_prompt, mock_review):
    """pack_context is called once per unit (2 units → 2 calls)."""
    mock_pack.return_value = _make_packed()
    mock_review.return_value = _accept_verdict()

    planner, drafter = _mock_providers()
    p = plan_of([unit("u1"), unit("u2", target_files=["b.py"])])
    run_plan(p, minimal_profile(), planner, "planner-m", drafter, "drafter-m", "/repo")

    assert mock_pack.call_count == 2


# ---------------------------------------------------------------------------
# test_drafter_generate_called_with_max_tokens_from_profile
# ---------------------------------------------------------------------------

@patch(_REVIEW_PATH)
@patch(_BUILD_PROMPT_PATH, return_value="prompt")
@patch(_PACK_PATH)
@patch(_READ_FILE_PATH, return_value="")
@patch(_UNIFIED_DIFF_PATH, return_value="--- diff ---")
def test_drafter_generate_called_with_max_tokens_from_profile(
    mock_diff, mock_read, mock_pack, mock_prompt, mock_review
):
    """drafter.generate is called with max_tokens matching profile.max_output_tokens."""
    profile = minimal_profile()
    mock_pack.return_value = _make_packed()
    mock_review.return_value = _accept_verdict()

    planner, drafter = _mock_providers()
    p = plan_of([unit("u1")])
    run_plan(p, profile, planner, "planner-m", drafter, "drafter-m", "/repo")

    drafter.generate.assert_called_once_with(
        prompt="prompt",
        model="drafter-m",
        max_tokens=profile.max_output_tokens,
    )


# ---------------------------------------------------------------------------
# test_strip_code_fences
# ---------------------------------------------------------------------------

@patch(_REVIEW_PATH)
@patch(_BUILD_PROMPT_PATH, return_value="prompt")
@patch(_PACK_PATH)
@patch(_READ_FILE_PATH, return_value="original")
@patch(_UNIFIED_DIFF_PATH, side_effect=lambda orig, mod, name: mod)
def test_strip_code_fences(mock_diff, mock_read, mock_pack, mock_prompt, mock_review):
    """Drafter output wrapped in fences is cleaned before diff generation."""
    mock_pack.return_value = _make_packed()
    mock_review.return_value = _accept_verdict()

    planner, drafter = _mock_providers(drafter_output="```python\nclean code\n```")
    p = plan_of([unit("u1")])
    result = run_plan(p, minimal_profile(), planner, "planner-m", drafter, "drafter-m", "/repo")

    assert len(result.units) == 1
    diff = result.units[0].diff
    assert diff is not None
    assert "```" not in diff


# ---------------------------------------------------------------------------
# test_strip_code_fences_unit (pure function test)
# ---------------------------------------------------------------------------

def test_strip_code_fences_pure():
    """_strip_code_fences removes leading and trailing fence lines."""
    fenced = "```python\ndef foo():\n    pass\n```"
    cleaned = _strip_code_fences(fenced)
    assert "```" not in cleaned
    assert "def foo():" in cleaned


# ---------------------------------------------------------------------------
# test_no_target_files_returns_raw_draft
# ---------------------------------------------------------------------------

@patch(_REVIEW_PATH)
@patch(_BUILD_PROMPT_PATH, return_value="prompt")
@patch(_PACK_PATH)
def test_no_target_files_returns_raw_draft(mock_pack, mock_prompt, mock_review):
    """Unit with no target_files: diff equals the cleaned draft (no unified_diff header)."""
    mock_pack.return_value = _make_packed()
    mock_review.return_value = _accept_verdict()

    planner, drafter = _mock_providers(drafter_output="plain output here")
    # Unit with context_files only (target_files=[])
    u = TaskUnit(id="u1", goal="goal", target_files=[], context_files=["b.py"])
    p = plan_of([u])

    result = run_plan(p, minimal_profile(), planner, "planner-m", drafter, "drafter-m", "/repo")

    assert len(result.units) == 1
    diff = result.units[0].diff
    assert diff == "plain output here"
    assert "---" not in diff or diff.startswith("plain")


# ---------------------------------------------------------------------------
# escalation_confirm callback tests (W27)
# ---------------------------------------------------------------------------

@patch("hierocode.broker.dispatcher._escalate")
@patch(_REVIEW_PATH)
@patch(_BUILD_PROMPT_PATH, return_value="prompt")
@patch(_PACK_PATH)
@patch(_READ_FILE_PATH, return_value="")
@patch(_UNIFIED_DIFF_PATH, return_value="--- diff ---")
def test_escalation_confirm_approves_proceeds(
    mock_diff, mock_read, mock_pack, mock_prompt, mock_review, mock_escalate
):
    """When escalation_confirm returns True, _escalate is called and unit is 'escalated'."""
    mock_pack.return_value = _make_packed()
    mock_review.return_value = _escalate_verdict()
    mock_escalate.return_value = "--- escalated diff ---"

    planner, drafter = _mock_providers()
    confirm = MagicMock(return_value=True)

    p = plan_of([unit("u1")])
    result = run_plan(
        p,
        minimal_profile(),
        planner,
        "planner-m",
        drafter,
        "drafter-m",
        "/repo",
        escalation_confirm=confirm,
    )

    confirm.assert_called_once()
    assert len(result.units) == 1
    ur = result.units[0]
    assert ur.status == "escalated"
    assert ur.escalated is True
    mock_escalate.assert_called_once()


@patch("hierocode.broker.dispatcher._escalate")
@patch(_REVIEW_PATH)
@patch(_BUILD_PROMPT_PATH, return_value="prompt")
@patch(_PACK_PATH)
@patch(_READ_FILE_PATH, return_value="")
@patch(_UNIFIED_DIFF_PATH, return_value="--- diff ---")
def test_escalation_confirm_declines_marks_failed(
    mock_diff, mock_read, mock_pack, mock_prompt, mock_review, mock_escalate
):
    """When escalation_confirm returns False, _escalate is NOT called; unit fails with 'declined'."""
    mock_pack.return_value = _make_packed()
    mock_review.return_value = _escalate_verdict()

    planner, drafter = _mock_providers()
    confirm = MagicMock(return_value=False)

    p = plan_of([unit("u1")])
    result = run_plan(
        p,
        minimal_profile(),
        planner,
        "planner-m",
        drafter,
        "drafter-m",
        "/repo",
        escalation_confirm=confirm,
    )

    confirm.assert_called_once()
    mock_escalate.assert_not_called()
    assert len(result.units) == 1
    ur = result.units[0]
    assert ur.status == "failed"
    assert ur.reason is not None
    assert "declined" in ur.reason


# ---------------------------------------------------------------------------
# test_dispatcher_calls_reporter_at_phase_transitions
# ---------------------------------------------------------------------------

@patch(_REVIEW_PATH)
@patch(_BUILD_PROMPT_PATH, return_value="prompt")
@patch(_PACK_PATH)
@patch(_READ_FILE_PATH, return_value="original content")
@patch(_UNIFIED_DIFF_PATH, return_value="--- diff ---")
def test_dispatcher_calls_reporter_at_phase_transitions(
    mock_diff, mock_read, mock_pack, mock_prompt, mock_review
):
    """progress_reporter is called with seed, phase transitions, and finished."""
    mock_pack.return_value = _make_packed()
    mock_review.return_value = _accept_verdict()

    planner, drafter = _mock_providers()
    p = plan_of([unit("u1")])

    reporter = MagicMock(spec=ProgressReporter)

    result = run_plan(
        p,
        minimal_profile(),
        planner,
        "planner-m",
        drafter,
        "drafter-m",
        "/repo",
        progress_reporter=reporter,
    )

    assert isinstance(result, DispatchResult)
    reporter.seed.assert_called_once()
    assert reporter.phase.called
    reporter.finished.assert_called_once()


# ---------------------------------------------------------------------------
# test_run_plan_forwards_reviewer_exploration_to_review_draft (W28)
# ---------------------------------------------------------------------------

@patch(_REVIEW_PATH)
@patch(_BUILD_PROMPT_PATH, return_value="prompt")
@patch(_PACK_PATH)
@patch(_READ_FILE_PATH, return_value="")
@patch(_UNIFIED_DIFF_PATH, return_value="--- diff ---")
def test_run_plan_forwards_reviewer_exploration_to_review_draft(
    mock_diff, mock_read, mock_pack, mock_prompt, mock_review
):
    """reviewer_exploration and reviewer_allowed_tools are forwarded to review_draft."""
    mock_pack.return_value = _make_packed()
    mock_review.return_value = _accept_verdict()

    planner, drafter = _mock_providers()
    p = plan_of([unit("u1")])
    run_plan(
        p,
        minimal_profile(),
        planner,
        "planner-m",
        drafter,
        "drafter-m",
        "/repo",
        reviewer_exploration="active",
        reviewer_allowed_tools=["Read"],
    )

    call_kwargs = mock_review.call_args.kwargs
    assert call_kwargs.get("exploration") == "active"
    assert call_kwargs.get("allowed_tools") == ["Read"]


# ---------------------------------------------------------------------------
# test_normalize_target — regression for v0.3.5 bug #1
# ---------------------------------------------------------------------------

class TestNormalizeTarget:
    """Regression tests for _normalize_target.

    Bug: the repo skeleton renders the repo-root basename as a top-level line
    (e.g. `fb-claude/`), which some planners mistake for a parent path segment
    and emit into `target_files` as a prefix. That produced diffs pointing at
    `<repo>/fb-claude/index.html` which git apply then rejected.
    """

    def test_strips_repo_name_prefix(self):
        assert _normalize_target(
            "fb-claude/index.html", "/Users/x/Desktop/fb-claude"
        ) == "index.html"

    def test_strips_repo_name_prefix_nested(self):
        assert _normalize_target(
            "myproj/src/app.py", "/tmp/myproj"
        ) == "src/app.py"

    def test_passthrough_when_no_prefix_match(self):
        assert _normalize_target(
            "index.html", "/Users/x/Desktop/fb-claude"
        ) == "index.html"

    def test_passthrough_when_genuine_subdir(self):
        assert _normalize_target(
            "src/app.py", "/tmp/unrelated"
        ) == "src/app.py"

    def test_handles_empty_repo_root(self):
        # Defensive: empty or pathological repo_root should never crash
        assert _normalize_target("index.html", "") == "index.html"
