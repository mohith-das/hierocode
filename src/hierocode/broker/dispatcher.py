"""Dispatcher — main pipeline loop: Plan → draft → QA → accept/revise/split/escalate."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

from hierocode.broker.budget import pack_context
from hierocode.broker.plan_schema import CapacityProfile, Plan, QAVerdict, TaskUnit
from hierocode.broker.progress import NULL_REPORTER, UnitPhase
from hierocode.broker.progress import ProgressReporter  # noqa: F401 — re-exported for callers
from hierocode.broker.edits import apply_edit_blocks, parse_edit_blocks, EditApplyError
from hierocode.broker.prompts import build_drafter_prompt, build_drafter_revision_prompt
from hierocode.broker.qa import review_draft
from hierocode.broker.usage import UsageAccumulator, UsageInfo
from hierocode.providers.base import BaseProvider
from hierocode.repo.diffing import generate_unified_diff
from hierocode.repo.files import read_file_safe

UnitStatus = Literal["completed", "failed", "escalated", "revised"]


@dataclass
class UnitResult:
    """Result for a single dispatched TaskUnit."""

    unit_id: str
    status: UnitStatus
    diff: Optional[str] = None
    verdict: Optional[QAVerdict] = None
    revision_count: int = 0
    escalated: bool = False
    reason: Optional[str] = None


@dataclass
class DispatchResult:
    """Aggregate result for an entire Plan run."""

    task: str
    units: list[UnitResult] = field(default_factory=list)
    total_revisions: int = 0
    total_escalations: int = 0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences from generated text."""
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def _normalize_target(target: str, repo_root: Path | str) -> str:
    """Strip a leading repo-root basename prefix from a planner-supplied path.

    The repo skeleton shown to the planner begins with a header line like
    ``fb-claude/`` (the directory's basename). Some planners mistake this for
    a parent path component and emit `target_files: ["fb-claude/index.html"]`.
    Those paths resolve to `<repo_root>/fb-claude/index.html` which doesn't
    exist and produces diffs that `git apply` rejects. Normalize away the
    prefix so on-disk paths match reality.
    """
    root_name = Path(repo_root).resolve().name
    if not root_name:
        return target
    parts = Path(target).parts
    if len(parts) > 1 and parts[0] == root_name:
        return str(Path(*parts[1:]))
    return target


def _current_content(path: str, repo_root: Path | str, file_state: dict[str, str]) -> str:
    """Return the unit-visible content of a file: overlay first, then disk."""
    if path in file_state:
        return file_state[path]
    return read_file_safe(Path(repo_root) / path)


def _draft_and_diff(
    unit: TaskUnit,
    prompt: str,
    drafter_provider: BaseProvider,
    drafter_model: str,
    profile: CapacityProfile,
    repo_root: Path | str,
    file_state: dict[str, str],
    original: str = "",
) -> tuple[str | None, str | None]:
    """Call the drafter and produce a unified diff and the new content (or None if no target)."""
    drafted = drafter_provider.generate(
        prompt=prompt,
        model=drafter_model,
        max_tokens=profile.max_output_tokens,
    )
    cleaned = _strip_code_fences(drafted)
    if unit.target_files:
        target = _normalize_target(unit.target_files[0], repo_root)
        
        blocks = parse_edit_blocks(cleaned)
        if blocks:
            new_content = apply_edit_blocks(original, blocks)
        else:
            new_content = cleaned
            
        diff = generate_unified_diff(original, new_content, target)
        return diff, new_content
    return cleaned, None


def _escalate(
    unit: TaskUnit,
    packed,
    planner_provider: BaseProvider,
    planner_model: str,
    profile: CapacityProfile,
    repo_root: Path | str,
    file_state: dict[str, str],
    original: str = "",
) -> tuple[str | None, str | None]:
    """Escalate a unit: let the planner draft directly with extra token headroom."""
    mode: Literal["whole_file", "edit_blocks"] = "whole_file" if original == "" else "edit_blocks"
    prompt = build_drafter_prompt(unit, packed.content, mode=mode)
    drafted = planner_provider.generate(
        prompt=prompt,
        model=planner_model,
        max_tokens=profile.max_output_tokens * 2,
    )
    cleaned = _strip_code_fences(drafted)
    if unit.target_files:
        target = _normalize_target(unit.target_files[0], repo_root)
        
        blocks = parse_edit_blocks(cleaned)
        if blocks:
            new_content = apply_edit_blocks(original, blocks)
        else:
            new_content = cleaned
            
        diff = generate_unified_diff(original, new_content, target)
        return diff, new_content
    return cleaned, None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_plan(
    plan: Plan,
    profile: CapacityProfile,
    planner_provider: BaseProvider,
    planner_model: str,
    drafter_provider: BaseProvider,
    drafter_model: str,
    repo_root: Path | str,
    max_revisions_per_unit: int = 2,
    max_escalations_per_task: int = 3,
    usage_accumulator: Optional[UsageAccumulator] = None,
    progress_reporter: Optional["ProgressReporter"] = None,
    escalation_confirm: Optional[Callable[[TaskUnit, str, int], bool]] = None,
    reviewer_exploration: Literal["passive", "active"] = "passive",
    reviewer_allowed_tools: Optional[list[str]] = None,
    max_total_units: Optional[int] = None,
) -> DispatchResult:
    """Run all TaskUnits in a Plan through the draft → QA → verdict loop.

    If ``usage_accumulator`` is provided, token/message usage from every provider
    call is recorded into it by role (drafter / reviewer / planner-escalation).

    If ``progress_reporter`` is provided, phase-transition callbacks are fired at
    each state change so live panels (or tests) can react.  Pass ``None`` (the
    default) for the existing behaviour — a null reporter is substituted internally.
    """
    reporter: ProgressReporter = progress_reporter if progress_reporter is not None else NULL_REPORTER  # type: ignore[assignment]

    result = DispatchResult(task=plan.task)
    total_escalations = 0
    total_revisions = 0
    
    if max_total_units is None:
        max_total_units = max(12, 3 * len(plan.units))

    unit_queue: list[TaskUnit] = list(plan.units)
    file_state: dict[str, str] = {}
    units_processed = 0

    reporter.seed(plan.task, [(u.id, u.goal) for u in plan.units])

    while unit_queue:
        if units_processed >= max_total_units:
            for remaining_unit in unit_queue:
                reporter.phase(remaining_unit.id, UnitPhase.FAILED)
                result.units.append(
                    UnitResult(
                        unit_id=remaining_unit.id,
                        status="failed",
                        reason="unit budget exhausted (possible split loop)"
                    )
                )
            break
            
        unit = unit_queue.pop(0)
        units_processed += 1

        reporter.phase(unit.id, UnitPhase.DRAFTING)
        packed = pack_context(unit, profile, repo_root, file_state=file_state)
        
        if packed.infeasible_targets:
            reporter.phase(unit.id, UnitPhase.FAILED)
            result.units.append(
                UnitResult(
                    unit_id=unit.id,
                    status="failed",
                    reason=f"target file(s) {packed.infeasible_targets} exceed the drafter's input budget; split the unit or use a larger drafter"
                )
            )
            continue
            
        original = ""
        if unit.target_files:
            target = _normalize_target(unit.target_files[0], repo_root)
            original = _current_content(target, repo_root, file_state)

        mode: Literal["whole_file", "edit_blocks"] = "whole_file" if original == "" else "edit_blocks"
        prompt = build_drafter_prompt(unit, packed.content, mode=mode)
        
        pre_review_verdict = None
        try:
            diff, new_content = _draft_and_diff(unit, prompt, drafter_provider, drafter_model, profile, repo_root, file_state, original)
        except EditApplyError as exc:
            diff, new_content = None, None
            pre_review_verdict = QAVerdict(
                action="revise",
                feedback=f"Your edit blocks could not be applied: {exc}. Re-emit the blocks with SEARCH text copied exactly from the file."
            )

        if usage_accumulator is not None and isinstance(drafter_provider.last_usage, UsageInfo):
            usage_accumulator.record("drafter", drafter_provider.last_usage)

        revisions = 0

        while True:
            if pre_review_verdict:
                verdict = pre_review_verdict
                pre_review_verdict = None
            else:
                reporter.phase(unit.id, UnitPhase.REVIEWING)
                verdict = review_draft(
                    planner_provider,
                    planner_model,
                    unit,
                    diff,
                    original_task=plan.task,
                    exploration=reviewer_exploration,
                    allowed_tools=reviewer_allowed_tools,
                )
                if usage_accumulator is not None and isinstance(planner_provider.last_usage, UsageInfo):
                    usage_accumulator.record("reviewer", planner_provider.last_usage)

            if verdict.action == "accept":
                reporter.phase(unit.id, UnitPhase.COMPLETED)
                if unit.target_files and new_content is not None:
                    file_state[_normalize_target(unit.target_files[0], repo_root)] = new_content
                result.units.append(
                    UnitResult(
                        unit_id=unit.id,
                        status="completed",
                        diff=diff,
                        verdict=verdict,
                        revision_count=revisions,
                    )
                )
                break

            elif verdict.action == "revise":
                revisions += 1
                total_revisions += 1

                if revisions > max_revisions_per_unit:
                    # Revision cap exhausted — escalate or fail.
                    if total_escalations < max_escalations_per_task:
                        if escalation_confirm is not None:
                            if not escalation_confirm(unit, planner_model, revisions):
                                reporter.phase(unit.id, UnitPhase.FAILED)
                                result.units.append(
                                    UnitResult(
                                        unit_id=unit.id,
                                        status="failed",
                                        diff=diff,
                                        verdict=verdict,
                                        revision_count=revisions,
                                        reason="escalation declined by user",
                                    )
                                )
                                break
                        total_escalations += 1
                        reporter.phase(unit.id, UnitPhase.ESCALATING)
                        diff, new_content = _escalate(
                            unit, packed, planner_provider, planner_model, profile, repo_root, file_state, original
                        )
                        if usage_accumulator is not None and isinstance(planner_provider.last_usage, UsageInfo):
                            usage_accumulator.record("planner", planner_provider.last_usage)
                        
                        # QA the escalated draft once
                        reporter.phase(unit.id, UnitPhase.REVIEWING)
                        esc_verdict = review_draft(
                            planner_provider,
                            planner_model,
                            unit,
                            diff or "",
                            original_task=plan.task,
                            exploration=reviewer_exploration,
                            allowed_tools=reviewer_allowed_tools,
                        )
                        if usage_accumulator is not None and isinstance(planner_provider.last_usage, UsageInfo):
                            usage_accumulator.record("reviewer", planner_provider.last_usage)

                        if esc_verdict.action == "accept":
                            reporter.phase(unit.id, UnitPhase.ESCALATED)
                            if unit.target_files and new_content is not None:
                                file_state[_normalize_target(unit.target_files[0], repo_root)] = new_content
                            result.units.append(
                                UnitResult(
                                    unit_id=unit.id,
                                    status="escalated",
                                    diff=diff,
                                    verdict=esc_verdict,
                                    revision_count=revisions,
                                    escalated=True,
                                )
                            )
                        else:
                            reporter.phase(unit.id, UnitPhase.FAILED)
                            result.units.append(
                                UnitResult(
                                    unit_id=unit.id,
                                    status="failed",
                                    diff=diff,
                                    verdict=esc_verdict,
                                    revision_count=revisions,
                                    escalated=True,
                                    reason=f"escalated draft rejected by reviewer: {esc_verdict.action}",
                                )
                            )
                    else:
                        reporter.phase(unit.id, UnitPhase.FAILED)
                        result.units.append(
                            UnitResult(
                                unit_id=unit.id,
                                status="failed",
                                diff=diff,
                                verdict=verdict,
                                revision_count=revisions,
                                reason="revision cap + escalation cap exhausted",
                            )
                        )
                    break

                # Retry drafter with revision prompt.
                reporter.revision(unit.id)
                reporter.phase(unit.id, UnitPhase.REVISING)
                prompt = build_drafter_revision_prompt(
                    unit, packed.content, diff or "", verdict.feedback or "", mode=mode
                )
                reporter.phase(unit.id, UnitPhase.DRAFTING)
                try:
                    diff, new_content = _draft_and_diff(
                        unit, prompt, drafter_provider, drafter_model, profile, repo_root, file_state, original
                    )
                except EditApplyError as exc:
                    diff, new_content = None, None
                    pre_review_verdict = QAVerdict(
                        action="revise",
                        feedback=f"Your edit blocks could not be applied: {exc}. Re-emit the blocks with SEARCH text copied exactly from the file."
                    )
                if usage_accumulator is not None and isinstance(drafter_provider.last_usage, UsageInfo):
                    usage_accumulator.record("drafter", drafter_provider.last_usage)
                continue

            elif verdict.action == "split":
                reporter.phase(unit.id, UnitPhase.SPLIT)
                result.units.append(
                    UnitResult(
                        unit_id=unit.id,
                        status="revised",
                        diff=diff,
                        verdict=verdict,
                        revision_count=revisions,
                        reason="split into sub_units",
                    )
                )
                for su in verdict.sub_units or []:
                    reporter.enqueue(su.id, su.goal)
                    unit_queue.append(su)
                break

            elif verdict.action == "escalate":
                if total_escalations < max_escalations_per_task:
                    if escalation_confirm is not None:
                        if not escalation_confirm(unit, planner_model, revisions):
                            reporter.phase(unit.id, UnitPhase.FAILED)
                            result.units.append(
                                UnitResult(
                                    unit_id=unit.id,
                                    status="failed",
                                    diff=diff,
                                    verdict=verdict,
                                    revision_count=revisions,
                                    reason="escalation declined by user",
                                )
                            )
                            break
                    total_escalations += 1
                    reporter.phase(unit.id, UnitPhase.ESCALATING)
                    diff, new_content = _escalate(
                        unit, packed, planner_provider, planner_model, profile, repo_root, file_state, original
                    )
                    if usage_accumulator is not None and isinstance(planner_provider.last_usage, UsageInfo):
                        usage_accumulator.record("planner", planner_provider.last_usage)
                    
                    # QA the escalated draft once
                    reporter.phase(unit.id, UnitPhase.REVIEWING)
                    esc_verdict = review_draft(
                        planner_provider,
                        planner_model,
                        unit,
                        diff or "",
                        original_task=plan.task,
                        exploration=reviewer_exploration,
                        allowed_tools=reviewer_allowed_tools,
                    )
                    if usage_accumulator is not None and isinstance(planner_provider.last_usage, UsageInfo):
                        usage_accumulator.record("reviewer", planner_provider.last_usage)

                    if esc_verdict.action == "accept":
                        reporter.phase(unit.id, UnitPhase.ESCALATED)
                        if unit.target_files and new_content is not None:
                            file_state[_normalize_target(unit.target_files[0], repo_root)] = new_content
                        result.units.append(
                            UnitResult(
                                unit_id=unit.id,
                                status="escalated",
                                diff=diff,
                                verdict=esc_verdict,
                                revision_count=revisions,
                                escalated=True,
                            )
                        )
                    else:
                        reporter.phase(unit.id, UnitPhase.FAILED)
                        result.units.append(
                            UnitResult(
                                unit_id=unit.id,
                                status="failed",
                                diff=diff,
                                verdict=esc_verdict,
                                revision_count=revisions,
                                escalated=True,
                                reason=f"escalated draft rejected by reviewer: {esc_verdict.action}",
                            )
                        )
                else:
                    reporter.phase(unit.id, UnitPhase.FAILED)
                    result.units.append(
                        UnitResult(
                            unit_id=unit.id,
                            status="failed",
                            diff=diff,
                            verdict=verdict,
                            revision_count=revisions,
                            reason="escalation cap exhausted",
                        )
                    )
                break

    result.total_revisions = total_revisions
    result.total_escalations = total_escalations
    reporter.finished()
    return result
