"""Dispatcher — main pipeline loop: Plan → draft → QA → accept/revise/split/escalate."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from hierocode.broker.budget import pack_context
from hierocode.broker.plan_schema import CapacityProfile, Plan, QAVerdict, TaskUnit
from hierocode.broker.prompts import build_drafter_prompt, build_drafter_revision_prompt
from hierocode.broker.qa import review_draft
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


def _draft_and_diff(
    unit: TaskUnit,
    prompt: str,
    drafter_provider: BaseProvider,
    drafter_model: str,
    profile: CapacityProfile,
    repo_root: Path | str,
) -> str:
    """Call the drafter and produce a unified diff (or raw output if no target files)."""
    drafted = drafter_provider.generate(
        prompt=prompt,
        model=drafter_model,
        max_tokens=profile.max_output_tokens,
    )
    cleaned = _strip_code_fences(drafted)
    if unit.target_files:
        target = unit.target_files[0]
        original = read_file_safe(Path(repo_root) / target)
        return generate_unified_diff(original, cleaned, target)
    return cleaned


def _escalate(
    unit: TaskUnit,
    packed,
    planner_provider: BaseProvider,
    planner_model: str,
    profile: CapacityProfile,
    repo_root: Path | str,
) -> str:
    """Escalate a unit: let the planner draft directly with extra token headroom."""
    prompt = build_drafter_prompt(unit, packed.content)
    drafted = planner_provider.generate(
        prompt=prompt,
        model=planner_model,
        max_tokens=profile.max_output_tokens * 2,
    )
    cleaned = _strip_code_fences(drafted)
    if unit.target_files:
        target = unit.target_files[0]
        original = read_file_safe(Path(repo_root) / target)
        return generate_unified_diff(original, cleaned, target)
    return cleaned


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
) -> DispatchResult:
    """Run all TaskUnits in a Plan through the draft → QA → verdict loop."""
    result = DispatchResult(task=plan.task)
    total_escalations = 0
    total_revisions = 0

    unit_queue: list[TaskUnit] = list(plan.units)

    while unit_queue and total_escalations <= max_escalations_per_task:
        unit = unit_queue.pop(0)

        packed = pack_context(unit, profile, repo_root)
        prompt = build_drafter_prompt(unit, packed.content)
        diff = _draft_and_diff(unit, prompt, drafter_provider, drafter_model, profile, repo_root)

        revisions = 0

        while True:
            verdict = review_draft(
                planner_provider,
                planner_model,
                unit,
                diff,
                original_task=plan.task,
            )

            if verdict.action == "accept":
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
                        total_escalations += 1
                        diff = _escalate(
                            unit, packed, planner_provider, planner_model, profile, repo_root
                        )
                        result.units.append(
                            UnitResult(
                                unit_id=unit.id,
                                status="escalated",
                                diff=diff,
                                verdict=verdict,
                                revision_count=revisions,
                                escalated=True,
                            )
                        )
                    else:
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
                prompt = build_drafter_revision_prompt(
                    unit, packed.content, diff, verdict.feedback or ""
                )
                diff = _draft_and_diff(
                    unit, prompt, drafter_provider, drafter_model, profile, repo_root
                )
                continue

            elif verdict.action == "split":
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
                    unit_queue.append(su)
                break

            elif verdict.action == "escalate":
                if total_escalations < max_escalations_per_task:
                    total_escalations += 1
                    diff = _escalate(
                        unit, packed, planner_provider, planner_model, profile, repo_root
                    )
                    result.units.append(
                        UnitResult(
                            unit_id=unit.id,
                            status="escalated",
                            diff=diff,
                            verdict=verdict,
                            revision_count=revisions,
                            escalated=True,
                        )
                    )
                else:
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
    return result
