"""QA module — asks the planner to review a drafter's diff and return a QAVerdict."""

from typing import Literal, Optional

from hierocode.broker.plan_schema import PlanParseError, QAVerdict, TaskUnit
from hierocode.broker.plan_schema import parse_qa_verdict_from_llm_output
from hierocode.providers.base import BaseProvider

try:
    from hierocode.broker.prompts import build_qa_prompt
except ImportError:
    build_qa_prompt = None  # type: ignore[assignment]

_QA_SYSTEM = (
    "You are a strict code reviewer. Always respond with valid JSON matching the QAVerdict schema."
)


def review_draft(
    planner_provider: BaseProvider,
    planner_model: str,
    unit: TaskUnit,
    diff: str,
    test_output: Optional[str] = None,
    original_task: str = "",
    max_tokens: int = 2000,
    exploration: Literal["passive", "active"] = "passive",
    allowed_tools: Optional[list[str]] = None,
) -> QAVerdict:
    """Send a drafter's diff to the planner for QA review and return a QAVerdict."""
    qa_prompt = build_qa_prompt(unit, diff, test_output, original_task)  # type: ignore[misc]

    raw: str = planner_provider.generate(
        prompt=qa_prompt,
        model=planner_model,
        json_mode=True,
        max_tokens=max_tokens,
        system=_QA_SYSTEM,
        exploration=exploration,
        allowed_tools=allowed_tools,
    )

    try:
        return parse_qa_verdict_from_llm_output(raw)
    except PlanParseError as exc:
        raise PlanParseError(
            f"QA response unparseable: {exc}\n---\nRaw response: {raw[:1000]}"
        ) from exc
