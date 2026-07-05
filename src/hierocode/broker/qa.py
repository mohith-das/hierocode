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
        # Retry exactly once. Note: If a usage accumulator is involved, it will only reflect
        # the usage from this second call.
        retry_prompt = qa_prompt + f"\n\n## Previous attempt failed\n\nYour previous response could not be parsed:\n{exc}\n\nRespond again with ONLY the valid JSON object. No prose, no code fences."
        raw_retry: str = planner_provider.generate(
            prompt=retry_prompt,
            model=planner_model,
            json_mode=True,
            max_tokens=max_tokens,
            system=_QA_SYSTEM,
            exploration=exploration,
            allowed_tools=allowed_tools,
        )
        try:
            return parse_qa_verdict_from_llm_output(raw_retry)
        except PlanParseError as exc_retry:
            raise PlanParseError(
                f"QA response unparseable after retry: {exc_retry}\n---\nRaw response: {raw_retry[:1000]}"
            ) from exc_retry
