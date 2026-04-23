"""Pydantic models for the hierocode hierarchical pipeline."""

import json
import re
from typing import Literal, Optional, TypeVar

from pydantic import BaseModel, Field, field_validator, model_validator

Tier = Literal["micro", "narrow", "standard", "capable", "strong"]
Action = Literal["accept", "revise", "split", "escalate"]

T = TypeVar("T", bound=BaseModel)


class PlanParseError(ValueError):
    """Raised when LLM output cannot be parsed into a plan schema model."""


class CapacityProfile(BaseModel):
    """Hardware and model capacity envelope for the drafter."""

    drafter_model: str
    param_count_b: Optional[float] = None
    quantization: Optional[str] = None
    context_window: int
    host_ram_gb: float
    host_vram_gb: float = 0.0
    host_cpu_cores: int
    has_gpu: bool = False
    tier: Tier
    max_input_tokens: int
    max_output_tokens: int
    max_files_per_unit: int

    @field_validator("context_window", "max_input_tokens", "max_output_tokens", "max_files_per_unit",
                     "host_cpu_cores")
    @classmethod
    def must_be_positive(cls, v: int, info) -> int:  # type: ignore[override]
        """Reject non-positive values for capacity integer fields."""
        if v <= 0:
            raise ValueError(f"{info.field_name} must be positive, got {v}")
        return v


class TaskUnit(BaseModel):
    """A single bounded unit of work for the drafter."""

    id: str = Field(..., min_length=1)
    goal: str = Field(..., min_length=1)
    target_files: list[str] = Field(default_factory=list)
    context_files: list[str] = Field(default_factory=list)
    acceptance: str = ""
    est_input_tokens: int = 0

    @field_validator("est_input_tokens")
    @classmethod
    def non_negative_tokens(cls, v: int) -> int:
        """est_input_tokens must be >= 0."""
        if v < 0:
            raise ValueError(f"est_input_tokens must be >= 0, got {v}")
        return v

    @model_validator(mode="after")
    def requires_at_least_one_file(self) -> "TaskUnit":
        """target_files + context_files combined must not be empty."""
        if not self.target_files and not self.context_files:
            raise ValueError("TaskUnit must have at least one target_file or context_file")
        return self


class Plan(BaseModel):
    """A planner-produced work breakdown for a user task."""

    task: str = Field(..., min_length=1)
    units: list[TaskUnit]

    @model_validator(mode="after")
    def validate_units(self) -> "Plan":
        """units must be non-empty and have unique ids."""
        if not self.units:
            raise ValueError("Plan must contain at least one TaskUnit")
        ids = [u.id for u in self.units]
        if len(ids) != len(set(ids)):
            seen: set[str] = set()
            dupes = [i for i in ids if i in seen or seen.add(i)]  # type: ignore[func-returns-value]
            raise ValueError(f"Duplicate TaskUnit ids: {dupes}")
        return self


class QAVerdict(BaseModel):
    """QA agent's verdict on a drafter's output."""

    action: Action
    feedback: Optional[str] = None
    sub_units: Optional[list[TaskUnit]] = None
    reason: Optional[str] = None

    @model_validator(mode="after")
    def validate_action_constraints(self) -> "QAVerdict":
        """Enforce per-action field requirements. Treats empty collections as
        equivalent to None — many LLMs emit `"feedback": ""` or `"sub_units": []`
        for accept/escalate instead of omitting the key, which is a valid
        semantic match for "no feedback / no sub_units"."""
        if self.action == "revise":
            if not self.feedback:
                raise ValueError("QAVerdict with action='revise' requires non-empty feedback")
        elif self.action == "split":
            if not self.sub_units:
                raise ValueError("QAVerdict with action='split' requires non-empty sub_units")
        elif self.action in ("accept", "escalate"):
            if self.feedback:
                raise ValueError(
                    f"QAVerdict with action='{self.action}' must not have feedback"
                )
            if self.sub_units:
                raise ValueError(
                    f"QAVerdict with action='{self.action}' must not have sub_units"
                )
        return self


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _strip_markdown_fences(raw: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` code fences if present."""
    stripped = raw.strip()
    match = re.match(r"^```(?:json)?\s*\n?([\s\S]*?)\n?```$", stripped, re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


def _extract_first_json_object(text: str) -> str:
    """Return the first balanced {...} substring from text, or raise PlanParseError."""
    depth = 0
    start: Optional[int] = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]
    raise PlanParseError("No JSON object found in LLM output")


def _parse_json_from_llm(raw: str) -> dict:
    """Strip fences, attempt direct parse, then fall back to brace extraction."""
    cleaned = _strip_markdown_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fallback: extract first balanced { } from the original raw text
    candidate = _extract_first_json_object(raw)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise PlanParseError(f"Could not parse JSON from LLM output: {exc}") from exc


def parse_plan_from_llm_output(raw: str) -> Plan:
    """Parse and validate a Plan from raw LLM text."""
    data = _parse_json_from_llm(raw)
    try:
        return Plan.model_validate(data)
    except Exception as exc:
        raise PlanParseError(f"LLM output is valid JSON but failed Plan validation: {exc}") from exc


def parse_qa_verdict_from_llm_output(raw: str) -> QAVerdict:
    """Parse and validate a QAVerdict from raw LLM text."""
    data = _parse_json_from_llm(raw)
    try:
        return QAVerdict.model_validate(data)
    except Exception as exc:
        raise PlanParseError(
            f"LLM output is valid JSON but failed QAVerdict validation: {exc}"
        ) from exc
