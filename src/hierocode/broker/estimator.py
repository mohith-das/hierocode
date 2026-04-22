"""Cost and quota estimator for hierocode planner calls."""

from dataclasses import dataclass, field
from typing import Literal, Optional

from hierocode.broker.plan_schema import CapacityProfile
from hierocode.broker.prompts import build_planner_system_prompt
from hierocode.models.schemas import ProviderConfig

PlannerKind = Literal["anthropic_api", "claude_code_cli", "codex_cli", "other"]

_HAIKU_MODEL = "claude-haiku-4-5"


def __getattr__(name: str):
    """Lazy re-export of ANTHROPIC_PRICING sourced from the pricing config loader."""
    if name == "ANTHROPIC_PRICING":
        from hierocode.broker.pricing import get_pricing
        return get_pricing().anthropic_models
    raise AttributeError(f"module 'hierocode.broker.estimator' has no attribute {name!r}")


@dataclass
class EstimateResult:
    """Predicted planner cost and quota usage for a task."""

    planner_kind: PlannerKind
    planner_input_tokens: int
    planner_output_tokens: int
    expected_plan_units: int
    expected_qa_calls: int
    expected_drafter_calls: int
    approximate_cost_usd: Optional[float]
    approximate_message_count: Optional[int]
    notes: list[str] = field(default_factory=list)


def estimate_tokens(text: str) -> int:
    """Char/4 heuristic for token count."""
    return max(0, len(text) // 4)


def classify_planner(provider_config: ProviderConfig) -> PlannerKind:
    """Map ProviderConfig.type to a PlannerKind."""
    t = provider_config.type
    if t == "anthropic":
        return "anthropic_api"
    if t == "claude_code_cli":
        return "claude_code_cli"
    if t == "codex_cli":
        return "codex_cli"
    return "other"


def _heuristic_plan_units(task: str) -> int:
    """Guess the number of plan units from task text."""
    has_split = "," in task or " and " in task or "\n" in task
    raw = 3 if has_split else 2
    return max(1, min(8, raw))


def estimate_task_cost(
    task: str,
    skeleton: str,
    profile: CapacityProfile,
    planner_provider_config: ProviderConfig,
    planner_model: str,
    expected_plan_units: Optional[int] = None,
    max_revisions_per_unit: int = 2,
) -> EstimateResult:
    """Predict planner cost / quota usage for a task."""
    planner_kind = classify_planner(planner_provider_config)
    notes: list[str] = []

    # Resolve plan units
    plan_units = (
        expected_plan_units
        if expected_plan_units is not None
        else _heuristic_plan_units(task)
    )
    plan_units = max(1, min(8, plan_units))

    expected_qa_calls = plan_units * (1 + max_revisions_per_unit)
    expected_drafter_calls = expected_qa_calls

    # Token estimation
    system_prompt = build_planner_system_prompt()
    base_input = (
        estimate_tokens(system_prompt)
        + estimate_tokens(skeleton)
        + estimate_tokens(task)
    )
    per_qa_input_per_call = 200 + profile.max_output_tokens
    total_per_qa_input = per_qa_input_per_call * expected_qa_calls
    planner_input_tokens = base_input + total_per_qa_input

    planner_output_tokens = 500 + 100 * expected_qa_calls

    # Cost / message calculation
    approximate_cost_usd: Optional[float] = None
    approximate_message_count: Optional[int] = None

    if planner_kind == "anthropic_api":
        from hierocode.broker.pricing import get_pricing
        anthropic_models = get_pricing().anthropic_models
        if planner_model in anthropic_models:
            input_price, output_price = anthropic_models[planner_model]
        else:
            input_price, output_price = anthropic_models.get(
                _HAIKU_MODEL, (0.25, 1.25)
            )
            notes.append(
                f"Model '{planner_model}' not found in pricing table; "
                f"falling back to {_HAIKU_MODEL} pricing."
            )
        approximate_cost_usd = (
            planner_input_tokens / 1_000_000 * input_price
            + planner_output_tokens / 1_000_000 * output_price
        )
    elif planner_kind in ("claude_code_cli", "codex_cli"):
        approximate_message_count = 1 + expected_qa_calls
        notes.append("Subscription mode — counts against your 5-hour quota.")
    else:
        notes.append("Unknown pricing for provider type; cost not estimated.")

    notes.append("Drafter runs locally — not included in planner cost.")

    return EstimateResult(
        planner_kind=planner_kind,
        planner_input_tokens=planner_input_tokens,
        planner_output_tokens=planner_output_tokens,
        expected_plan_units=plan_units,
        expected_qa_calls=expected_qa_calls,
        expected_drafter_calls=expected_drafter_calls,
        approximate_cost_usd=approximate_cost_usd,
        approximate_message_count=approximate_message_count,
        notes=notes,
    )
