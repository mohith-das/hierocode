"""Planner — invokes the planner provider and parses a JSON Plan."""

from typing import Literal, Optional

from hierocode.broker.plan_schema import CapacityProfile, Plan, parse_plan_from_llm_output
from hierocode.broker.prompts import build_planner_system_prompt, build_planner_user_prompt
from hierocode.providers.base import BaseProvider


def generate_plan(
    task: str,
    skeleton: str,
    profile: CapacityProfile,
    provider: BaseProvider,
    model: str,
    max_tokens: int = 4000,
    exploration: Literal["passive", "active"] = "passive",
    allowed_tools: Optional[list[str]] = None,
) -> Plan:
    """Ask the planner provider to produce a validated JSON Plan."""
    system = build_planner_system_prompt()
    user = build_planner_user_prompt(task=task, skeleton=skeleton, profile=profile)
    raw = provider.generate(
        prompt=user,
        model=model,
        system=system,
        json_mode=True,
        max_tokens=max_tokens,
        exploration=exploration,
        allowed_tools=allowed_tools,
    )
    return parse_plan_from_llm_output(raw)
