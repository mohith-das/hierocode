"""Tests for exploration kwargs in broker/planner.py."""

import json
from unittest.mock import MagicMock

from hierocode.broker.plan_schema import CapacityProfile


def _minimal_profile() -> CapacityProfile:
    """Return a minimal valid CapacityProfile."""
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


def _plan_json() -> str:
    """Return a minimal valid JSON Plan string."""
    return json.dumps({
        "task": "test task",
        "units": [
            {
                "id": "u1",
                "goal": "do something",
                "target_files": ["foo.py"],
                "context_files": [],
                "acceptance": "works",
            }
        ],
    })


class TestGeneratePlanExplorationKwargs:
    def test_generate_plan_forwards_exploration_kwargs(self):
        """generate_plan must forward exploration and allowed_tools to provider.generate."""
        from hierocode.broker.planner import generate_plan

        provider = MagicMock()
        provider.generate.return_value = _plan_json()

        generate_plan(
            task="add null check",
            skeleton="# skeleton",
            profile=_minimal_profile(),
            provider=provider,
            model="claude-sonnet-4-6",
            exploration="active",
            allowed_tools=["Read"],
        )

        kwargs = provider.generate.call_args.kwargs
        assert kwargs.get("exploration") == "active"
        assert kwargs.get("allowed_tools") == ["Read"]

    def test_generate_plan_defaults_to_passive(self):
        """generate_plan without exploration kwarg defaults to passive."""
        from hierocode.broker.planner import generate_plan

        provider = MagicMock()
        provider.generate.return_value = _plan_json()

        generate_plan(
            task="add null check",
            skeleton="# skeleton",
            profile=_minimal_profile(),
            provider=provider,
            model="claude-sonnet-4-6",
        )

        kwargs = provider.generate.call_args.kwargs
        assert kwargs.get("exploration") == "passive"
        assert kwargs.get("allowed_tools") is None
