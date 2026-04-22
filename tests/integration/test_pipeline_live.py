"""Live end-to-end smoke test for the hierocode planning pipeline.

Requires HIEROCODE_TEST_PIPELINE=1 and a running Ollama instance, plus at least
one live planner (HIEROCODE_TEST_CLAUDE_CLI=1 or HIEROCODE_TEST_ANTHROPIC=1 +
ANTHROPIC_API_KEY).  The test exercises generate_plan() but does NOT invoke the
dispatcher or run any drafter calls.
"""

import os
from pathlib import Path
from typing import Tuple

import pytest

from hierocode.broker.capacity import build_capacity_profile
from hierocode.broker.plan_schema import Plan
from hierocode.broker.planner import generate_plan
from hierocode.models.schemas import AuthConfig, ProviderConfig
from hierocode.providers.base import BaseProvider
from hierocode.providers.ollama import OllamaProvider

from tests.integration.conftest import _cli_alive, skip_no_pipeline

_PREFERRED_DRAFTER = "llama3.2:1b"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ollama_provider() -> OllamaProvider:
    """Return an OllamaProvider pointed at localhost:11434."""
    config = ProviderConfig.model_construct(type="ollama", base_url="http://localhost:11434")
    return OllamaProvider(name="ollama_pipeline", config=config)


def _pick_drafter_model(provider: OllamaProvider) -> str:
    """Return the smallest available drafter model or skip if none found."""
    available = provider.list_models()
    if _PREFERRED_DRAFTER in available:
        return _PREFERRED_DRAFTER
    for m in available:
        if "llama3.2" in m or "llama3" in m:
            return m
    if available:
        return available[0]
    pytest.skip("No models found in Ollama — pull a model first (e.g. `ollama pull llama3.2:1b`)")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def planner_setup() -> Tuple[BaseProvider, str]:
    """Pick an available planner provider/model or skip if neither is present."""
    if _cli_alive("claude") and os.environ.get("HIEROCODE_TEST_CLAUDE_CLI") == "1":
        from hierocode.providers.claude_code_cli import ClaudeCodeCliProvider

        config = ProviderConfig.model_construct(
            type="claude_code_cli", auth=AuthConfig(type="none")
        )
        return ClaudeCodeCliProvider(name="claude_cli_planner", config=config), "claude-haiku-4-5"

    if os.environ.get("HIEROCODE_TEST_ANTHROPIC") == "1" and os.environ.get("ANTHROPIC_API_KEY"):
        from hierocode.providers.anthropic import AnthropicProvider

        config = ProviderConfig.model_construct(
            type="anthropic",
            base_url=None,
            auth=AuthConfig(type="bearer_env", env_var="ANTHROPIC_API_KEY"),
        )
        return AnthropicProvider(name="anthropic_planner", config=config), "claude-haiku-4-5"

    pytest.skip("no live planner available")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@skip_no_pipeline
def test_plan_small_refactor(
    tmp_path: Path,
    planner_setup: Tuple[BaseProvider, str],
) -> None:
    """generate_plan() returns a valid Plan with >=1 unit targeting foo.py."""
    planner_provider, planner_model = planner_setup

    # Build a tiny 2-file repo in tmp_path.
    foo = tmp_path / "foo.py"
    foo.write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    bar = tmp_path / "bar.py"
    bar.write_text(
        "from foo import add\n\nresult = add(1, 2)\n",
        encoding="utf-8",
    )

    # Build capacity profile from live Ollama with the smallest available model.
    ollama_provider = _make_ollama_provider()
    drafter_model = _pick_drafter_model(ollama_provider)
    profile = build_capacity_profile(ollama_provider, drafter_model)

    task = "Add a docstring to the function in foo.py."
    skeleton = (
        "foo.py\n"
        "  def add(a, b) -> ...\n\n"
        "bar.py\n"
        "  from foo import add\n"
    )

    plan: Plan = generate_plan(
        task=task,
        skeleton=skeleton,
        profile=profile,
        provider=planner_provider,
        model=planner_model,
        max_tokens=1024,
    )

    assert isinstance(plan, Plan)
    assert len(plan.units) >= 1
    foo_targeted = any("foo.py" in u.target_files for u in plan.units)
    assert foo_targeted, (
        f"Expected at least one TaskUnit targeting foo.py; got units: {plan.units}"
    )
