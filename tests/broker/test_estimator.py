"""Tests for hierocode.broker.estimator."""


from hierocode.broker.estimator import (
    EstimateResult,
    classify_planner,
    estimate_task_cost,
    estimate_tokens,
)
from hierocode.broker.plan_schema import CapacityProfile
from hierocode.models.schemas import AuthConfig, ProviderConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def minimal_profile() -> CapacityProfile:
    """A valid narrow-tier CapacityProfile for testing."""
    return CapacityProfile(
        drafter_model="llama3.2:3b",
        context_window=8192,
        host_ram_gb=16.0,
        host_vram_gb=0.0,
        host_cpu_cores=4,
        has_gpu=False,
        tier="narrow",
        max_input_tokens=4096,
        max_output_tokens=512,
        max_files_per_unit=3,
    )


def anthropic_config() -> ProviderConfig:
    """ProviderConfig for the Anthropic API."""
    return ProviderConfig.model_construct(
        type="anthropic",
        auth=AuthConfig(type="bearer_env", env_var="ANTHROPIC_API_KEY"),
    )


def cli_config(kind: str) -> ProviderConfig:
    """ProviderConfig for a CLI-subscription provider."""
    return ProviderConfig.model_construct(
        type=kind,
        auth=AuthConfig(type="none"),
    )


def _run(
    task: str = "Add logging",
    skeleton: str = "def main(): pass",
    profile: CapacityProfile | None = None,
    provider_config: ProviderConfig | None = None,
    model: str = "claude-haiku-4-5",
    expected_plan_units: int | None = None,
    max_revisions_per_unit: int = 2,
) -> EstimateResult:
    return estimate_task_cost(
        task=task,
        skeleton=skeleton,
        profile=profile or minimal_profile(),
        planner_provider_config=provider_config or anthropic_config(),
        planner_model=model,
        expected_plan_units=expected_plan_units,
        max_revisions_per_unit=max_revisions_per_unit,
    )


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_estimate_tokens_char_div_4(self):
        assert estimate_tokens("hello world!") == 3

    def test_estimate_tokens_empty(self):
        assert estimate_tokens("") == 0

    def test_estimate_tokens_never_negative(self):
        assert estimate_tokens("abc") == 0


# ---------------------------------------------------------------------------
# classify_planner
# ---------------------------------------------------------------------------


class TestClassifyPlanner:
    def test_classify_planner_variants(self):
        assert classify_planner(ProviderConfig.model_construct(type="anthropic")) == "anthropic_api"
        cli_cfg = ProviderConfig.model_construct(type="claude_code_cli")
        assert classify_planner(cli_cfg) == "claude_code_cli"
        assert classify_planner(ProviderConfig.model_construct(type="codex_cli")) == "codex_cli"
        assert classify_planner(ProviderConfig.model_construct(type="ollama")) == "other"
        assert classify_planner(ProviderConfig.model_construct(type="lmstudio")) == "other"
        assert classify_planner(ProviderConfig.model_construct(type="openai_compatible")) == "other"


# ---------------------------------------------------------------------------
# Anthropic API cost
# ---------------------------------------------------------------------------


class TestAnthropicApiCost:
    def test_anthropic_api_cost_nonzero(self):
        result = _run(model="claude-haiku-4-5")
        assert result.approximate_cost_usd is not None
        assert result.approximate_cost_usd > 0
        assert result.approximate_message_count is None

    def test_anthropic_api_unknown_model_falls_back_to_haiku(self):
        result = _run(model="claude-made-up")
        assert result.approximate_cost_usd is not None
        assert result.approximate_cost_usd > 0
        assert any("fallback" in n.lower() or "claude-haiku" in n for n in result.notes)

    def test_sonnet_is_more_expensive_than_haiku(self):
        haiku_result = _run(task="Refactor utils module", model="claude-haiku-4-5")
        sonnet_result = _run(task="Refactor utils module", model="claude-sonnet-4-6")
        assert haiku_result.approximate_cost_usd is not None
        assert sonnet_result.approximate_cost_usd is not None
        assert sonnet_result.approximate_cost_usd > haiku_result.approximate_cost_usd


# ---------------------------------------------------------------------------
# Subscription / CLI mode
# ---------------------------------------------------------------------------


class TestSubscriptionMode:
    def test_subscription_mode_returns_message_count(self):
        result = _run(provider_config=cli_config("claude_code_cli"), model="claude-sonnet-4-6")
        assert result.approximate_cost_usd is None
        assert result.approximate_message_count is not None
        assert result.approximate_message_count > 0
        assert result.approximate_message_count == 1 + result.expected_qa_calls

    def test_codex_cli_subscription_mode(self):
        result = _run(provider_config=cli_config("codex_cli"), model="gpt-4o")
        assert result.approximate_cost_usd is None
        assert result.approximate_message_count is not None
        assert result.approximate_message_count == 1 + result.expected_qa_calls


# ---------------------------------------------------------------------------
# Other / unknown planner
# ---------------------------------------------------------------------------


class TestOtherPlanner:
    def test_other_planner_both_none(self):
        result = _run(provider_config=ProviderConfig.model_construct(type="ollama"))
        assert result.approximate_cost_usd is None
        assert result.approximate_message_count is None
        assert any("unknown pricing" in n.lower() for n in result.notes)


# ---------------------------------------------------------------------------
# Plan unit heuristic
# ---------------------------------------------------------------------------


class TestExpectedPlanUnits:
    def test_expected_plan_units_heuristic_with_comma(self):
        result = _run(task="Add logging, tests, and docs")
        assert result.expected_plan_units == 3

    def test_expected_plan_units_heuristic_with_and(self):
        result = _run(task="Add logging and tests")
        assert result.expected_plan_units == 3

    def test_expected_plan_units_heuristic_short(self):
        result = _run(task="Fix typo")
        assert result.expected_plan_units == 2

    def test_expected_plan_units_respects_override(self):
        result = _run(task="Fix typo", expected_plan_units=5)
        assert result.expected_plan_units == 5

    def test_expected_plan_units_heuristic_newline(self):
        result = _run(task="Add logging\nAdd tests")
        assert result.expected_plan_units == 3


# ---------------------------------------------------------------------------
# QA calls math
# ---------------------------------------------------------------------------


class TestExpectedQaCalls:
    def test_expected_qa_calls_math(self):
        result = _run(expected_plan_units=3, max_revisions_per_unit=2)
        assert result.expected_qa_calls == 9

    def test_expected_drafter_calls_equals_qa_calls(self):
        result = _run(expected_plan_units=3, max_revisions_per_unit=2)
        assert result.expected_drafter_calls == result.expected_qa_calls


# ---------------------------------------------------------------------------
# Drafter note
# ---------------------------------------------------------------------------


class TestDrafterNote:
    def test_drafter_free_note_always_present_anthropic(self):
        result = _run(provider_config=anthropic_config())
        assert any("Drafter runs locally" in n for n in result.notes)

    def test_drafter_free_note_always_present_cli(self):
        result = _run(provider_config=cli_config("claude_code_cli"))
        assert any("Drafter runs locally" in n for n in result.notes)

    def test_drafter_free_note_always_present_other(self):
        result = _run(provider_config=ProviderConfig.model_construct(type="ollama"))
        assert any("Drafter runs locally" in n for n in result.notes)


# ---------------------------------------------------------------------------
# Cost scales with skeleton size
# ---------------------------------------------------------------------------


class TestCostScalesWithSkeletonSize:
    def test_cost_scales_with_skeleton_size(self):
        small_skeleton = "def main(): pass"
        large_skeleton = small_skeleton * 10

        small_result = _run(skeleton=small_skeleton, model="claude-haiku-4-5")
        large_result = _run(skeleton=large_skeleton, model="claude-haiku-4-5")

        assert small_result.approximate_cost_usd is not None
        assert large_result.approximate_cost_usd is not None
        assert large_result.approximate_cost_usd > small_result.approximate_cost_usd
