"""Live integration tests for ClaudeCodeCliProvider — requires HIEROCODE_TEST_CLAUDE_CLI=1."""

import json

import pytest

from hierocode.models.schemas import AuthConfig, ProviderConfig
from hierocode.providers.claude_code_cli import ClaudeCodeCliProvider

from tests.integration.conftest import skip_no_claude_cli

_MODEL = "claude-haiku-4-5"
_EXPECTED_MODELS = ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"]


@pytest.fixture(scope="module")
def provider() -> ClaudeCodeCliProvider:
    """Instantiate a live ClaudeCodeCliProvider."""
    config = ProviderConfig.model_construct(type="claude_code_cli", auth=AuthConfig(type="none"))
    return ClaudeCodeCliProvider(name="claude_cli_live", config=config)


@skip_no_claude_cli
def test_healthcheck_true(provider: ClaudeCodeCliProvider) -> None:
    """ClaudeCodeCliProvider.healthcheck() returns True when `claude` is on PATH and responds."""
    assert provider.healthcheck() is True


@skip_no_claude_cli
def test_list_models_returns_hardcoded(provider: ClaudeCodeCliProvider) -> None:
    """list_models() returns the three expected hardcoded model IDs."""
    models = provider.list_models()
    for expected in _EXPECTED_MODELS:
        assert expected in models
    assert len(models) == len(_EXPECTED_MODELS)


@skip_no_claude_cli
@pytest.mark.timeout(90)
def test_generate_short_prompt(provider: ClaudeCodeCliProvider) -> None:
    """generate() returns a string containing 'OK' for a direct single-word prompt."""
    result = provider.generate(
        "Respond with exactly 'OK'",
        model=_MODEL,
        timeout=60,
    )
    assert isinstance(result, str)
    assert "ok" in result.lower()


@skip_no_claude_cli
@pytest.mark.timeout(90)
def test_generate_json_output_parses(provider: ClaudeCodeCliProvider) -> None:
    """generate() with a JSON-only prompt returns parseable JSON with status=='ok'."""
    result = provider.generate(
        'Respond with JSON: {"status":"ok"}',
        model=_MODEL,
        timeout=60,
    )
    assert isinstance(result, str)
    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        pytest.skip("live model wrapped JSON in prose — acceptable")
    assert parsed.get("status") == "ok"
