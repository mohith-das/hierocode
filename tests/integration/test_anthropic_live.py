"""Live integration tests for AnthropicProvider — requires HIEROCODE_TEST_ANTHROPIC=1."""

import json

import pytest

from hierocode.models.schemas import AuthConfig, ProviderConfig
from hierocode.providers.anthropic import AnthropicProvider

from tests.integration.conftest import skip_no_anthropic

_MODEL = "claude-haiku-4-5"
_EXPECTED_MODELS = ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"]


@pytest.fixture(scope="module")
def provider() -> AnthropicProvider:
    """Instantiate a live AnthropicProvider using ANTHROPIC_API_KEY from the environment."""
    config = ProviderConfig.model_construct(
        type="anthropic",
        base_url=None,
        auth=AuthConfig(type="bearer_env", env_var="ANTHROPIC_API_KEY"),
    )
    return AnthropicProvider(name="anthropic_live", config=config)


@skip_no_anthropic
def test_healthcheck_true(provider: AnthropicProvider) -> None:
    """AnthropicProvider.healthcheck() returns True when the API key is valid."""
    assert provider.healthcheck() is True


@skip_no_anthropic
def test_list_models_returns_hardcoded(provider: AnthropicProvider) -> None:
    """list_models() returns the three expected hardcoded model IDs."""
    models = provider.list_models()
    for expected in _EXPECTED_MODELS:
        assert expected in models
    assert len(models) == len(_EXPECTED_MODELS)


@skip_no_anthropic
def test_generate_short_prompt(provider: AnthropicProvider) -> None:
    """generate() returns a string containing 'OK' for a minimal prompt."""
    result = provider.generate(
        "Respond with 'OK'",
        model=_MODEL,
        max_tokens=10,
    )
    assert isinstance(result, str)
    assert "ok" in result.lower()


@skip_no_anthropic
def test_generate_json_mode_produces_valid_json(provider: AnthropicProvider) -> None:
    """generate() with json_mode=True returns valid JSON containing ok=True."""
    result = provider.generate(
        'Respond with {"ok":true}',
        model=_MODEL,
        system="You are a JSON generator.",
        json_mode=True,
        max_tokens=50,
    )
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert isinstance(parsed, dict)
    assert parsed.get("ok") is True
