"""Live integration tests for OllamaProvider — requires HIEROCODE_TEST_OLLAMA=1."""

import pytest

from hierocode.broker.capacity import build_capacity_profile
from hierocode.models.schemas import ProviderConfig
from hierocode.providers.ollama import OllamaProvider

from tests.integration.conftest import skip_no_ollama

_PREFERRED_MODEL = "llama3.2:1b"


@pytest.fixture(scope="module")
def provider() -> OllamaProvider:
    """Instantiate a live OllamaProvider pointing at localhost:11434."""
    config = ProviderConfig.model_construct(type="ollama", base_url="http://localhost:11434")
    return OllamaProvider(name="ollama_live", config=config)


@pytest.fixture(scope="module")
def small_model(provider: OllamaProvider) -> str:
    """Return the preferred small model, or skip if it is not present."""
    available = provider.list_models()
    if _PREFERRED_MODEL in available:
        return _PREFERRED_MODEL
    # Accept any model whose name contains the preferred tag stem.
    for m in available:
        if "llama3.2" in m or "llama3" in m:
            return m
    pytest.skip(
        f"Model '{_PREFERRED_MODEL}' not found in Ollama. Pull it with: "
        f"`ollama pull {_PREFERRED_MODEL}`"
    )


@skip_no_ollama
def test_healthcheck_true(provider: OllamaProvider) -> None:
    """OllamaProvider.healthcheck() returns True when Ollama is running."""
    assert provider.healthcheck() is True


@skip_no_ollama
def test_list_models_nonempty(provider: OllamaProvider) -> None:
    """list_models() returns a non-empty list when at least one model is pulled."""
    models = provider.list_models()
    assert isinstance(models, list)
    assert len(models) >= 1


@skip_no_ollama
def test_get_model_info_returns_expected_keys(provider: OllamaProvider, small_model: str) -> None:
    """get_model_info() returns dict with num_ctx (int > 0), param_count_b, quantization."""
    info = provider.get_model_info(small_model)
    assert "num_ctx" in info
    assert isinstance(info["num_ctx"], int)
    assert info["num_ctx"] > 0
    assert "param_count_b" in info
    assert info["param_count_b"] is None or isinstance(info["param_count_b"], float)
    assert "quantization" in info
    assert info["quantization"] is None or isinstance(info["quantization"], str)


@skip_no_ollama
def test_generate_short_prompt(provider: OllamaProvider, small_model: str) -> None:
    """generate() returns a non-empty string for a trivial prompt."""
    result = provider.generate(
        "Reply with the single word: OK",
        model=small_model,
        num_predict=10,
    )
    assert isinstance(result, str)
    assert len(result.strip()) > 0


@skip_no_ollama
def test_capacity_profile_built_from_live_provider(
    provider: OllamaProvider, small_model: str
) -> None:
    """build_capacity_profile() produces a valid CapacityProfile for a small model."""
    profile = build_capacity_profile(provider, small_model)
    assert profile.tier in ("micro", "narrow", "standard", "capable", "strong")
    assert profile.max_input_tokens >= 256
    assert profile.drafter_model == small_model
