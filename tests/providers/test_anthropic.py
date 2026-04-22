import os
import pytest
from unittest.mock import MagicMock, patch

from hierocode.models.schemas import AuthConfig, ProviderConfig
from hierocode.exceptions import ProviderConnectionError, ModelNotFoundError

# Build a ProviderConfig that bypasses the Literal restriction on `type`.
_CONFIG = ProviderConfig.model_construct(
    type="anthropic",
    base_url=None,
    auth=AuthConfig(type="bearer_env", env_var="ANTHROPIC_API_KEY"),
)


def _make_provider(config=_CONFIG):
    # Import here so patching of Anthropic at module level takes effect.
    from hierocode.providers.anthropic import AnthropicProvider
    return AnthropicProvider(name="test_anthropic", config=config)


class TestAnthropicProviderInit:
    def test_requires_api_key_env(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with patch("hierocode.providers.anthropic.Anthropic", MagicMock()):
                provider = _make_provider()
                with pytest.raises(ProviderConnectionError, match="ANTHROPIC_API_KEY not set"):
                    provider._get_client()

    def test_custom_env_var_missing(self):
        config = ProviderConfig.model_construct(
            type="anthropic",
            base_url=None,
            auth=AuthConfig(type="bearer_env", env_var="MY_CUSTOM_KEY"),
        )
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MY_CUSTOM_KEY", None)
            with patch("hierocode.providers.anthropic.Anthropic", MagicMock()):
                provider = _make_provider(config)
                with pytest.raises(ProviderConnectionError, match="MY_CUSTOM_KEY not set"):
                    provider._get_client()


class TestAnthropicGenerate:
    def _patched_provider(self, response_text="hello"):
        mock_content = MagicMock(text=response_text)
        mock_response = MagicMock()
        mock_response.content = [mock_content]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        mock_anthropic_cls = MagicMock(return_value=mock_client)
        return mock_anthropic_cls, mock_client

    def test_generate_returns_text(self):
        mock_cls, mock_client = self._patched_provider("hello")
        with patch("hierocode.providers.anthropic.Anthropic", mock_cls):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
                provider = _make_provider()
                result = provider.generate("say hi", model="claude-haiku-4-5")
        assert result == "hello"

    def test_generate_passes_model_and_max_tokens(self):
        mock_cls, mock_client = self._patched_provider()
        with patch("hierocode.providers.anthropic.Anthropic", mock_cls):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
                provider = _make_provider()
                provider.generate("prompt text", model="claude-sonnet-4-6", max_tokens=512)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-6"
        assert call_kwargs["max_tokens"] == 512
        assert call_kwargs["messages"] == [{"role": "user", "content": "prompt text"}]

    def test_generate_default_max_tokens(self):
        mock_cls, mock_client = self._patched_provider()
        with patch("hierocode.providers.anthropic.Anthropic", mock_cls):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
                provider = _make_provider()
                provider.generate("prompt", model="claude-haiku-4-5")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 4096

    def test_generate_json_mode_adds_system_instruction(self):
        mock_cls, mock_client = self._patched_provider()
        with patch("hierocode.providers.anthropic.Anthropic", mock_cls):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
                provider = _make_provider()
                provider.generate("prompt", model="claude-haiku-4-5", json_mode=True)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "system" in call_kwargs
        assert "JSON" in call_kwargs["system"]

    def test_generate_system_kwarg_passed_through(self):
        mock_cls, mock_client = self._patched_provider()
        with patch("hierocode.providers.anthropic.Anthropic", mock_cls):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
                provider = _make_provider()
                provider.generate("prompt", model="claude-haiku-4-5", system="You are helpful.")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "You are helpful." in call_kwargs["system"]

    def test_generate_model_not_found_error(self):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("model not found: bad-model")
        mock_cls = MagicMock(return_value=mock_client)
        with patch("hierocode.providers.anthropic.Anthropic", mock_cls):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
                provider = _make_provider()
                with pytest.raises(ModelNotFoundError):
                    provider.generate("prompt", model="bad-model")

    def test_generate_connection_error_on_generic_exception(self):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("network timeout")
        mock_cls = MagicMock(return_value=mock_client)
        with patch("hierocode.providers.anthropic.Anthropic", mock_cls):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
                provider = _make_provider()
                with pytest.raises(ProviderConnectionError):
                    provider.generate("prompt", model="claude-haiku-4-5")


class TestAnthropicListModels:
    def test_list_models_returns_hardcoded(self):
        with patch("hierocode.providers.anthropic.Anthropic", MagicMock()):
            provider = _make_provider()
            models = provider.list_models()
        assert "claude-opus-4-7" in models
        assert "claude-sonnet-4-6" in models
        assert "claude-haiku-4-5" in models
        assert len(models) == 3

    def test_missing_sdk_raises_on_list_models(self):
        with patch("hierocode.providers.anthropic.Anthropic", None):
            provider = _make_provider()
            with pytest.raises(ProviderConnectionError, match="anthropic package not installed"):
                provider.list_models()


class TestAnthropicHealthcheck:
    def test_healthcheck_returns_true_on_success(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock()
        mock_cls = MagicMock(return_value=mock_client)
        with patch("hierocode.providers.anthropic.Anthropic", mock_cls):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
                provider = _make_provider()
                assert provider.healthcheck() is True

    def test_healthcheck_returns_false_on_exception(self):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("connection refused")
        mock_cls = MagicMock(return_value=mock_client)
        with patch("hierocode.providers.anthropic.Anthropic", mock_cls):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
                provider = _make_provider()
                assert provider.healthcheck() is False


class TestAnthropicMeta:
    def test_is_local_false(self):
        with patch("hierocode.providers.anthropic.Anthropic", MagicMock()):
            provider = _make_provider()
        assert provider.is_local() is False

    def test_missing_sdk_raises_on_generate(self):
        with patch("hierocode.providers.anthropic.Anthropic", None):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
                provider = _make_provider()
                with pytest.raises(ProviderConnectionError, match="anthropic package not installed"):
                    provider.generate("prompt", model="claude-haiku-4-5")

    def test_missing_sdk_raises_on_healthcheck(self):
        with patch("hierocode.providers.anthropic.Anthropic", None):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
                provider = _make_provider()
                with pytest.raises(ProviderConnectionError, match="anthropic package not installed"):
                    provider.healthcheck()
