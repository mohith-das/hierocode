import pytest
from unittest.mock import MagicMock, patch

from hierocode.models.schemas import AuthConfig, ProviderConfig

_CONFIG = ProviderConfig.model_construct(
    type="anthropic",
    base_url=None,
    auth=AuthConfig(type="bearer_env", env_var="ANTHROPIC_API_KEY"),
)

_LONG = "x" * 2000
_SHORT = "short"


def _make_provider(config=_CONFIG):
    from hierocode.providers.anthropic import AnthropicProvider

    return AnthropicProvider(name="test_anthropic_cache", config=config)


def _patched_provider(response_text="hi"):
    """Return (mock_cls, mock_client) with a canned response."""
    mock_content = MagicMock(text=response_text)
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    mock_cls = MagicMock(return_value=mock_client)
    return mock_cls, mock_client


@pytest.fixture()
def api_key(monkeypatch):
    """Set a fake API key for all caching tests."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")


class TestSystemCaching:
    def test_short_system_not_cached(self, api_key):
        """system < 1024 chars with cache=True → plain string, no cache_control."""
        mock_cls, mock_client = _patched_provider()
        with patch("hierocode.providers.anthropic.Anthropic", mock_cls):
            provider = _make_provider()
            provider.generate("prompt", model="claude-haiku-4-5", system=_SHORT, cache=True)

        system = mock_client.messages.create.call_args.kwargs["system"]
        assert isinstance(system, str)
        assert _SHORT in system

    def test_long_system_cached_by_default(self, api_key):
        """system >= 1024 chars with no cache kwarg → list with cache_control."""
        mock_cls, mock_client = _patched_provider()
        with patch("hierocode.providers.anthropic.Anthropic", mock_cls):
            provider = _make_provider()
            provider.generate("prompt", model="claude-haiku-4-5", system=_LONG)

        system = mock_client.messages.create.call_args.kwargs["system"]
        assert isinstance(system, list)
        assert len(system) == 1
        assert system[0]["type"] == "text"
        assert system[0]["text"] == _LONG
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    def test_long_system_not_cached_when_cache_false(self, api_key):
        """system >= 1024 chars with cache=False → plain string."""
        mock_cls, mock_client = _patched_provider()
        with patch("hierocode.providers.anthropic.Anthropic", mock_cls):
            provider = _make_provider()
            provider.generate("prompt", model="claude-haiku-4-5", system=_LONG, cache=False)

        system = mock_client.messages.create.call_args.kwargs["system"]
        assert isinstance(system, str)
        assert system == _LONG

    def test_no_system_no_caching(self, api_key):
        """No system provided → 'system' key absent from create kwargs."""
        mock_cls, mock_client = _patched_provider()
        with patch("hierocode.providers.anthropic.Anthropic", mock_cls):
            provider = _make_provider()
            provider.generate("prompt", model="claude-haiku-4-5")

        kwargs = mock_client.messages.create.call_args.kwargs
        assert "system" not in kwargs


class TestUserPrefixCaching:
    def test_cache_user_prefix_long(self, api_key):
        """Long cache_user_prefix with cache=True → messages use content list."""
        mock_cls, mock_client = _patched_provider()
        with patch("hierocode.providers.anthropic.Anthropic", mock_cls):
            provider = _make_provider()
            provider.generate(
                "task part",
                model="claude-haiku-4-5",
                cache=True,
                cache_user_prefix=_LONG,
            )

        messages = mock_client.messages.create.call_args.kwargs["messages"]
        assert len(messages) == 1
        msg = messages[0]
        assert msg["role"] == "user"
        content = msg["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        # First block: cached prefix
        assert content[0]["type"] == "text"
        assert content[0]["text"] == _LONG
        assert content[0]["cache_control"] == {"type": "ephemeral"}
        # Second block: uncached prompt
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "task part"
        assert "cache_control" not in content[1]

    def test_cache_user_prefix_short_not_applied(self, api_key):
        """Short cache_user_prefix → plain string content, no content list."""
        mock_cls, mock_client = _patched_provider()
        with patch("hierocode.providers.anthropic.Anthropic", mock_cls):
            provider = _make_provider()
            provider.generate(
                "task part",
                model="claude-haiku-4-5",
                cache=True,
                cache_user_prefix="short prefix",
            )

        messages = mock_client.messages.create.call_args.kwargs["messages"]
        assert messages == [{"role": "user", "content": "task part"}]

    def test_cache_user_prefix_ignored_when_cache_false(self, api_key):
        """Long prefix but cache=False → plain prompt string, no content list."""
        mock_cls, mock_client = _patched_provider()
        with patch("hierocode.providers.anthropic.Anthropic", mock_cls):
            provider = _make_provider()
            provider.generate(
                "task part",
                model="claude-haiku-4-5",
                cache=False,
                cache_user_prefix=_LONG,
            )

        messages = mock_client.messages.create.call_args.kwargs["messages"]
        assert messages == [{"role": "user", "content": "task part"}]


class TestCachingInteractionsWithOptions:
    def test_json_mode_still_applied_with_caching(self, api_key):
        """json_mode=True with long system + cache=True → JSON instruction in cached system."""
        mock_cls, mock_client = _patched_provider()
        with patch("hierocode.providers.anthropic.Anthropic", mock_cls):
            provider = _make_provider()
            provider.generate(
                "prompt",
                model="claude-haiku-4-5",
                system=_LONG,
                json_mode=True,
                cache=True,
            )

        system = mock_client.messages.create.call_args.kwargs["system"]
        # System should be a cached list because combined text >= 1024 chars
        assert isinstance(system, list)
        assert system[0]["cache_control"] == {"type": "ephemeral"}
        # JSON instruction must be present in the system text
        assert "JSON" in system[0]["text"]

    def test_returns_text_content_as_before(self, api_key):
        """Response text returned correctly regardless of caching path."""
        mock_cls, mock_client = _patched_provider(response_text="hi")
        with patch("hierocode.providers.anthropic.Anthropic", mock_cls):
            provider = _make_provider()
            result = provider.generate("prompt", model="claude-haiku-4-5", system=_LONG)
        assert result == "hi"

    def test_max_tokens_forwarded_through_cache_path(self, api_key):
        """max_tokens kwarg still reaches the API call when caching is active."""
        mock_cls, mock_client = _patched_provider()
        with patch("hierocode.providers.anthropic.Anthropic", mock_cls):
            provider = _make_provider()
            provider.generate(
                "prompt",
                model="claude-haiku-4-5",
                system=_LONG,
                cache=True,
                max_tokens=512,
            )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 512
