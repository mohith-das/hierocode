import subprocess
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from hierocode.exceptions import ProviderConnectionError
from hierocode.models.schemas import AuthConfig, ProviderConfig
from hierocode.providers.claude_code_cli import ClaudeCodeCliProvider


@pytest.fixture()
def provider():
    config = ProviderConfig.model_construct(type="claude_code_cli", auth=AuthConfig(type="none"))
    return ClaudeCodeCliProvider("claude_pro", config)


def _completed(returncode=0, stdout="", stderr=""):
    return CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# generate — happy paths
# ---------------------------------------------------------------------------

def test_generate_invokes_claude_cli(provider):
    """Command list must include the expected flags."""
    mock_result = _completed(stdout='{"result": "hello"}')
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        provider.generate("my prompt", "claude-sonnet-4-6")
        cmd = mock_run.call_args[0][0]
    assert "claude" in cmd
    assert "-p" in cmd
    assert "--output-format" in cmd
    assert "json" in cmd
    assert "--model" in cmd
    assert "claude-sonnet-4-6" in cmd


def test_generate_returns_result_field(provider):
    """Extract the `result` key from the JSON response."""
    mock_result = _completed(stdout='{"result": "hello"}')
    with patch("subprocess.run", return_value=mock_result):
        out = provider.generate("prompt", "claude-sonnet-4-6")
    assert out == "hello"


def test_generate_handles_missing_result_field(provider):
    """Fall back to `content` key when `result` is absent."""
    mock_result = _completed(stdout='{"content": "x"}')
    with patch("subprocess.run", return_value=mock_result):
        out = provider.generate("prompt", "claude-sonnet-4-6")
    assert out == "x"


def test_generate_handles_invalid_json(provider):
    """Return raw stdout when response is not valid JSON."""
    mock_result = _completed(stdout="raw text")
    with patch("subprocess.run", return_value=mock_result):
        out = provider.generate("prompt", "claude-sonnet-4-6")
    assert out == "raw text"


# ---------------------------------------------------------------------------
# generate — error paths
# ---------------------------------------------------------------------------

def test_cli_not_found(provider):
    """FileNotFoundError from subprocess → ProviderConnectionError."""
    with patch("subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(ProviderConnectionError, match="not found on PATH"):
            provider.generate("prompt", "claude-sonnet-4-6")


def test_cli_nonzero_exit(provider):
    """Non-zero returncode → ProviderConnectionError containing stderr."""
    mock_result = _completed(returncode=1, stderr="something went wrong")
    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(ProviderConnectionError, match="something went wrong"):
            provider.generate("prompt", "claude-sonnet-4-6")


def test_cli_timeout(provider):
    """TimeoutExpired from subprocess → ProviderConnectionError."""
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=180)):
        with pytest.raises(ProviderConnectionError, match="timed out"):
            provider.generate("prompt", "claude-sonnet-4-6")


# ---------------------------------------------------------------------------
# healthcheck
# ---------------------------------------------------------------------------

def test_healthcheck_true(provider):
    """Returncode 0 from `claude --version` → True."""
    with patch("subprocess.run", return_value=_completed(returncode=0)):
        assert provider.healthcheck() is True


def test_healthcheck_false_on_missing(provider):
    """FileNotFoundError during healthcheck → False (no exception raised)."""
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert provider.healthcheck() is False


# ---------------------------------------------------------------------------
# list_models / is_local
# ---------------------------------------------------------------------------

def test_list_models_returns_hardcoded(provider):
    """list_models must return the three hardcoded model names."""
    models = provider.list_models()
    assert models == ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"]


def test_is_local_true(provider):
    """is_local must always return True."""
    assert provider.is_local() is True


# ---------------------------------------------------------------------------
# system prompt passthrough
# ---------------------------------------------------------------------------

def test_system_prompt_passed(provider):
    """system option must add --append-system-prompt and the value to the command."""
    mock_result = _completed(stdout='{"result": "ok"}')
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        provider.generate("prompt", "claude-sonnet-4-6", system="foo")
        cmd = mock_run.call_args[0][0]
    assert "--append-system-prompt" in cmd
    assert "foo" in cmd
