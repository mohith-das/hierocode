"""Tests for active-exploration mode in ClaudeCodeCliProvider."""

from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from hierocode.models.schemas import AuthConfig, ProviderConfig
from hierocode.providers.claude_code_cli import ClaudeCodeCliProvider


@pytest.fixture()
def provider():
    config = ProviderConfig.model_construct(type="claude_code_cli", auth=AuthConfig(type="none"))
    return ClaudeCodeCliProvider("claude_pro", config)


def _completed(returncode=0, stdout="", stderr=""):
    return CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# Passive mode — no tool flags
# ---------------------------------------------------------------------------

class TestPassiveMode:
    def test_passive_mode_no_tools_flags_no_kwarg(self, provider):
        """No exploration kwarg → command has neither --allowedTools nor --disallowedTools."""
        mock_result = _completed(stdout='{"result": "ok"}')
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            provider.generate("plan this", "claude-sonnet-4-6")
            cmd = mock_run.call_args[0][0]
        assert "--allowedTools" not in cmd
        assert "--disallowedTools" not in cmd

    def test_passive_mode_no_tools_flags_explicit_passive(self, provider):
        """exploration='passive' → command has neither --allowedTools nor --disallowedTools."""
        mock_result = _completed(stdout='{"result": "ok"}')
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            provider.generate("plan this", "claude-sonnet-4-6", exploration="passive")
            cmd = mock_run.call_args[0][0]
        assert "--allowedTools" not in cmd
        assert "--disallowedTools" not in cmd


# ---------------------------------------------------------------------------
# Active mode — default tools
# ---------------------------------------------------------------------------

class TestActiveModeDefaultTools:
    def test_active_mode_default_tools(self, provider):
        """exploration='active' with no allowed_tools → --allowedTools Read,Grep,Glob."""
        mock_result = _completed(stdout='{"result": "ok"}')
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            provider.generate("plan this", "claude-sonnet-4-6", exploration="active")
            cmd = mock_run.call_args[0][0]
        assert "--allowedTools" in cmd
        idx = cmd.index("--allowedTools")
        assert cmd[idx + 1] == "Read,Grep,Glob"

    def test_active_mode_disallows_write_edit_bash(self, provider):
        """exploration='active' → --disallowedTools Write,Edit,Bash is always appended."""
        mock_result = _completed(stdout='{"result": "ok"}')
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            provider.generate("plan this", "claude-sonnet-4-6", exploration="active")
            cmd = mock_run.call_args[0][0]
        assert "--disallowedTools" in cmd
        idx = cmd.index("--disallowedTools")
        assert cmd[idx + 1] == "Write,Edit,Bash"


# ---------------------------------------------------------------------------
# Active mode — custom tools
# ---------------------------------------------------------------------------

class TestActiveModeCustomTools:
    def test_active_mode_custom_tools(self, provider):
        """exploration='active', allowed_tools=['Read', 'Bash'] → --allowedTools Read,Bash."""
        mock_result = _completed(stdout='{"result": "ok"}')
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            provider.generate(
                "plan this", "claude-sonnet-4-6",
                exploration="active", allowed_tools=["Read", "Bash"],
            )
            cmd = mock_run.call_args[0][0]
        assert "--allowedTools" in cmd
        idx = cmd.index("--allowedTools")
        assert cmd[idx + 1] == "Read,Bash"

    def test_active_mode_custom_tools_still_disallows_write_edit(self, provider):
        """Even when caller passes allowed_tools=['Bash'], --disallowedTools Write,Edit,Bash
        is still appended.  Claude Code's disallowed list takes precedence, so this is the
        correct behaviour — we always enforce the write-safe guard regardless of the allowlist.
        """
        mock_result = _completed(stdout='{"result": "ok"}')
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            provider.generate(
                "plan this", "claude-sonnet-4-6",
                exploration="active", allowed_tools=["Bash"],
            )
            cmd = mock_run.call_args[0][0]
        assert "--disallowedTools" in cmd
        idx = cmd.index("--disallowedTools")
        assert cmd[idx + 1] == "Write,Edit,Bash"
