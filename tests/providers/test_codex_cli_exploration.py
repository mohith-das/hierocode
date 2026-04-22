"""Tests for active-exploration mode in CodexCliProvider."""

from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from hierocode.models.schemas import AuthConfig, ProviderConfig
from hierocode.providers.codex_cli import CodexCliProvider


@pytest.fixture()
def provider():
    config = ProviderConfig.model_construct(type="codex_cli", auth=AuthConfig(type="none"))
    return CodexCliProvider("codex_plus", config)


def _completed(returncode=0, stdout="", stderr=""):
    return CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# Passive mode — no approval flags
# ---------------------------------------------------------------------------

class TestPassiveMode:
    def test_passive_mode_no_approval_flags_no_kwarg(self, provider):
        """No exploration kwarg → command has no --approval-policy and no --sandbox."""
        jsonl = '{"type": "agent_message", "content": "done"}\n'
        mock_result = _completed(stdout=jsonl)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            provider.generate("plan this", "gpt-5-codex")
            cmd = mock_run.call_args[0][0]
        assert "--approval-policy" not in cmd
        assert "--sandbox" not in cmd

    def test_passive_mode_no_approval_flags_explicit_passive(self, provider):
        """exploration='passive' → command has no --approval-policy and no --sandbox."""
        jsonl = '{"type": "agent_message", "content": "done"}\n'
        mock_result = _completed(stdout=jsonl)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            provider.generate("plan this", "gpt-5-codex", exploration="passive")
            cmd = mock_run.call_args[0][0]
        assert "--approval-policy" not in cmd
        assert "--sandbox" not in cmd


# ---------------------------------------------------------------------------
# Active mode — approval + sandbox flags
# ---------------------------------------------------------------------------

class TestActiveMode:
    def test_active_mode_sets_readonly_sandbox(self, provider):
        """exploration='active' → --sandbox read-only only.

        Earlier versions also passed --approval-policy, but codex 0.122+ doesn't
        accept that flag; exec mode is non-interactive by default anyway.
        """
        jsonl = '{"type": "agent_message", "content": "done"}\n'
        mock_result = _completed(stdout=jsonl)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            provider.generate("plan this", "gpt-5-codex", exploration="active")
            cmd = mock_run.call_args[0][0]

        assert "--approval-policy" not in cmd
        assert "--sandbox" in cmd
        sb_idx = cmd.index("--sandbox")
        assert cmd[sb_idx + 1] == "read-only"

    def test_active_mode_ignores_allowed_tools_on_codex(self, provider):
        """allowed_tools kwarg is accepted without error but has no effect on the command.

        Codex CLI does not expose per-tool allowlisting via --allowedTools the same way
        Claude Code does.  The kwarg is silently ignored — it is accepted for API symmetry
        and may be wired to a Codex-native mechanism in a future release.
        """
        jsonl = '{"type": "agent_message", "content": "done"}\n'
        mock_result = _completed(stdout=jsonl)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            # Must not raise, and must not insert --allowedTools into the command.
            provider.generate(
                "plan this", "gpt-5-codex",
                exploration="active", allowed_tools=["Read", "Grep"],
            )
            cmd = mock_run.call_args[0][0]

        assert "--allowedTools" not in cmd
        assert "--sandbox" in cmd
