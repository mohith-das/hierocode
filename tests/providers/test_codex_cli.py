import subprocess
import pytest
from unittest.mock import patch

from hierocode.models.schemas import AuthConfig, ProviderConfig
from hierocode.exceptions import ProviderConnectionError
from hierocode.providers.codex_cli import CodexCliProvider

_CONFIG = ProviderConfig.model_construct(
    type="codex_cli",
    base_url=None,
    auth=AuthConfig(type="none"),
)

_PROVIDER = CodexCliProvider(name="test_codex_cli", config=_CONFIG)


def _completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class TestCodexCliGenerate:
    def test_generate_invokes_codex_cli(self):
        stdout = '{"type":"agent_message","content":"hello"}\n'
        with patch("subprocess.run", return_value=_completed(stdout=stdout)) as mock_run:
            _PROVIDER.generate("write a function", model="gpt-5-codex")
        cmd = mock_run.call_args.args[0]
        assert "codex" in cmd
        assert "exec" in cmd
        assert "--model" in cmd
        assert "gpt-5-codex" in cmd
        assert "--json" in cmd

    def test_generate_parses_agent_message(self):
        stdout = '{"type":"agent_message","content":"hello"}\n'
        with patch("subprocess.run", return_value=_completed(stdout=stdout)):
            result = _PROVIDER.generate("write a function", model="gpt-5-codex")
        assert result == "hello"

    def test_generate_multiple_jsonl_picks_terminal(self):
        stdout = (
            '{"type":"tool_call","text":"running..."}\n'
            '{"type":"tool_result","text":"done"}\n'
            '{"type":"agent_message","content":"final answer"}\n'
        )
        with patch("subprocess.run", return_value=_completed(stdout=stdout)):
            result = _PROVIDER.generate("do something", model="gpt-5")
        assert result == "final answer"

    def test_generate_handles_invalid_json(self):
        raw = "this is plain text output"
        with patch("subprocess.run", return_value=_completed(stdout=raw)):
            result = _PROVIDER.generate("do something", model="gpt-5")
        assert result == raw

    def test_cli_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(ProviderConnectionError, match="codex CLI not found on PATH"):
                _PROVIDER.generate("prompt", model="gpt-5")

    def test_cli_nonzero_exit(self):
        with patch("subprocess.run", return_value=_completed(returncode=1, stderr="auth error")):
            with pytest.raises(ProviderConnectionError, match="codex CLI failed"):
                _PROVIDER.generate("prompt", model="gpt-5")

    def test_cli_timeout(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="codex", timeout=180)):
            with pytest.raises(ProviderConnectionError, match="codex CLI timed out"):
                _PROVIDER.generate("prompt", model="gpt-5")

    def test_system_prompt_prepended_to_prompt(self):
        """codex (>=0.122) has no --system flag; system text is folded into the prompt."""
        stdout = '{"type":"agent_message","content":"ok"}\n'
        with patch("subprocess.run", return_value=_completed(stdout=stdout)) as mock_run:
            _PROVIDER.generate("prompt", model="gpt-5", system="You are a coding assistant.")
        cmd = mock_run.call_args.args[0]
        assert "--system" not in cmd
        # Last positional arg is the combined prompt text.
        combined = cmd[-1]
        assert "You are a coding assistant." in combined
        assert "prompt" in combined


class TestCodexCliHealthcheck:
    def test_healthcheck_true(self):
        with patch("subprocess.run", return_value=_completed(returncode=0)):
            assert _PROVIDER.healthcheck() is True

    def test_healthcheck_false_on_missing(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert _PROVIDER.healthcheck() is False

    def test_healthcheck_false_on_nonzero(self):
        with patch("subprocess.run", return_value=_completed(returncode=1)):
            assert _PROVIDER.healthcheck() is False

    def test_healthcheck_false_on_timeout(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="codex", timeout=5)):
            assert _PROVIDER.healthcheck() is False


class TestCodexCliListModels:
    def test_list_models_returns_hardcoded(self):
        models = _PROVIDER.list_models()
        assert "gpt-5" in models
        assert "gpt-5-codex" in models
        assert "o4-mini" in models
        assert len(models) == 3


class TestCodexCliMeta:
    def test_is_local_true(self):
        assert _PROVIDER.is_local() is True


class TestCodexCliRegressions:
    """Regressions for bugs found during live testing against codex 0.122.0."""

    def test_generate_always_passes_skip_git_repo_check(self):
        """Hierocode is often run outside a git repo (/tmp, scratch dirs). Always skip."""
        stdout = '{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}\n'
        with patch("subprocess.run", return_value=_completed(stdout=stdout)) as mock_run:
            _PROVIDER.generate("hi", model="")
        assert "--skip-git-repo-check" in mock_run.call_args.args[0]

    def test_generate_omits_model_when_default_sentinel(self):
        """model='default' / '' / None → no --model flag (codex picks tier default)."""
        stdout = '{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}\n'
        for sentinel in ("default", "DEFAULT", "", None):
            with patch("subprocess.run", return_value=_completed(stdout=stdout)) as mock_run:
                _PROVIDER.generate("hi", model=sentinel)  # type: ignore[arg-type]
            assert "--model" not in mock_run.call_args.args[0], f"unexpected --model for {sentinel!r}"

    def test_generate_never_passes_system_flag(self):
        """--system was removed in codex 0.122; regression guard."""
        stdout = '{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}\n'
        with patch("subprocess.run", return_value=_completed(stdout=stdout)) as mock_run:
            _PROVIDER.generate("hi", model="gpt-5", system="you are a helper")
        assert "--system" not in mock_run.call_args.args[0]

    def test_parse_jsonl_handles_nested_item_completed_shape(self):
        """codex 0.122+ wraps agent_message inside an item.completed event."""
        stdout = (
            '{"type":"thread.started"}\n'
            '{"type":"turn.started"}\n'
            '{"type":"item.completed","item":{"type":"agent_message","text":"final answer"}}\n'
            '{"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":5}}\n'
        )
        with patch("subprocess.run", return_value=_completed(stdout=stdout)):
            result = _PROVIDER.generate("hi", model="")
        assert result == "final answer"

    def test_generate_records_usage_from_turn_completed(self):
        stdout = (
            '{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}\n'
            '{"type":"turn.completed","usage":{"input_tokens":1234,"output_tokens":56}}\n'
        )
        with patch("subprocess.run", return_value=_completed(stdout=stdout)):
            _PROVIDER.generate("hi", model="")
        assert _PROVIDER.last_usage is not None
        assert _PROVIDER.last_usage.input_tokens == 1234
        assert _PROVIDER.last_usage.output_tokens == 56
        assert _PROVIDER.last_usage.messages == 1
