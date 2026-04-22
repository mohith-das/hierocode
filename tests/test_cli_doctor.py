"""Tests for `hierocode doctor` — a CLI command, not a handler."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from hierocode.cli import app
from hierocode.models.schemas import (
    AuthConfig,
    HierocodeConfig,
    ProviderConfig,
    RoleRouting,
    RoutingConfig,
)


def _config_with_anthropic_no_key() -> HierocodeConfig:
    """Wizard-style config that includes `anthropic_api` even when the user only
    wants the claude_pro path. Mirrors the real-world shape that exposed the bug."""
    return HierocodeConfig(
        default_provider="local_ollama",
        providers={
            "local_ollama": ProviderConfig.model_construct(
                type="ollama",
                base_url="http://localhost:11434",
                auth=AuthConfig(type="none"),
            ),
            "claude_pro": ProviderConfig.model_construct(
                type="claude_code_cli",
                auth=AuthConfig(type="none"),
            ),
            "anthropic_api": ProviderConfig.model_construct(
                type="anthropic",
                auth=AuthConfig(type="bearer_env", env_var="ANTHROPIC_API_KEY"),
            ),
        },
        routing=RoutingConfig(
            planner=RoleRouting(provider="claude_pro", model="claude-sonnet-4-6"),
            drafter=RoleRouting(provider="local_ollama", model="llama3.2:3b"),
        ),
    )


def test_doctor_survives_provider_raising_on_healthcheck(monkeypatch):
    """Regression: `hierocode doctor` used to crash when a configured provider
    raised during healthcheck (e.g. AnthropicProvider without ANTHROPIC_API_KEY).
    The command should now report the failure per-provider and continue."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    runner = CliRunner()
    cfg = _config_with_anthropic_no_key()

    # Reachable healthchecks for the two providers that should work; the real
    # AnthropicProvider will raise ProviderConnectionError when the key is missing.
    def _fake_get_provider(name, conf):
        prov = MagicMock()
        if conf.type == "anthropic":
            from hierocode.exceptions import ProviderConnectionError
            prov.healthcheck.side_effect = ProviderConnectionError("ANTHROPIC_API_KEY not set")
        else:
            prov.healthcheck.return_value = True
        return prov

    with (
        patch("hierocode.cli.load_config", return_value=cfg),
        patch("hierocode.cli.get_provider", side_effect=_fake_get_provider),
    ):
        result = runner.invoke(app, ["doctor"])

    # Non-zero exit is fine because one provider is unreachable — but the
    # critical bit is that typer did not crash with an unhandled exception.
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"doctor crashed: {result.exception!r}\n---\n{result.output}"
    )
    out = result.output
    assert "local_ollama" in out
    assert "claude_pro" in out
    assert "anthropic_api" in out
    # The failing provider's error is surfaced somewhere in the output
    assert "ANTHROPIC_API_KEY" in out or "check failed" in out
