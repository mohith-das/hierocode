"""Tests for hierocode.cli_wizard."""

from __future__ import annotations

import pytest
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_httpx_client_mock(reachable: bool, models: list[dict] | None = None) -> MagicMock:
    """Return a mock httpx.Client context-compatible object."""
    client = MagicMock()
    if reachable:
        root_resp = MagicMock()
        root_resp.status_code = 200

        tags_resp = MagicMock()
        tags_resp.json.return_value = {"models": models or []}

        def _get(url: str, **_kw):
            if url.endswith("/api/tags"):
                return tags_resp
            return root_resp

        client.get.side_effect = _get
    else:
        client.get.side_effect = Exception("connection refused")
    return client


def _make_subprocess_result(returncode: int = 0) -> MagicMock:
    result = MagicMock()
    result.returncode = returncode
    return result


# ---------------------------------------------------------------------------
# detect_environment tests
# ---------------------------------------------------------------------------

class TestDetectAllAbsent:
    def test_detect_all_absent(self, monkeypatch):
        """When nothing is available the result is all-False / empty."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with (
            patch("shutil.which", return_value=None),
            patch("httpx.Client", return_value=_make_httpx_client_mock(False)),
            patch("hierocode.cli_wizard.get_total_ram_gb", return_value=16.0),
        ):
            import warnings
            from hierocode.cli_wizard import detect_environment

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = detect_environment()

        assert result.claude_cli_available is False
        assert result.codex_cli_available is False
        assert result.anthropic_api_key_present is False
        assert result.ollama_reachable is False
        assert result.ollama_models == []
        assert result.recommended_planner_type == "ollama"


class TestDetectClaudeCliWins:
    def test_detect_claude_cli_wins(self, monkeypatch):
        """When claude CLI is present it becomes the recommended planner."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        def _which(binary):
            return f"/usr/local/bin/{binary}" if binary == "claude" else None

        with (
            patch("shutil.which", side_effect=_which),
            patch(
                "subprocess.run",
                return_value=_make_subprocess_result(0),
            ),
            patch("httpx.Client", return_value=_make_httpx_client_mock(False)),
            patch("hierocode.cli_wizard.get_total_ram_gb", return_value=16.0),
        ):
            from hierocode.cli_wizard import detect_environment

            result = detect_environment()

        assert result.claude_cli_available is True
        assert result.recommended_planner_type == "claude_code_cli"


class TestDetectCodexWithoutClaude:
    def test_detect_codex_without_claude(self, monkeypatch):
        """When codex is present but claude is absent, codex wins."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        def _which(binary):
            return f"/usr/local/bin/{binary}" if binary == "codex" else None

        with (
            patch("shutil.which", side_effect=_which),
            patch(
                "subprocess.run",
                return_value=_make_subprocess_result(0),
            ),
            patch("httpx.Client", return_value=_make_httpx_client_mock(False)),
            patch("hierocode.cli_wizard.get_total_ram_gb", return_value=16.0),
        ):
            from hierocode.cli_wizard import detect_environment

            result = detect_environment()

        assert result.claude_cli_available is False
        assert result.codex_cli_available is True
        assert result.recommended_planner_type == "codex_cli"


class TestDetectAnthropicKeyWhenCLIsAbsent:
    def test_detect_anthropic_key_when_clis_absent(self, monkeypatch):
        """ANTHROPIC_API_KEY wins when both CLIs are absent."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

        with (
            patch("shutil.which", return_value=None),
            patch("httpx.Client", return_value=_make_httpx_client_mock(False)),
            patch("hierocode.cli_wizard.get_total_ram_gb", return_value=16.0),
        ):
            from hierocode.cli_wizard import detect_environment

            result = detect_environment()

        assert result.claude_cli_available is False
        assert result.codex_cli_available is False
        assert result.anthropic_api_key_present is True
        assert result.recommended_planner_type == "anthropic"


class TestPriorityClaudeOverCodexOverKey:
    def test_priority_claude_over_codex_over_key(self, monkeypatch):
        """When all three planners are available, claude_code_cli wins."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

        def _which(binary):
            return f"/usr/local/bin/{binary}"

        with (
            patch("shutil.which", side_effect=_which),
            patch("subprocess.run", return_value=_make_subprocess_result(0)),
            patch("httpx.Client", return_value=_make_httpx_client_mock(False)),
            patch("hierocode.cli_wizard.get_total_ram_gb", return_value=16.0),
        ):
            from hierocode.cli_wizard import detect_environment

            result = detect_environment()

        assert result.recommended_planner_type == "claude_code_cli"


class TestOllamaReachablePopulatesModels:
    def test_ollama_reachable_populates_models(self, monkeypatch):
        """When Ollama is reachable, model list is populated."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        ollama_models = [
            {"name": "llama3.2:3b"},
            {"name": "qwen2.5-coder:7b"},
        ]
        mock_client = _make_httpx_client_mock(True, models=ollama_models)

        with (
            patch("shutil.which", return_value=None),
            patch("httpx.Client", return_value=mock_client),
            patch("hierocode.cli_wizard.get_total_ram_gb", return_value=16.0),
        ):
            import warnings
            from hierocode.cli_wizard import detect_environment

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = detect_environment()

        assert result.ollama_reachable is True
        assert "llama3.2:3b" in result.ollama_models
        assert "qwen2.5-coder:7b" in result.ollama_models


class TestDrafterRecommendationByRam:
    @pytest.mark.parametrize(
        "ram_gb, expected_substring",
        [
            (8.0, "llama3.2:1b"),
            (16.0, "llama3.2:3b"),
            (32.0, "qwen2.5-coder:7b"),
            (64.0, "qwen2.5-coder:14b"),
        ],
    )
    def test_drafter_recommendation_by_ram(self, ram_gb: float, expected_substring: str):
        """Drafter model selection follows RAM thresholds."""
        from hierocode.cli_wizard import _pick_drafter

        result = _pick_drafter(ram_gb, [])
        assert result == expected_substring


class TestDrafterPrefersInstalledModel:
    def test_drafter_prefers_installed_model(self):
        """When preferred model is already pulled, use it exactly."""
        from hierocode.cli_wizard import _pick_drafter

        # RAM=32 GB → preferred is qwen2.5-coder:7b
        result = _pick_drafter(32.0, ["qwen2.5-coder:7b", "llama3.2:3b"])
        assert result == "qwen2.5-coder:7b"


# ---------------------------------------------------------------------------
# build_config_yaml tests
# ---------------------------------------------------------------------------

def _minimal_detection(**overrides):
    from hierocode.cli_wizard import DetectionResult

    defaults = dict(
        claude_cli_available=False,
        codex_cli_available=False,
        anthropic_api_key_present=False,
        ollama_reachable=True,
        ollama_models=["llama3.2:3b"],
        total_ram_gb=16.0,
        recommended_drafter_model="llama3.2:3b",
        recommended_planner_type="claude_code_cli",
    )
    defaults.update(overrides)
    return DetectionResult(**defaults)


class TestBuildConfigYamlIsParseable:
    def test_build_config_yaml_is_parseable(self):
        """Output must be valid YAML with the required top-level keys."""
        from hierocode.cli_wizard import build_config_yaml

        detection = _minimal_detection()
        yaml_str = build_config_yaml(detection)
        parsed = yaml.safe_load(yaml_str)

        assert isinstance(parsed, dict)
        assert "providers" in parsed
        assert "routing" in parsed
        assert "policy" in parsed


class TestBuildConfigYamlPlannerMatchesDetection:
    def test_build_config_yaml_planner_matches_detection(self):
        """When recommended is claude_code_cli, routing.planner.provider == 'claude_pro'."""
        from hierocode.cli_wizard import build_config_yaml

        detection = _minimal_detection(recommended_planner_type="claude_code_cli")
        parsed = yaml.safe_load(build_config_yaml(detection))

        assert parsed["routing"]["planner"]["provider"] == "claude_pro"

    def test_build_config_yaml_codex_maps_to_codex_plus(self):
        """When recommended is codex_cli, routing.planner.provider == 'codex_plus'."""
        from hierocode.cli_wizard import build_config_yaml

        detection = _minimal_detection(recommended_planner_type="codex_cli")
        parsed = yaml.safe_load(build_config_yaml(detection))

        assert parsed["routing"]["planner"]["provider"] == "codex_plus"

    def test_build_config_yaml_anthropic_maps_to_anthropic_api(self):
        """When recommended is anthropic, routing.planner.provider == 'anthropic_api'."""
        from hierocode.cli_wizard import build_config_yaml

        detection = _minimal_detection(recommended_planner_type="anthropic")
        parsed = yaml.safe_load(build_config_yaml(detection))

        assert parsed["routing"]["planner"]["provider"] == "anthropic_api"


class TestBuildConfigYamlPassesPydanticValidation:
    def test_build_config_yaml_passes_pydantic_validation(self):
        """Generated YAML must deserialize into HierocodeConfig without error."""
        from hierocode.cli_wizard import build_config_yaml
        from hierocode.models.schemas import HierocodeConfig

        detection = _minimal_detection()
        yaml_str = build_config_yaml(detection)
        parsed = yaml.safe_load(yaml_str)

        # Remove keys not in HierocodeConfig to avoid unexpected-field errors.
        config = HierocodeConfig(**parsed)
        assert config.providers  # at least one provider present


# ---------------------------------------------------------------------------
# run_wizard tests
# ---------------------------------------------------------------------------

class TestRunWizardWriteFalseReturnsYaml:
    def test_run_wizard_write_false_returns_yaml(self, monkeypatch):
        """write=False must return non-empty yaml and not touch disk."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        written_paths: list[Path] = []

        with (
            patch("shutil.which", return_value=None),
            patch("httpx.Client", return_value=_make_httpx_client_mock(False)),
            patch("hierocode.cli_wizard.get_total_ram_gb", return_value=16.0),
        ):
            import warnings
            from hierocode.cli_wizard import run_wizard

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                detection, yaml_str = run_wizard(write=False)

        assert yaml_str  # non-empty
        assert not written_paths  # nothing written


class TestRunWizardRefusesOverwriteWithoutForce:
    def test_run_wizard_refuses_overwrite_without_force(self, tmp_path, monkeypatch):
        """Raises ConfigError when config exists and force=False."""
        config_file = tmp_path / ".hierocode.yaml"
        config_file.write_text("existing: true\n", encoding="utf-8")

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr("hierocode.cli_wizard.get_config_path", lambda: config_file)

        with (
            patch("shutil.which", return_value=None),
            patch("httpx.Client", return_value=_make_httpx_client_mock(False)),
            patch("hierocode.cli_wizard.get_total_ram_gb", return_value=16.0),
        ):
            import warnings
            from hierocode.exceptions import ConfigError
            from hierocode.cli_wizard import run_wizard

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(ConfigError):
                    run_wizard(write=True, force=False)

        # File should be unchanged.
        assert config_file.read_text() == "existing: true\n"


class TestRunWizardForceOverwrites:
    def test_run_wizard_force_overwrites(self, tmp_path, monkeypatch):
        """force=True must overwrite an existing config with new content."""
        config_file = tmp_path / ".hierocode.yaml"
        config_file.write_text("existing: true\n", encoding="utf-8")

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr("hierocode.cli_wizard.get_config_path", lambda: config_file)

        with (
            patch("shutil.which", return_value=None),
            patch("httpx.Client", return_value=_make_httpx_client_mock(False)),
            patch("hierocode.cli_wizard.get_total_ram_gb", return_value=16.0),
        ):
            import warnings
            from hierocode.cli_wizard import run_wizard

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                detection, yaml_str = run_wizard(write=True, force=True)

        new_content = config_file.read_text()
        assert new_content != "existing: true\n"
        assert "Generated by hierocode init --wizard" in new_content
        assert yaml_str == new_content


# ---------------------------------------------------------------------------
# Active exploration — build_config_yaml tests
# ---------------------------------------------------------------------------

class TestBuildConfigYamlExploration:
    def test_build_config_yaml_omits_exploration_when_false(self):
        """active_exploration=False → rendered YAML does NOT contain 'exploration:'."""
        from hierocode.cli_wizard import build_config_yaml

        detection = _minimal_detection(
            recommended_planner_type="claude_code_cli",
        )
        detection.active_exploration = False
        yaml_str = build_config_yaml(detection)

        assert "exploration:" not in yaml_str

    def test_build_config_yaml_emits_exploration_when_true_claude(self):
        """active_exploration=True + claude_code_cli → 'exploration: active' under planner and reviewer."""
        from hierocode.cli_wizard import build_config_yaml

        detection = _minimal_detection(
            recommended_planner_type="claude_code_cli",
        )
        detection.active_exploration = True
        yaml_str = build_config_yaml(detection)

        assert "exploration: active" in yaml_str
        # Appears for both planner and reviewer — count at least 2 occurrences.
        assert yaml_str.count("exploration: active") >= 2

    def test_build_config_yaml_emits_exploration_when_true_codex(self):
        """active_exploration=True + codex_cli → 'exploration: active' in YAML."""
        from hierocode.cli_wizard import build_config_yaml

        detection = _minimal_detection(
            recommended_planner_type="codex_cli",
        )
        detection.active_exploration = True
        yaml_str = build_config_yaml(detection)

        assert "exploration: active" in yaml_str

    def test_build_config_yaml_never_sets_exploration_on_drafter(self):
        """Drafter block (local_ollama) never contains 'exploration:' — Ollama has no MCP tools."""
        from hierocode.cli_wizard import build_config_yaml
        import yaml

        detection = _minimal_detection(
            recommended_planner_type="claude_code_cli",
        )
        detection.active_exploration = True
        parsed = yaml.safe_load(build_config_yaml(detection))

        drafter_cfg = parsed.get("routing", {}).get("drafter", {})
        assert "exploration" not in drafter_cfg

    def test_build_config_yaml_omits_exploration_for_ollama_planner(self):
        """active_exploration=True but planner=ollama → no exploration field (Ollama unsupported)."""
        from hierocode.cli_wizard import build_config_yaml

        detection = _minimal_detection(
            recommended_planner_type="ollama",
            recommended_drafter_model="llama3.2:3b",
        )
        detection.active_exploration = True
        yaml_str = build_config_yaml(detection)

        assert "exploration:" not in yaml_str

    def test_build_config_yaml_with_exploration_is_valid_pydantic(self):
        """Generated YAML with exploration: active must still pass HierocodeConfig validation."""
        from hierocode.cli_wizard import build_config_yaml
        from hierocode.models.schemas import HierocodeConfig

        detection = _minimal_detection(
            recommended_planner_type="claude_code_cli",
        )
        detection.active_exploration = True
        parsed = yaml.safe_load(build_config_yaml(detection))

        config = HierocodeConfig(**parsed)
        assert config.routing.planner is not None
        assert config.routing.planner.exploration == "active"
        assert config.routing.reviewer is not None
        assert config.routing.reviewer.exploration == "active"


# ---------------------------------------------------------------------------
# RoleRouting schema tests
# ---------------------------------------------------------------------------

class TestRoleRoutingExplorationSchema:
    def test_role_routing_defaults_exploration_to_passive(self):
        """RoleRouting without exploration field defaults to 'passive'."""
        from hierocode.models.schemas import RoleRouting

        r = RoleRouting(provider="local_ollama", model="llama3.2:3b")
        assert r.exploration == "passive"

    def test_role_routing_accepts_active_exploration(self):
        """RoleRouting with exploration='active' is valid."""
        from hierocode.models.schemas import RoleRouting

        r = RoleRouting(provider="claude_pro", model="claude-sonnet-4-6", exploration="active")
        assert r.exploration == "active"

    def test_role_routing_accepts_allowed_tools_list(self):
        """RoleRouting with allowed_tools list is valid; defaults to None."""
        from hierocode.models.schemas import RoleRouting

        r_none = RoleRouting(provider="claude_pro", model="claude-sonnet-4-6")
        assert r_none.allowed_tools is None

        r_tools = RoleRouting(
            provider="claude_pro",
            model="claude-sonnet-4-6",
            exploration="active",
            allowed_tools=["Read", "Grep"],
        )
        assert r_tools.allowed_tools == ["Read", "Grep"]


# ---------------------------------------------------------------------------
# auto_apply — build_config_yaml tests
# ---------------------------------------------------------------------------


class TestBuildConfigYamlAutoApply:
    def test_build_config_yaml_omits_auto_apply_when_false(self):
        """detection.auto_apply=False → rendered YAML does NOT contain 'auto_apply:'."""
        from hierocode.cli_wizard import build_config_yaml

        detection = _minimal_detection()
        detection.auto_apply = False
        yaml_str = build_config_yaml(detection)

        assert "auto_apply:" not in yaml_str

    def test_build_config_yaml_emits_auto_apply_when_true(self):
        """detection.auto_apply=True → YAML under policy: contains 'auto_apply: true'."""
        from hierocode.cli_wizard import build_config_yaml

        detection = _minimal_detection()
        detection.auto_apply = True
        yaml_str = build_config_yaml(detection)
        parsed = yaml.safe_load(yaml_str)

        assert "auto_apply:" in yaml_str
        assert parsed["policy"]["auto_apply"] is True

    def test_build_config_yaml_auto_apply_true_passes_pydantic(self):
        """YAML with auto_apply: true must pass HierocodeConfig validation."""
        from hierocode.cli_wizard import build_config_yaml
        from hierocode.models.schemas import HierocodeConfig

        detection = _minimal_detection()
        detection.auto_apply = True
        parsed = yaml.safe_load(build_config_yaml(detection))

        config = HierocodeConfig(**parsed)
        assert config.policy.auto_apply is True


# ---------------------------------------------------------------------------
# auto_apply — run_wizard TTY prompt tests
# ---------------------------------------------------------------------------


class TestRunWizardAutoApplyPrompt:
    def test_run_wizard_prompts_for_auto_apply_when_tty(self, monkeypatch, tmp_path):
        """isatty True + input 'y' → detection.auto_apply is True."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config_file = tmp_path / ".hierocode.yaml"
        monkeypatch.setattr("hierocode.cli_wizard.get_config_path", lambda: config_file)

        # Both wizard prompts (active_exploration skipped — planner=ollama, not cli)
        # so only the auto_apply prompt fires.
        call_count = {"n": 0}

        def _fake_input(prompt: str) -> str:
            call_count["n"] += 1
            return "y"

        with (
            patch("shutil.which", return_value=None),
            patch("httpx.Client", return_value=_make_httpx_client_mock(False)),
            patch("hierocode.cli_wizard.get_total_ram_gb", return_value=16.0),
            patch("sys.stdin") as mock_stdin,
            patch("builtins.input", side_effect=_fake_input),
        ):
            mock_stdin.isatty.return_value = True
            import warnings
            from hierocode.cli_wizard import run_wizard

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                detection, _ = run_wizard(write=False)

        assert detection.auto_apply is True
        assert call_count["n"] >= 1

    def test_run_wizard_omits_prompt_when_not_tty(self, monkeypatch):
        """isatty False → auto_apply prompt not called; detection.auto_apply stays False."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with (
            patch("shutil.which", return_value=None),
            patch("httpx.Client", return_value=_make_httpx_client_mock(False)),
            patch("hierocode.cli_wizard.get_total_ram_gb", return_value=16.0),
            patch("sys.stdin") as mock_stdin,
            patch("builtins.input") as mock_input,
        ):
            mock_stdin.isatty.return_value = False
            import warnings
            from hierocode.cli_wizard import run_wizard

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                detection, _ = run_wizard(write=False)

        mock_input.assert_not_called()
        assert detection.auto_apply is False
