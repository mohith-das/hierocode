"""Tests for hierocode.config_writer."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hierocode.config_writer import ConfigWriteError, list_roles, set_role_model
from hierocode.models.schemas import HierocodeConfig


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_MINIMAL_PROVIDERS = {
    "claude_pro": {"type": "claude_code_cli"},
    "local_ollama": {"type": "ollama", "base_url": "http://localhost:11434"},
}


def _write_config(path: Path, extra: dict | None = None) -> Path:
    """Write a minimal valid config and return path."""
    data: dict = {
        "default_provider": "local_ollama",
        "providers": _MINIMAL_PROVIDERS,
        "routing": {
            "draft_model": "llama3.2:3b",
            "review_model": "claude-haiku-4-5",
        },
    }
    if extra:
        _deep_merge(data, extra)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ---------------------------------------------------------------------------
# set_role_model tests
# ---------------------------------------------------------------------------


class TestSetRoleModelWritesAndValidates:
    def test_set_role_model_writes_and_validates(self, tmp_path):
        """Writes routing.planner block and returns a valid HierocodeConfig."""
        cfg = _write_config(tmp_path / ".hierocode.yaml")
        result = set_role_model(
            "planner", "claude-sonnet-4-6", provider="claude_pro", config_path=cfg
        )

        assert isinstance(result, HierocodeConfig)
        reloaded = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        assert reloaded["routing"]["planner"] == {
            "provider": "claude_pro",
            "model": "claude-sonnet-4-6",
        }


class TestSetRoleModelReusesExistingProvider:
    def test_set_role_model_reuses_existing_provider(self, tmp_path):
        """When routing.planner already has a provider, reuses it if provider arg is omitted."""
        cfg = _write_config(
            tmp_path / ".hierocode.yaml",
            extra={"routing": {"planner": {"provider": "claude_pro", "model": "old-model"}}},
        )
        set_role_model("planner", "new_model", config_path=cfg)

        reloaded = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        assert reloaded["routing"]["planner"] == {
            "provider": "claude_pro",
            "model": "new_model",
        }


class TestSetRoleModelFallsBackToDefaultProvider:
    def test_set_role_model_falls_back_to_default_provider(self, tmp_path):
        """Uses default_provider when routing.planner has no provider and arg is omitted."""
        cfg = _write_config(tmp_path / ".hierocode.yaml")
        # Ensure no planner block exists
        data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        data["routing"].pop("planner", None)
        cfg.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

        result = set_role_model("planner", "some-model", config_path=cfg)

        reloaded = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        assert reloaded["routing"]["planner"]["provider"] == "local_ollama"
        assert isinstance(result, HierocodeConfig)


class TestSetRoleModelRejectsUnknownRole:
    def test_set_role_model_rejects_unknown_role(self, tmp_path):
        cfg = _write_config(tmp_path / ".hierocode.yaml")
        with pytest.raises(ConfigWriteError, match="Invalid role 'wizard'"):
            set_role_model("wizard", "some-model", provider="claude_pro", config_path=cfg)


class TestSetRoleModelRejectsUnknownProvider:
    def test_set_role_model_rejects_unknown_provider(self, tmp_path):
        """Raises ConfigWriteError and leaves file unchanged when provider is not configured."""
        cfg = _write_config(tmp_path / ".hierocode.yaml")
        original_bytes = cfg.read_bytes()

        with pytest.raises(ConfigWriteError, match="Provider 'not_in_providers' is not configured"):
            set_role_model("planner", "some-model", provider="not_in_providers", config_path=cfg)

        assert cfg.read_bytes() == original_bytes


class TestSetRoleModelMissingConfigFileRaises:
    def test_set_role_model_missing_config_file_raises(self, tmp_path):
        nonexistent = tmp_path / "no_such_file.yaml"
        with pytest.raises(ConfigWriteError, match="Config not found at"):
            set_role_model("planner", "some-model", provider="claude_pro", config_path=nonexistent)


class TestSetRoleModelRejectsInvalidResult:
    def test_set_role_model_rejects_invalid_result(self, tmp_path):
        """Raises ConfigWriteError and leaves file unchanged when resulting config is invalid."""
        cfg = _write_config(tmp_path / ".hierocode.yaml")
        # Corrupt providers so that Pydantic validation fails
        data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        data["providers"]["claude_pro"] = {"type": "not_a_valid_type"}
        cfg.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        original_bytes = cfg.read_bytes()

        with pytest.raises(ConfigWriteError, match="Resulting config is invalid"):
            set_role_model("planner", "any-model", provider="claude_pro", config_path=cfg)

        assert cfg.read_bytes() == original_bytes


class TestSetRoleModelAtomicTmpCleanup:
    def test_set_role_model_atomic_tmp_cleanup(self, tmp_path):
        """No .yaml.tmp file remains after a successful write."""
        cfg = _write_config(tmp_path / ".hierocode.yaml")
        set_role_model("drafter", "qwen2.5-coder:7b", provider="local_ollama", config_path=cfg)

        tmp = cfg.with_suffix(cfg.suffix + ".tmp")
        assert not tmp.exists()


class TestSetRoleModelPreservesOtherFields:
    def test_set_role_model_preserves_other_fields(self, tmp_path):
        """Extra providers, parallelization, policy fields survive the update."""
        extra = {
            "providers": {
                "extra_prov": {"type": "openai_compatible", "base_url": "http://x/v1"},
            },
            "parallelization": {"default_strategy": "aggressive", "max_local_workers": 8},
            "policy": {"max_revisions_per_unit": 5},
        }
        cfg = _write_config(tmp_path / ".hierocode.yaml", extra=extra)
        set_role_model("reviewer", "review-model", provider="local_ollama", config_path=cfg)

        reloaded = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        assert "extra_prov" in reloaded["providers"]
        assert reloaded["parallelization"]["default_strategy"] == "aggressive"
        assert reloaded["parallelization"]["max_local_workers"] == 8
        assert reloaded["policy"]["max_revisions_per_unit"] == 5


# ---------------------------------------------------------------------------
# list_roles tests
# ---------------------------------------------------------------------------


class TestListRolesExplicit:
    def test_list_roles_explicit(self, tmp_path):
        """Planner with an explicit RoleRouting block shows source='explicit'."""
        cfg = _write_config(
            tmp_path / ".hierocode.yaml",
            extra={"routing": {"planner": {"provider": "claude_pro", "model": "claude-haiku-4-5"}}},
        )
        roles = list_roles(config_path=cfg)

        assert roles["planner"]["source"] == "explicit"
        assert roles["planner"]["provider"] == "claude_pro"
        assert roles["planner"]["model"] == "claude-haiku-4-5"


class TestListRolesLegacyForDrafter:
    def test_list_roles_legacy_for_drafter(self, tmp_path):
        """Drafter without RoleRouting block falls back to draft_model + default_provider."""
        cfg = _write_config(tmp_path / ".hierocode.yaml")  # has draft_model in routing

        roles = list_roles(config_path=cfg)

        assert roles["drafter"]["source"] == "legacy"
        assert roles["drafter"]["provider"] == "local_ollama"
        assert roles["drafter"]["model"] == "llama3.2:3b"


class TestListRolesLegacyForPlannerAndReviewer:
    def test_list_roles_legacy_for_planner_and_reviewer(self, tmp_path):
        """Planner and reviewer without RoleRouting blocks use review_model + default_provider."""
        cfg = _write_config(tmp_path / ".hierocode.yaml")  # has review_model in routing

        roles = list_roles(config_path=cfg)

        for role in ("planner", "reviewer"):
            assert roles[role]["source"] == "legacy", f"Expected legacy for {role}"
            assert roles[role]["provider"] == "local_ollama"
            assert roles[role]["model"] == "claude-haiku-4-5"


class TestListRolesDefaultWhenConfigEmpty:
    def test_list_roles_default_when_config_empty(self, tmp_path):
        """Even an empty config returns all three role keys."""
        cfg = tmp_path / ".hierocode.yaml"
        cfg.write_text(
            yaml.safe_dump(
                {"default_provider": "local_ollama", "providers": _MINIMAL_PROVIDERS},
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        roles = list_roles(config_path=cfg)

        assert set(roles.keys()) == {"planner", "drafter", "reviewer"}
        for row in roles.values():
            assert row["source"] in {"legacy", "default"}
