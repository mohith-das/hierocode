"""Unit tests for shell_handlers/broker_cmds.py.

All broker modules and cli_shell are mocked — no live providers or filesystem
access required. Tests will only run after W21 commits cli_shell.py.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Literal, Optional
from unittest.mock import MagicMock, patch

from rich.console import Console

from hierocode.broker.usage import UsageAccumulator
from hierocode.config_writer import ConfigWriteError

# ---------------------------------------------------------------------------
# Minimal stubs for cli_shell types (W21 hasn't landed yet).
# Once W21 merges these will be replaced by the real imports.
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field


@dataclass
class _SessionState:
    repo_root: Path
    interaction_mode: Literal["prompt", "immediate"] = "prompt"
    last_plan: Optional[object] = None
    last_diff: Optional[str] = None
    task_history: list = field(default_factory=list)
    usage: UsageAccumulator = field(default_factory=UsageAccumulator)


@dataclass
class _HandlerContext:
    args: list
    session: _SessionState
    config: object
    console: Console
    reload_config: object


# Patch the cli_shell imports before importing broker_cmds
import sys  # noqa: E402
import types  # noqa: E402

_cli_shell_mod = types.ModuleType("hierocode.cli_shell")
_cli_shell_mod.SessionState = _SessionState  # type: ignore[attr-defined]
_cli_shell_mod.HandlerContext = _HandlerContext  # type: ignore[attr-defined]
_cli_shell_mod.HandlerResult = Literal["continue", "reload_config", "exit"]  # type: ignore[attr-defined]
_cli_shell_mod.HandlerRegistry = MagicMock  # type: ignore[attr-defined]
sys.modules.setdefault("hierocode.cli_shell", _cli_shell_mod)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_config():
    """Return a minimal HierocodeConfig-like mock."""
    cfg = MagicMock()
    cfg.policy.max_revisions_per_unit = 2
    cfg.policy.max_escalations_per_task = 3
    # providers dict with one entry
    prov_cfg = MagicMock()
    prov_cfg.type = "ollama"
    cfg.providers = {"local": prov_cfg}
    return cfg


def _make_console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    con = Console(file=buf, highlight=False, markup=False)
    return con, buf


def make_ctx(args, config=None, session=None, console=None):
    """Construct a fake HandlerContext."""
    con = console or _make_console()[0]
    return _HandlerContext(
        args=args,
        session=session or _SessionState(repo_root=Path(".")),
        config=config or _minimal_config(),
        console=con,
        reload_config=lambda: config,
    )


# ---------------------------------------------------------------------------
# Module import (must happen after stub injection above)
# ---------------------------------------------------------------------------

BASE = "hierocode.shell_handlers.broker_cmds"


# ---------------------------------------------------------------------------
# /run tests
# ---------------------------------------------------------------------------

class TestHandleRun:
    def test_run_with_no_args_prints_usage(self):
        con, buf = _make_console()
        ctx = make_ctx([], console=con)
        with patch(f"{BASE}.get_route"), patch(f"{BASE}.get_provider"):
            from hierocode.shell_handlers.broker_cmds import handle_run
            result = handle_run(ctx)
        assert result == "continue"
        assert "Usage" in buf.getvalue()

    def test_run_calls_pipeline_and_stores_diff(self):
        session = _SessionState(repo_root=Path("."))
        ctx = make_ctx(["add", "retry"], session=session)

        fake_plan = MagicMock()
        fake_plan.units = [MagicMock(unit_id="u1", status="completed", diff="diff1",
                                     revision_count=0, reason=None)]
        fake_plan.task = "add retry"

        fake_result = MagicMock()
        fake_result.total_revisions = 0
        fake_result.total_escalations = 0
        fake_result.units = [MagicMock(unit_id="u1", status="completed", diff="diff1",
                                       revision_count=0, reason=None)]

        fake_profile = MagicMock()
        fake_prov = MagicMock()

        with (
            patch(f"{BASE}.get_route", return_value=("local", "m1")),
            patch(f"{BASE}.get_provider", return_value=fake_prov),
            patch(f"{BASE}.build_capacity_profile", return_value=fake_profile),
            patch(f"{BASE}.build_skeleton", return_value="skeleton"),
            patch(f"{BASE}.cache_key", return_value="key123"),
            patch(f"{BASE}.read_cached_plan", return_value=None),
            patch(f"{BASE}.generate_plan", return_value=fake_plan),
            patch(f"{BASE}.write_cached_plan"),
            patch(f"{BASE}.run_plan", return_value=fake_result),
        ):
            from hierocode.shell_handlers.broker_cmds import handle_run
            result = handle_run(ctx)

        assert result == "continue"
        assert session.last_plan is fake_plan
        assert session.last_diff == "diff1"
        assert "add retry" in session.task_history

    def test_run_uses_cached_plan_when_present(self):
        session = _SessionState(repo_root=Path("."))
        ctx = make_ctx(["cached", "task"], session=session)

        cached_plan = MagicMock()
        cached_plan.units = []

        fake_result = MagicMock()
        fake_result.total_revisions = 0
        fake_result.total_escalations = 0
        fake_result.units = []

        fake_profile = MagicMock()
        fake_prov = MagicMock()

        with (
            patch(f"{BASE}.get_route", return_value=("local", "m1")),
            patch(f"{BASE}.get_provider", return_value=fake_prov),
            patch(f"{BASE}.build_capacity_profile", return_value=fake_profile),
            patch(f"{BASE}.build_skeleton", return_value="sk"),
            patch(f"{BASE}.cache_key", return_value="k"),
            patch(f"{BASE}.read_cached_plan", return_value=cached_plan),
            patch(f"{BASE}.generate_plan") as mock_gen,
            patch(f"{BASE}.write_cached_plan"),
            patch(f"{BASE}.run_plan", return_value=fake_result),
        ):
            from hierocode.shell_handlers.broker_cmds import handle_run
            handle_run(ctx)

        mock_gen.assert_not_called()

    def test_run_task_history_grows(self):
        session = _SessionState(repo_root=Path("."))
        assert len(session.task_history) == 0

        ctx = make_ctx(["my", "task"], session=session)
        fake_plan = MagicMock()
        fake_plan.units = []
        fake_result = MagicMock()
        fake_result.total_revisions = 0
        fake_result.total_escalations = 0
        fake_result.units = []

        with (
            patch(f"{BASE}.get_route", return_value=("local", "m1")),
            patch(f"{BASE}.get_provider", return_value=MagicMock()),
            patch(f"{BASE}.build_capacity_profile", return_value=MagicMock()),
            patch(f"{BASE}.build_skeleton", return_value="s"),
            patch(f"{BASE}.cache_key", return_value="k"),
            patch(f"{BASE}.read_cached_plan", return_value=fake_plan),
            patch(f"{BASE}.write_cached_plan"),
            patch(f"{BASE}.run_plan", return_value=fake_result),
        ):
            from hierocode.shell_handlers.broker_cmds import handle_run
            handle_run(ctx)

        assert len(session.task_history) == 1
        assert session.task_history[0] == "my task"


# ---------------------------------------------------------------------------
# /plan tests
# ---------------------------------------------------------------------------

class TestHandlePlan:
    def test_plan_show_prints_last_plan(self):
        fake_plan = MagicMock()
        fake_plan.units = [MagicMock(id="u1", goal="do thing", target_files=["a.py"],
                                     acceptance="it works")]
        session = _SessionState(repo_root=Path("."), last_plan=fake_plan)
        con, buf = _make_console()
        ctx = make_ctx(["show"], session=session, console=con)

        from hierocode.shell_handlers.broker_cmds import handle_plan
        result = handle_plan(ctx)

        assert result == "continue"
        assert "u1" in buf.getvalue() or "do thing" in buf.getvalue() or "Plan" in buf.getvalue()

    def test_plan_show_when_no_last_plan_says_so(self):
        session = _SessionState(repo_root=Path("."), last_plan=None)
        con, buf = _make_console()
        ctx = make_ctx(["show"], session=session, console=con)

        from hierocode.shell_handlers.broker_cmds import handle_plan
        result = handle_plan(ctx)

        assert result == "continue"
        assert "No plan yet" in buf.getvalue()

    def test_plan_stores_last_plan(self):
        session = _SessionState(repo_root=Path("."))
        ctx = make_ctx(["add", "feature"], session=session)
        fake_plan = MagicMock()
        fake_plan.units = []

        with (
            patch(f"{BASE}.get_route", return_value=("local", "m1")),
            patch(f"{BASE}.get_provider", return_value=MagicMock()),
            patch(f"{BASE}.build_capacity_profile", return_value=MagicMock()),
            patch(f"{BASE}.build_skeleton", return_value="sk"),
            patch(f"{BASE}.cache_key", return_value="k"),
            patch(f"{BASE}.read_cached_plan", return_value=None),
            patch(f"{BASE}.generate_plan", return_value=fake_plan),
            patch(f"{BASE}.write_cached_plan"),
        ):
            from hierocode.shell_handlers.broker_cmds import handle_plan
            result = handle_plan(ctx)

        assert result == "continue"
        assert session.last_plan is fake_plan

    def test_plan_does_not_invoke_dispatcher(self):
        session = _SessionState(repo_root=Path("."))
        ctx = make_ctx(["task"], session=session)
        fake_plan = MagicMock()
        fake_plan.units = []

        with (
            patch(f"{BASE}.get_route", return_value=("local", "m1")),
            patch(f"{BASE}.get_provider", return_value=MagicMock()),
            patch(f"{BASE}.build_capacity_profile", return_value=MagicMock()),
            patch(f"{BASE}.build_skeleton", return_value="sk"),
            patch(f"{BASE}.cache_key", return_value="k"),
            patch(f"{BASE}.read_cached_plan", return_value=fake_plan),
            patch(f"{BASE}.write_cached_plan"),
            patch(f"{BASE}.run_plan") as mock_run,
        ):
            from hierocode.shell_handlers.broker_cmds import handle_plan
            handle_plan(ctx)

        mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# /estimate tests
# ---------------------------------------------------------------------------

class TestHandleEstimate:
    def test_estimate_prints_expected_fields(self):
        con, buf = _make_console()
        ctx = make_ctx(["write", "tests"], console=con)

        fake_est = MagicMock()
        fake_est.planner_kind = "anthropic_api"
        fake_est.expected_plan_units = 2
        fake_est.expected_qa_calls = 6
        fake_est.planner_input_tokens = 1000
        fake_est.planner_output_tokens = 300
        fake_est.approximate_cost_usd = 0.0015
        fake_est.approximate_message_count = None
        fake_est.notes = ["note one"]

        with (
            patch(f"{BASE}.get_route", return_value=("local", "m1")),
            patch(f"{BASE}.get_provider", return_value=MagicMock()),
            patch(f"{BASE}.build_capacity_profile", return_value=MagicMock()),
            patch(f"{BASE}.build_skeleton", return_value="sk"),
            patch(f"{BASE}.estimate_task_cost", return_value=fake_est),
        ):
            from hierocode.shell_handlers.broker_cmds import handle_estimate
            result = handle_estimate(ctx)

        assert result == "continue"
        out = buf.getvalue()
        assert "Cost estimate" in out or "planner" in out.lower() or "anthropic_api" in out

    def test_estimate_no_args_prints_usage(self):
        con, buf = _make_console()
        ctx = make_ctx([], console=con)

        from hierocode.shell_handlers.broker_cmds import handle_estimate
        result = handle_estimate(ctx)

        assert result == "continue"
        assert "Usage" in buf.getvalue()


# ---------------------------------------------------------------------------
# /draft tests
# ---------------------------------------------------------------------------

class TestHandleDraft:
    def test_draft_requires_min_args(self):
        con, buf = _make_console()
        ctx = make_ctx(["onlyonearg"], console=con)

        from hierocode.shell_handlers.broker_cmds import handle_draft
        result = handle_draft(ctx)

        assert result == "continue"
        assert "Usage" in buf.getvalue()

    def test_draft_happy_path_stores_diff(self):
        session = _SessionState(repo_root=Path("."))
        con, buf = _make_console()
        ctx = make_ctx(["add retry", "api.py"], session=session, console=con)

        fake_prov = MagicMock()
        fake_prov.generate.return_value = "new content"
        fake_profile = MagicMock()
        fake_profile.max_output_tokens = 2000
        fake_packed = MagicMock()
        fake_packed.content = "packed"

        with (
            patch(f"{BASE}.get_route", return_value=("local", "m1")),
            patch(f"{BASE}.get_provider", return_value=fake_prov),
            patch(f"{BASE}.build_capacity_profile", return_value=fake_profile),
            patch(f"{BASE}.pack_context", return_value=fake_packed),
            patch(f"{BASE}.build_drafter_prompt", return_value="prompt"),
            patch(f"{BASE}.read_file_safe", return_value="original content"),
            patch(f"{BASE}.generate_unified_diff", return_value="--- diff ---"),
        ):
            from hierocode.shell_handlers.broker_cmds import handle_draft
            result = handle_draft(ctx)

        assert result == "continue"
        assert session.last_diff == "--- diff ---"
        assert "Patch" in buf.getvalue() or "diff" in buf.getvalue()


# ---------------------------------------------------------------------------
# /review tests
# ---------------------------------------------------------------------------

class TestHandleReview:
    def test_review_requires_min_args(self):
        con, buf = _make_console()
        ctx = make_ctx(["onlyonearg"], console=con)

        from hierocode.shell_handlers.broker_cmds import handle_review
        result = handle_review(ctx)

        assert result == "continue"
        assert "Usage" in buf.getvalue()

    def test_review_prints_output(self):
        con, buf = _make_console()
        ctx = make_ctx(["check auth", "auth.py"], console=con)

        fake_prov = MagicMock()
        fake_prov.generate.return_value = "LGTM with minor issues"

        with (
            patch(f"{BASE}.get_route", return_value=("local", "m1")),
            patch(f"{BASE}.get_provider", return_value=fake_prov),
            patch(f"{BASE}.read_file_safe", return_value="file contents"),
        ):
            from hierocode.shell_handlers.broker_cmds import handle_review
            result = handle_review(ctx)

        assert result == "continue"
        assert "LGTM" in buf.getvalue() or "Review" in buf.getvalue()


# ---------------------------------------------------------------------------
# /diff tests
# ---------------------------------------------------------------------------

class TestHandleDiff:
    def test_diff_prints_last_diff(self):
        session = _SessionState(repo_root=Path("."), last_diff="--- a.py\n+++ b.py\n")
        con, buf = _make_console()
        ctx = make_ctx([], session=session, console=con)

        from hierocode.shell_handlers.broker_cmds import handle_diff
        result = handle_diff(ctx)

        assert result == "continue"
        assert "a.py" in buf.getvalue()

    def test_diff_no_last_diff(self):
        session = _SessionState(repo_root=Path("."), last_diff=None)
        con, buf = _make_console()
        ctx = make_ctx([], session=session, console=con)

        from hierocode.shell_handlers.broker_cmds import handle_diff
        result = handle_diff(ctx)

        assert result == "continue"
        assert "No diff yet" in buf.getvalue()


# ---------------------------------------------------------------------------
# /models tests
# ---------------------------------------------------------------------------

class TestHandleModels:
    def test_models_calls_list_roles(self):
        con, buf = _make_console()
        ctx = make_ctx([], console=con)

        fake_roles = {
            "planner": {"provider": "local", "model": "m1", "source": "explicit"},
            "drafter": {"provider": "local", "model": "m2", "source": "default"},
        }

        with patch(f"{BASE}.list_roles", return_value=fake_roles) as mock_lr:
            from hierocode.shell_handlers.broker_cmds import handle_models
            result = handle_models(ctx)

        mock_lr.assert_called_once()
        assert result == "continue"
        out = buf.getvalue()
        assert "planner" in out or "m1" in out


class TestHandleModelsSet:
    def test_models_set_success_returns_reload_config(self):
        con, buf = _make_console()
        ctx = make_ctx(["planner", "claude-sonnet-4-6"], console=con)

        with patch(f"{BASE}.set_role_model") as mock_set:
            from hierocode.shell_handlers.broker_cmds import handle_models_set
            result = handle_models_set(ctx)

        mock_set.assert_called_once_with("planner", "claude-sonnet-4-6", provider=None)
        assert result == "reload_config"
        assert "Updated" in buf.getvalue()

    def test_models_set_with_provider_flag(self):
        con, buf = _make_console()
        ctx = make_ctx(["drafter", "llama3", "--provider", "mylocal"], console=con)

        with patch(f"{BASE}.set_role_model") as mock_set:
            from hierocode.shell_handlers.broker_cmds import handle_models_set
            result = handle_models_set(ctx)

        mock_set.assert_called_once_with("drafter", "llama3", provider="mylocal")
        assert result == "reload_config"

    def test_models_set_failure_returns_continue_and_prints_error(self):
        con, buf = _make_console()
        ctx = make_ctx(["planner", "bad-model"], console=con)

        with patch(f"{BASE}.set_role_model", side_effect=ConfigWriteError("bad role")):
            from hierocode.shell_handlers.broker_cmds import handle_models_set
            result = handle_models_set(ctx)

        assert result == "continue"
        assert "Error" in buf.getvalue() or "bad role" in buf.getvalue()

    def test_models_set_missing_args_prints_usage(self):
        con, buf = _make_console()
        ctx = make_ctx(["onlyrole"], console=con)

        from hierocode.shell_handlers.broker_cmds import handle_models_set
        result = handle_models_set(ctx)

        assert result == "continue"
        assert "Usage" in buf.getvalue()


# ---------------------------------------------------------------------------
# /cache clear tests
# ---------------------------------------------------------------------------

class TestHandleCacheClear:
    def test_cache_clear_calls_clear_cache(self):
        con, buf = _make_console()
        ctx = make_ctx([], console=con)

        with patch(f"{BASE}.clear_cache", return_value=5) as mock_cc:
            from hierocode.shell_handlers.broker_cmds import handle_cache_clear
            result = handle_cache_clear(ctx)

        mock_cc.assert_called_once()
        assert result == "continue"
        assert "5" in buf.getvalue()

    def test_cache_clear_singular_when_one(self):
        con, buf = _make_console()
        ctx = make_ctx([], console=con)

        with patch(f"{BASE}.clear_cache", return_value=1):
            from hierocode.shell_handlers.broker_cmds import handle_cache_clear
            handle_cache_clear(ctx)

        out = buf.getvalue()
        # Should say "1 cached plan" not "1 cached plans"
        assert "plans" not in out or "1 cached plan." in out


# ---------------------------------------------------------------------------
# /doctor tests
# ---------------------------------------------------------------------------

class TestHandleDoctor:
    def test_doctor_checks_providers(self):
        con, buf = _make_console()
        cfg = _minimal_config()

        healthy_prov = MagicMock()
        healthy_prov.healthcheck.return_value = True
        cfg.providers = {"local": MagicMock(type="ollama")}

        ctx = make_ctx([], config=cfg, console=con)

        with patch(f"{BASE}.get_provider", return_value=healthy_prov):
            from hierocode.shell_handlers.broker_cmds import handle_doctor
            result = handle_doctor(ctx)

        assert result == "continue"
        healthy_prov.healthcheck.assert_called_once()

    def test_doctor_reports_unhealthy_provider(self):
        con, buf = _make_console()
        cfg = _minimal_config()
        cfg.providers = {"remote": MagicMock(type="anthropic")}

        sick_prov = MagicMock()
        sick_prov.healthcheck.return_value = False

        ctx = make_ctx([], config=cfg, console=con)

        with patch(f"{BASE}.get_provider", return_value=sick_prov):
            from hierocode.shell_handlers.broker_cmds import handle_doctor
            result = handle_doctor(ctx)

        assert result == "continue"
        out = buf.getvalue()
        assert "unreachable" in out or "WARN" in out


# ---------------------------------------------------------------------------
# /resources tests
# ---------------------------------------------------------------------------

class TestHandleResources:
    def test_resources_prints_cpu_and_ram(self):
        con, buf = _make_console()
        ctx = make_ctx([], console=con)

        fake_gpu = MagicMock()
        fake_gpu.gpu_name = "Apple Silicon (unified memory)"
        fake_gpu.vram_gb = 16.0
        fake_gpu.backend = "apple"

        with (
            patch(f"{BASE}.get_cpu_count", return_value=8),
            patch(f"{BASE}.get_total_ram_gb", return_value=32.0),
            patch(f"{BASE}.get_available_ram_gb", return_value=16.0),
            patch(f"{BASE}.probe_gpu", return_value=fake_gpu),
        ):
            from hierocode.shell_handlers.broker_cmds import handle_resources
            result = handle_resources(ctx)

        assert result == "continue"
        out = buf.getvalue()
        assert "8" in out
        assert "32" in out or "32.0" in out


# ---------------------------------------------------------------------------
# /config edit tests
# ---------------------------------------------------------------------------

class TestHandleConfigEdit:
    def test_config_edit_no_editor_env_prints_hint(self, monkeypatch):
        monkeypatch.delenv("EDITOR", raising=False)
        con, buf = _make_console()
        ctx = make_ctx([], console=con)

        from hierocode.shell_handlers.broker_cmds import handle_config_edit
        result = handle_config_edit(ctx)

        assert result == "continue"
        assert "EDITOR" in buf.getvalue()

    def test_config_edit_calls_subprocess_and_reloads(self, monkeypatch, tmp_path):
        monkeypatch.setenv("EDITOR", "vi")
        con, buf = _make_console()
        ctx = make_ctx([], console=con)

        fake_path = tmp_path / ".hierocode.yaml"
        fake_path.write_text("providers: {}")

        with (
            patch(f"{BASE}.get_config_path", return_value=fake_path),
            patch(f"{BASE}.subprocess.call") as mock_call,
        ):
            from hierocode.shell_handlers.broker_cmds import handle_config_edit
            result = handle_config_edit(ctx)

        mock_call.assert_called_once_with(["vi", str(fake_path)])
        assert result == "reload_config"


# ---------------------------------------------------------------------------
# register_all tests
# ---------------------------------------------------------------------------

class TestRegisterAll:
    def test_register_all_registers_expected_commands(self):
        registry = MagicMock()

        from hierocode.shell_handlers.broker_cmds import register_all
        register_all(registry)

        registered_names = [c[0][0] for c in registry.register.call_args_list]
        expected = {
            "run",
            "plan",
            "estimate",
            "draft",
            "review",
            "diff",
            "models",
            "models set",
            "cache clear",
            "doctor",
            "resources",
            "config edit",
        }
        assert expected.issubset(set(registered_names))

    def test_register_all_count(self):
        registry = MagicMock()

        from hierocode.shell_handlers.broker_cmds import register_all
        register_all(registry)

        assert registry.register.call_count == 12


# ---------------------------------------------------------------------------
# handle_run — escalation_confirm wiring (W27)
# ---------------------------------------------------------------------------

class TestHandleRunEscalationConfirm:
    """Verify that handle_run passes escalation_confirm based on policy.warn_before_escalation."""

    def _make_run_dependencies(self):
        """Return a dict of patches for the full run_plan pipeline."""
        fake_plan = MagicMock()
        fake_plan.units = []
        fake_result = MagicMock()
        fake_result.total_revisions = 0
        fake_result.total_escalations = 0
        fake_result.units = []
        return fake_plan, fake_result

    def test_handle_run_passes_escalation_confirm_when_policy_warn(self):
        """With warn_before_escalation=True, run_plan receives a non-None escalation_confirm."""
        session = _SessionState(repo_root=Path("."))
        cfg = _minimal_config()
        cfg.policy.warn_before_escalation = True
        ctx = make_ctx(["add tests"], config=cfg, session=session)

        fake_plan, fake_result = self._make_run_dependencies()

        with (
            patch(f"{BASE}.get_route", return_value=("local", "m1")),
            patch(f"{BASE}.get_provider", return_value=MagicMock()),
            patch(f"{BASE}.build_capacity_profile", return_value=MagicMock()),
            patch(f"{BASE}.build_skeleton", return_value="sk"),
            patch(f"{BASE}.cache_key", return_value="k"),
            patch(f"{BASE}.read_cached_plan", return_value=fake_plan),
            patch(f"{BASE}.write_cached_plan"),
            patch(f"{BASE}.run_plan", return_value=fake_result) as mock_run,
        ):
            from hierocode.shell_handlers.broker_cmds import handle_run
            handle_run(ctx)

        _, kwargs = mock_run.call_args
        assert kwargs.get("escalation_confirm") is not None

    def test_handle_run_omits_escalation_confirm_when_policy_silent(self):
        """With warn_before_escalation=False, run_plan receives escalation_confirm=None."""
        session = _SessionState(repo_root=Path("."))
        cfg = _minimal_config()
        cfg.policy.warn_before_escalation = False
        ctx = make_ctx(["add tests"], config=cfg, session=session)

        fake_plan, fake_result = self._make_run_dependencies()

        with (
            patch(f"{BASE}.get_route", return_value=("local", "m1")),
            patch(f"{BASE}.get_provider", return_value=MagicMock()),
            patch(f"{BASE}.build_capacity_profile", return_value=MagicMock()),
            patch(f"{BASE}.build_skeleton", return_value="sk"),
            patch(f"{BASE}.cache_key", return_value="k"),
            patch(f"{BASE}.read_cached_plan", return_value=fake_plan),
            patch(f"{BASE}.write_cached_plan"),
            patch(f"{BASE}.run_plan", return_value=fake_result) as mock_run,
        ):
            from hierocode.shell_handlers.broker_cmds import handle_run
            handle_run(ctx)

        _, kwargs = mock_run.call_args
        assert kwargs.get("escalation_confirm") is None
