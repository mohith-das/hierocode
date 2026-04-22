"""End-to-end tests for `hierocode run --apply` and `hierocode draft --apply`."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from hierocode.broker.dispatcher import DispatchResult, UnitResult
from hierocode.broker.patcher import ApplyResult, FilePatch, PatchAction
from hierocode.cli import app
from hierocode.models.schemas import (
    AuthConfig,
    HierocodeConfig,
    PolicyConfig,
    ProviderConfig,
    RoleRouting,
    RoutingConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_DIFF = (
    "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new\n"
)

runner = CliRunner()


def _make_unit_result(diff: str | None = _SAMPLE_DIFF) -> UnitResult:
    return UnitResult(unit_id="u1", status="completed", diff=diff)


def _make_dispatch_result(diff: str | None = _SAMPLE_DIFF) -> DispatchResult:
    return DispatchResult(task="test", units=[_make_unit_result(diff)])


def _make_file_patch() -> FilePatch:
    return FilePatch(
        path="foo.py",
        action=PatchAction.MODIFY,
        line_count_added=1,
        line_count_removed=1,
        raw_diff=_SAMPLE_DIFF,
    )


def _wired_config(auto_apply: bool = False) -> HierocodeConfig:
    """Minimal routable config so `hierocode run` doesn't bail on route resolution."""
    return HierocodeConfig(
        default_provider="local",
        providers={
            "local": ProviderConfig.model_construct(
                type="ollama",
                base_url="http://localhost:11434",
                auth=AuthConfig(type="none"),
            ),
        },
        routing=RoutingConfig(
            planner=RoleRouting(provider="local", model="m1"),
            drafter=RoleRouting(provider="local", model="m1"),
        ),
        policy=PolicyConfig(auto_apply=auto_apply),
    )


def _default_config() -> HierocodeConfig:
    return _wired_config(auto_apply=False)


def _auto_apply_config() -> HierocodeConfig:
    return _wired_config(auto_apply=True)


# ---------------------------------------------------------------------------
# run --apply
# ---------------------------------------------------------------------------


class TestRunWithApplyFlag:
    def test_run_with_apply_flag_writes_files(self):
        """run -a: apply_patch called after pipeline; exit 0; output mentions Applied."""
        patch_obj = _make_file_patch()
        apply_ok = ApplyResult(path="foo.py", status="applied")

        with (
            patch("hierocode.cli.load_config", return_value=_default_config()),
            patch("hierocode.cli.get_provider", return_value=MagicMock()),
            patch(
                "hierocode.cli.build_skeleton", return_value="skeleton"
            ),
            patch(
                "hierocode.broker.capacity.build_capacity_profile",
                return_value=MagicMock(
                    tier="small",
                    context_window=4096,
                    max_input_tokens=2048,
                    max_files_per_unit=3,
                ),
            ),
            patch("hierocode.cli.build_skeleton", return_value="skeleton"),
            patch(
                "hierocode.broker.plan_cache.read_cached_plan", return_value=None
            ),
            patch(
                "hierocode.broker.plan_cache.write_cached_plan"
            ),
            patch(
                "hierocode.broker.planner.generate_plan",
                return_value=MagicMock(units=[]),
            ),
            patch(
                "hierocode.broker.dispatcher.run_plan",
                return_value=_make_dispatch_result(),
            ),
            patch(
                "hierocode.broker.patcher.parse_diff",
                return_value=[patch_obj],
            ),
            patch(
                "hierocode.broker.patcher.apply_patch",
                return_value=apply_ok,
            ) as mock_apply,
        ):
            result = runner.invoke(app, ["run", "--task", "do stuff", "--apply"])

        assert mock_apply.called
        assert result.exit_code == 0
        assert "Applied" in result.output or "wrote" in result.output

    def test_run_without_apply_flag_does_not_write(self):
        """run without -a: apply_patch NOT called even when diff is present."""
        with (
            patch("hierocode.cli.load_config", return_value=_default_config()),
            patch("hierocode.cli.get_provider", return_value=MagicMock()),
            patch("hierocode.cli.build_skeleton", return_value="skeleton"),
            patch(
                "hierocode.broker.capacity.build_capacity_profile",
                return_value=MagicMock(
                    tier="small",
                    context_window=4096,
                    max_input_tokens=2048,
                    max_files_per_unit=3,
                ),
            ),
            patch(
                "hierocode.broker.plan_cache.read_cached_plan", return_value=None
            ),
            patch("hierocode.broker.plan_cache.write_cached_plan"),
            patch(
                "hierocode.broker.planner.generate_plan",
                return_value=MagicMock(units=[]),
            ),
            patch(
                "hierocode.broker.dispatcher.run_plan",
                return_value=_make_dispatch_result(),
            ),
            patch(
                "hierocode.broker.patcher.apply_patch",
            ) as mock_apply,
        ):
            runner.invoke(app, ["run", "--task", "do stuff"])

        mock_apply.assert_not_called()

    def test_run_respects_policy_auto_apply(self):
        """policy.auto_apply=True → apply_patch IS called even without the flag."""
        patch_obj = _make_file_patch()
        apply_ok = ApplyResult(path="foo.py", status="applied")

        with (
            patch("hierocode.cli.load_config", return_value=_auto_apply_config()),
            patch("hierocode.cli.get_provider", return_value=MagicMock()),
            patch("hierocode.cli.build_skeleton", return_value="skeleton"),
            patch(
                "hierocode.broker.capacity.build_capacity_profile",
                return_value=MagicMock(
                    tier="small",
                    context_window=4096,
                    max_input_tokens=2048,
                    max_files_per_unit=3,
                ),
            ),
            patch(
                "hierocode.broker.plan_cache.read_cached_plan", return_value=None
            ),
            patch("hierocode.broker.plan_cache.write_cached_plan"),
            patch(
                "hierocode.broker.planner.generate_plan",
                return_value=MagicMock(units=[]),
            ),
            patch(
                "hierocode.broker.dispatcher.run_plan",
                return_value=_make_dispatch_result(),
            ),
            patch("hierocode.broker.patcher.parse_diff", return_value=[patch_obj]),
            patch(
                "hierocode.broker.patcher.apply_patch", return_value=apply_ok
            ) as mock_apply,
        ):
            runner.invoke(app, ["run", "--task", "do stuff"])

        assert mock_apply.called

    def test_run_apply_reports_errors(self):
        """apply_patch returns status='error' → output contains error info."""
        patch_obj = _make_file_patch()
        apply_err = ApplyResult(path="foo.py", status="error", message="git exploded")

        with (
            patch("hierocode.cli.load_config", return_value=_default_config()),
            patch("hierocode.cli.get_provider", return_value=MagicMock()),
            patch("hierocode.cli.build_skeleton", return_value="skeleton"),
            patch(
                "hierocode.broker.capacity.build_capacity_profile",
                return_value=MagicMock(
                    tier="small",
                    context_window=4096,
                    max_input_tokens=2048,
                    max_files_per_unit=3,
                ),
            ),
            patch(
                "hierocode.broker.plan_cache.read_cached_plan", return_value=None
            ),
            patch("hierocode.broker.plan_cache.write_cached_plan"),
            patch(
                "hierocode.broker.planner.generate_plan",
                return_value=MagicMock(units=[]),
            ),
            patch(
                "hierocode.broker.dispatcher.run_plan",
                return_value=_make_dispatch_result(),
            ),
            patch("hierocode.broker.patcher.parse_diff", return_value=[patch_obj]),
            patch("hierocode.broker.patcher.apply_patch", return_value=apply_err),
        ):
            result = runner.invoke(app, ["run", "--task", "do stuff", "--apply"])

        assert "git exploded" in result.output or "errors" in result.output


# ---------------------------------------------------------------------------
# draft --apply
# ---------------------------------------------------------------------------


class TestDraftWithApplyFlag:
    def test_draft_with_apply_flag_writes_file(self):
        """draft -a: apply_patch called after diff is produced."""
        patch_obj = _make_file_patch()
        apply_ok = ApplyResult(path="foo.py", status="applied")

        with (
            patch("hierocode.cli.load_config", return_value=_default_config()),
            patch("hierocode.cli.get_provider", return_value=MagicMock()),
            patch(
                "hierocode.broker.capacity.build_capacity_profile",
                return_value=MagicMock(
                    tier="small",
                    context_window=4096,
                    max_input_tokens=2048,
                    max_files_per_unit=3,
                ),
            ),
            patch(
                "hierocode.broker.budget.pack_context",
                return_value=MagicMock(content="packed"),
            ),
            patch(
                "hierocode.broker.prompts.build_drafter_prompt",
                return_value="prompt text",
            ),
            patch(
                "hierocode.cli.get_provider",
                return_value=MagicMock(
                    generate=MagicMock(return_value="new content")
                ),
            ),
            patch("hierocode.repo.files.read_file_safe", return_value="old content"),
            patch(
                "hierocode.repo.diffing.generate_unified_diff",
                return_value=_SAMPLE_DIFF,
            ),
            patch("hierocode.broker.patcher.parse_diff", return_value=[patch_obj]),
            patch(
                "hierocode.broker.patcher.apply_patch", return_value=apply_ok
            ) as mock_apply,
        ):
            runner.invoke(
                app, ["draft", "--task", "fix it", "--file", "foo.py", "--apply"]
            )

        assert mock_apply.called

    def test_draft_without_apply_flag_prints_diff_only(self):
        """draft without -a: apply_patch NOT called; diff is printed."""
        with (
            patch("hierocode.cli.load_config", return_value=_default_config()),
            patch("hierocode.cli.get_provider", return_value=MagicMock()),
            patch(
                "hierocode.broker.capacity.build_capacity_profile",
                return_value=MagicMock(
                    tier="small",
                    context_window=4096,
                    max_input_tokens=2048,
                    max_files_per_unit=3,
                ),
            ),
            patch(
                "hierocode.broker.budget.pack_context",
                return_value=MagicMock(content="packed"),
            ),
            patch(
                "hierocode.broker.prompts.build_drafter_prompt",
                return_value="prompt text",
            ),
            patch(
                "hierocode.cli.get_provider",
                return_value=MagicMock(
                    generate=MagicMock(return_value="new content")
                ),
            ),
            patch("hierocode.repo.files.read_file_safe", return_value="old content"),
            patch(
                "hierocode.repo.diffing.generate_unified_diff",
                return_value=_SAMPLE_DIFF,
            ),
            patch("hierocode.broker.patcher.apply_patch") as mock_apply,
        ):
            runner.invoke(
                app, ["draft", "--task", "fix it", "--file", "foo.py"]
            )

        mock_apply.assert_not_called()
