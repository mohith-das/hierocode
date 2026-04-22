"""Tests for hierocode.broker.prompts — pure string-builder functions, no mocks needed."""


from hierocode.broker.plan_schema import CapacityProfile, TaskUnit
from hierocode.broker.prompts import (
    build_drafter_prompt,
    build_drafter_revision_prompt,
    build_planner_system_prompt,
    build_planner_user_prompt,
    build_qa_prompt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_profile(**kwargs) -> CapacityProfile:
    base = {
        "drafter_model": "llama3.2:3b",
        "param_count_b": 3.0,
        "context_window": 4096,
        "host_ram_gb": 16.0,
        "host_cpu_cores": 8,
        "tier": "narrow",
        "max_input_tokens": 2048,
        "max_output_tokens": 512,
        "max_files_per_unit": 5,
    }
    base.update(kwargs)
    return CapacityProfile(**base)


def _make_unit(**kwargs) -> TaskUnit:
    base = {
        "id": "u1",
        "goal": "Add a health-check endpoint to the FastAPI app",
        "target_files": ["src/app.py"],
        "context_files": ["src/models.py"],
        "acceptance": "GET /health returns 200 with {\"status\": \"ok\"}",
        "est_input_tokens": 800,
    }
    base.update(kwargs)
    return TaskUnit(**base)


# ---------------------------------------------------------------------------
# build_planner_system_prompt
# ---------------------------------------------------------------------------


class TestPlannerSystemPrompt:
    def test_planner_system_prompt_has_schema_marker(self):
        result = build_planner_system_prompt()
        lower = result.lower()
        assert "json" in lower
        assert "drafter" in lower

    def test_planner_system_prompt_is_non_empty_string(self):
        result = build_planner_system_prompt()
        assert isinstance(result, str)
        assert len(result) > 50


# ---------------------------------------------------------------------------
# build_planner_user_prompt
# ---------------------------------------------------------------------------


class TestPlannerUserPrompt:
    def setup_method(self):
        self.task = "Refactor the authentication module to use JWT tokens"
        self.skeleton = "src/\n  auth.py (login, logout, verify)\n  models.py (User)"
        self.profile = _make_profile()

    def test_planner_user_prompt_includes_task(self):
        result = build_planner_user_prompt(self.task, self.skeleton, self.profile)
        assert self.task in result

    def test_planner_user_prompt_includes_skeleton(self):
        result = build_planner_user_prompt(self.task, self.skeleton, self.profile)
        assert self.skeleton in result

    def test_planner_user_prompt_includes_profile_limits(self):
        result = build_planner_user_prompt(self.task, self.skeleton, self.profile)
        assert self.profile.tier in result
        assert str(self.profile.context_window) in result
        assert str(self.profile.max_input_tokens) in result
        assert str(self.profile.max_files_per_unit) in result

    def test_planner_user_prompt_has_json_example(self):
        result = build_planner_user_prompt(self.task, self.skeleton, self.profile)
        assert '"task"' in result
        assert '"units"' in result
        assert '"target_files"' in result

    def test_planner_user_prompt_ends_with_no_commentary_instruction(self):
        result = build_planner_user_prompt(self.task, self.skeleton, self.profile)
        assert "No markdown" in result

    def test_planner_user_prompt_with_none_param_count(self):
        profile = _make_profile(param_count_b=None)
        result = build_planner_user_prompt(self.task, self.skeleton, profile)
        assert "unknown" in result


# ---------------------------------------------------------------------------
# build_drafter_prompt
# ---------------------------------------------------------------------------


class TestDrafterPrompt:
    def test_drafter_prompt_includes_goal_and_context(self):
        unit = _make_unit()
        packed_context = "### BEGIN src/models.py ###\nclass User: ...\n### END src/models.py ###"
        result = build_drafter_prompt(unit, packed_context)
        assert unit.goal in result
        assert packed_context in result

    def test_drafter_prompt_names_primary_target(self):
        unit = _make_unit(target_files=["foo.py"])
        result = build_drafter_prompt(unit, "ctx")
        assert "foo.py" in result
        assert "Primary target" in result

    def test_drafter_prompt_includes_acceptance(self):
        unit = _make_unit(acceptance="Returns HTTP 200")
        result = build_drafter_prompt(unit, "ctx")
        assert "Returns HTTP 200" in result

    def test_drafter_prompt_handles_no_target_files(self):
        # model_construct bypasses the validator so we can test the empty-list guard
        unit = TaskUnit.model_construct(
            id="u1",
            goal="Some goal",
            target_files=[],
            context_files=["ref.py"],
            acceptance="",
            est_input_tokens=0,
        )
        # Must not raise
        result = build_drafter_prompt(unit, "ctx")
        assert isinstance(result, str)
        assert "no target file specified" in result

    def test_drafter_prompt_omits_acceptance_section_when_empty(self):
        unit = _make_unit(acceptance="")
        result = build_drafter_prompt(unit, "ctx")
        assert "Acceptance Criteria" not in result

    def test_drafter_prompt_has_return_only_instruction(self):
        unit = _make_unit()
        result = build_drafter_prompt(unit, "ctx")
        assert "Return ONLY" in result


# ---------------------------------------------------------------------------
# build_drafter_revision_prompt
# ---------------------------------------------------------------------------


class TestDrafterRevisionPrompt:
    def test_drafter_revision_prompt_has_prior_diff_and_feedback(self):
        unit = _make_unit()
        packed_context = "### BEGIN src/app.py ###\npass\n### END ###"
        prior_diff = "--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-pass\n+def health(): ..."
        feedback = "Missing status code 200; health() must return a dict"
        result = build_drafter_revision_prompt(unit, packed_context, prior_diff, feedback)
        assert prior_diff in result
        assert feedback in result

    def test_drafter_revision_prompt_includes_goal(self):
        unit = _make_unit()
        result = build_drafter_revision_prompt(unit, "ctx", "diff", "feedback")
        assert unit.goal in result

    def test_drafter_revision_prompt_includes_packed_context(self):
        unit = _make_unit()
        packed = "### BEGIN src/app.py ###\nold code\n### END ###"
        result = build_drafter_revision_prompt(unit, packed, "diff", "fb")
        assert packed in result

    def test_drafter_revision_prompt_has_address_all_instruction(self):
        unit = _make_unit()
        result = build_drafter_revision_prompt(unit, "ctx", "diff", "fix this")
        assert "Address ALL" in result

    def test_drafter_revision_prompt_names_primary_target(self):
        unit = _make_unit(target_files=["src/app.py"])
        result = build_drafter_revision_prompt(unit, "ctx", "diff", "fb")
        assert "src/app.py" in result
        assert "Primary target" in result


# ---------------------------------------------------------------------------
# build_qa_prompt
# ---------------------------------------------------------------------------


class TestQAPrompt:
    def test_qa_prompt_includes_diff_and_original_task(self):
        unit = _make_unit()
        diff = "--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n+def health(): ..."
        original_task = "Add a health endpoint to the FastAPI service"
        result = build_qa_prompt(unit, diff, original_task=original_task)
        assert diff in result
        assert original_task in result

    def test_qa_prompt_includes_test_output_when_provided(self):
        unit = _make_unit()
        result = build_qa_prompt(unit, "diff", test_output="pytest failed")
        assert "pytest failed" in result

    def test_qa_prompt_omits_test_output_section_when_none(self):
        unit = _make_unit()
        result = build_qa_prompt(unit, "diff", test_output=None)
        # The section header must not appear when test_output is None
        assert "## Test Output" not in result

    def test_qa_prompt_has_action_values(self):
        unit = _make_unit()
        result = build_qa_prompt(unit, "diff")
        assert "accept" in result
        assert "revise" in result
        assert "split" in result
        assert "escalate" in result

    def test_qa_prompt_includes_unit_goal(self):
        unit = _make_unit()
        result = build_qa_prompt(unit, "diff")
        assert unit.goal in result

    def test_qa_prompt_includes_acceptance_when_present(self):
        unit = _make_unit(acceptance="Must pass all tests")
        result = build_qa_prompt(unit, "diff")
        assert "Must pass all tests" in result

    def test_qa_prompt_has_strict_reviewer_preamble(self):
        unit = _make_unit()
        result = build_qa_prompt(unit, "diff")
        lower = result.lower()
        assert "strict" in lower
        assert "reviewer" in lower

    def test_qa_prompt_has_respond_json_only_instruction(self):
        unit = _make_unit()
        result = build_qa_prompt(unit, "diff")
        assert "JSON object only" in result

    def test_qa_prompt_original_task_omitted_when_empty(self):
        unit = _make_unit()
        result = build_qa_prompt(unit, "diff", original_task="")
        assert "## Original Task" not in result
