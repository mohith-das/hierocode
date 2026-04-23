"""Tests for hierocode.broker.plan_schema."""

import json
import pytest

from hierocode.broker.plan_schema import (
    CapacityProfile,
    Plan,
    PlanParseError,
    QAVerdict,
    TaskUnit,
    parse_plan_from_llm_output,
    parse_qa_verdict_from_llm_output,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _unit(**kwargs) -> dict:
    base = {"id": "u1", "goal": "Do something", "target_files": ["src/foo.py"]}
    base.update(kwargs)
    return base


def _plan(**kwargs) -> dict:
    base = {"task": "Implement feature X", "units": [_unit()]}
    base.update(kwargs)
    return base


def _capacity(**kwargs) -> dict:
    base = {
        "drafter_model": "llama3.2:3b",
        "context_window": 4096,
        "host_ram_gb": 16.0,
        "host_cpu_cores": 8,
        "tier": "narrow",
        "max_input_tokens": 2048,
        "max_output_tokens": 512,
        "max_files_per_unit": 5,
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# TaskUnit
# ---------------------------------------------------------------------------

class TestTaskUnit:
    def test_task_unit_requires_goal_and_id(self):
        with pytest.raises(Exception):
            TaskUnit(id="", goal="do thing", target_files=["f.py"])
        with pytest.raises(Exception):
            TaskUnit(id="u1", goal="", target_files=["f.py"])

    def test_task_unit_requires_at_least_one_file(self):
        with pytest.raises(ValueError, match="at least one"):
            TaskUnit(id="u1", goal="do thing")

    def test_task_unit_context_files_alone_is_valid(self):
        unit = TaskUnit(id="u1", goal="read stuff", context_files=["ref.py"])
        assert unit.context_files == ["ref.py"]

    def test_task_unit_target_files_alone_is_valid(self):
        unit = TaskUnit(id="u1", goal="edit stuff", target_files=["main.py"])
        assert unit.target_files == ["main.py"]

    def test_est_input_tokens_non_negative(self):
        with pytest.raises(ValueError, match="est_input_tokens"):
            TaskUnit(id="u1", goal="do thing", target_files=["f.py"], est_input_tokens=-1)

    def test_est_input_tokens_zero_is_valid(self):
        unit = TaskUnit(id="u1", goal="do thing", target_files=["f.py"], est_input_tokens=0)
        assert unit.est_input_tokens == 0


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

class TestPlan:
    def test_plan_rejects_empty_units(self):
        with pytest.raises(ValueError, match="at least one"):
            Plan(task="task", units=[])

    def test_plan_rejects_duplicate_unit_ids(self):
        u1 = TaskUnit(id="u1", goal="goal", target_files=["f.py"])
        u2 = TaskUnit(id="u1", goal="goal2", target_files=["g.py"])
        with pytest.raises(ValueError, match="Duplicate"):
            Plan(task="task", units=[u1, u2])

    def test_plan_valid_with_unique_units(self):
        u1 = TaskUnit(id="u1", goal="goal1", target_files=["a.py"])
        u2 = TaskUnit(id="u2", goal="goal2", target_files=["b.py"])
        plan = Plan(task="do it", units=[u1, u2])
        assert len(plan.units) == 2

    def test_plan_requires_task(self):
        u1 = TaskUnit(id="u1", goal="g", target_files=["f.py"])
        with pytest.raises(Exception):
            Plan(task="", units=[u1])


# ---------------------------------------------------------------------------
# QAVerdict
# ---------------------------------------------------------------------------

class TestQAVerdict:
    def test_qa_revise_requires_feedback(self):
        with pytest.raises(ValueError, match="feedback"):
            QAVerdict(action="revise")

    def test_qa_revise_requires_non_empty_feedback(self):
        with pytest.raises(ValueError, match="feedback"):
            QAVerdict(action="revise", feedback="")

    def test_qa_revise_valid(self):
        v = QAVerdict(action="revise", feedback="Fix the imports")
        assert v.feedback == "Fix the imports"

    def test_qa_split_requires_sub_units(self):
        with pytest.raises(ValueError, match="sub_units"):
            QAVerdict(action="split")

    def test_qa_split_requires_non_empty_sub_units(self):
        with pytest.raises(ValueError, match="sub_units"):
            QAVerdict(action="split", sub_units=[])

    def test_qa_split_valid(self):
        u = TaskUnit(id="s1", goal="sub", target_files=["x.py"])
        v = QAVerdict(action="split", sub_units=[u])
        assert len(v.sub_units) == 1  # type: ignore[arg-type]

    def test_qa_accept_forbids_feedback_or_sub_units(self):
        with pytest.raises(ValueError, match="feedback"):
            QAVerdict(action="accept", feedback="nice")
        with pytest.raises(ValueError, match="sub_units"):
            u = TaskUnit(id="s1", goal="sub", target_files=["x.py"])
            QAVerdict(action="accept", sub_units=[u])

    def test_qa_escalate_forbids_feedback_or_sub_units(self):
        with pytest.raises(ValueError, match="feedback"):
            QAVerdict(action="escalate", feedback="too hard")
        with pytest.raises(ValueError, match="sub_units"):
            u = TaskUnit(id="s1", goal="sub", target_files=["x.py"])
            QAVerdict(action="escalate", sub_units=[u])

    def test_qa_accept_valid(self):
        v = QAVerdict(action="accept")
        assert v.action == "accept"
        assert v.feedback is None
        assert v.sub_units is None

    def test_qa_accept_allows_empty_sub_units_list(self):
        """Regression: LLMs often emit `sub_units: []` rather than null for
        accept/escalate. Empty list is semantically equivalent to None here."""
        v = QAVerdict(action="accept", sub_units=[])
        assert v.action == "accept"
        assert v.sub_units == []

    def test_qa_accept_allows_empty_feedback_string(self):
        """Regression: LLMs often emit `feedback: ""` rather than null for
        accept/escalate. Empty string is semantically equivalent to None here."""
        v = QAVerdict(action="accept", feedback="")
        assert v.action == "accept"
        assert v.feedback == ""

    def test_qa_escalate_allows_empty_sub_units_list(self):
        v = QAVerdict(action="escalate", sub_units=[])
        assert v.action == "escalate"

    def test_qa_accept_still_rejects_populated_sub_units(self):
        """Truthiness check only — a non-empty sub_units list still errors."""
        u = TaskUnit(id="s1", goal="sub", target_files=["x.py"])
        with pytest.raises(ValueError, match="sub_units"):
            QAVerdict(action="accept", sub_units=[u])

    def test_qa_escalate_with_reason(self):
        v = QAVerdict(action="escalate", reason="Too complex for drafter")
        assert v.reason == "Too complex for drafter"


# ---------------------------------------------------------------------------
# CapacityProfile
# ---------------------------------------------------------------------------

class TestCapacityProfile:
    def test_capacity_profile_rejects_nonpositive_context(self):
        with pytest.raises(ValueError, match="positive"):
            CapacityProfile(**_capacity(context_window=0))

    def test_capacity_profile_rejects_nonpositive_max_input(self):
        with pytest.raises(ValueError, match="positive"):
            CapacityProfile(**_capacity(max_input_tokens=-1))

    def test_capacity_profile_rejects_nonpositive_max_output(self):
        with pytest.raises(ValueError, match="positive"):
            CapacityProfile(**_capacity(max_output_tokens=0))

    def test_capacity_profile_rejects_nonpositive_max_files(self):
        with pytest.raises(ValueError, match="positive"):
            CapacityProfile(**_capacity(max_files_per_unit=0))

    def test_capacity_profile_rejects_nonpositive_cpu_cores(self):
        with pytest.raises(ValueError, match="positive"):
            CapacityProfile(**_capacity(host_cpu_cores=0))

    def test_capacity_profile_valid(self):
        cap = CapacityProfile(**_capacity())
        assert cap.tier == "narrow"
        assert cap.has_gpu is False


# ---------------------------------------------------------------------------
# parse_plan_from_llm_output
# ---------------------------------------------------------------------------

class TestParsePlanFromLLMOutput:
    _PLAN_JSON = json.dumps(_plan())

    def test_parse_plan_happy_path(self):
        plan = parse_plan_from_llm_output(self._PLAN_JSON)
        assert plan.task == "Implement feature X"
        assert plan.units[0].id == "u1"

    def test_parse_plan_with_markdown_fences(self):
        raw = f"```json\n{self._PLAN_JSON}\n```"
        plan = parse_plan_from_llm_output(raw)
        assert plan.task == "Implement feature X"

    def test_parse_plan_with_plain_fences(self):
        raw = f"```\n{self._PLAN_JSON}\n```"
        plan = parse_plan_from_llm_output(raw)
        assert plan.task == "Implement feature X"

    def test_parse_plan_with_surrounding_prose(self):
        raw = f"Sure! Here's the plan: {self._PLAN_JSON} Hope that helps!"
        plan = parse_plan_from_llm_output(raw)
        assert plan.task == "Implement feature X"

    def test_parse_plan_invalid_json_raises_plan_parse_error(self):
        with pytest.raises(PlanParseError):
            parse_plan_from_llm_output("not json at all {{broken")

    def test_parse_plan_valid_json_bad_schema_raises_plan_parse_error(self):
        bad = json.dumps({"task": "oops", "units": "not-a-list"})
        with pytest.raises(PlanParseError):
            parse_plan_from_llm_output(bad)

    def test_parse_plan_empty_units_raises_plan_parse_error(self):
        bad = json.dumps({"task": "oops", "units": []})
        with pytest.raises(PlanParseError):
            parse_plan_from_llm_output(bad)


# ---------------------------------------------------------------------------
# parse_qa_verdict_from_llm_output
# ---------------------------------------------------------------------------

class TestParseQAVerdictFromLLMOutput:
    _ACCEPT_JSON = json.dumps({"action": "accept"})
    _REVISE_JSON = json.dumps({"action": "revise", "feedback": "Fix types"})

    def test_parse_qa_verdict_happy_path(self):
        verdict = parse_qa_verdict_from_llm_output(self._ACCEPT_JSON)
        assert verdict.action == "accept"

    def test_parse_qa_verdict_with_markdown_fences(self):
        raw = f"```json\n{self._REVISE_JSON}\n```"
        verdict = parse_qa_verdict_from_llm_output(raw)
        assert verdict.action == "revise"
        assert verdict.feedback == "Fix types"

    def test_parse_qa_verdict_invalid_action_raises_plan_parse_error(self):
        bad = json.dumps({"action": "unknown_action"})
        with pytest.raises(PlanParseError):
            parse_qa_verdict_from_llm_output(bad)

    def test_parse_qa_verdict_revise_missing_feedback_raises_plan_parse_error(self):
        bad = json.dumps({"action": "revise"})
        with pytest.raises(PlanParseError):
            parse_qa_verdict_from_llm_output(bad)
