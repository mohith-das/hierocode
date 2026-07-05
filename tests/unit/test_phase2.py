import pytest
from unittest.mock import MagicMock

from hierocode.broker.planner import generate_plan
from hierocode.broker.qa import review_draft
from hierocode.broker.plan_schema import PlanParseError, CapacityProfile, TaskUnit, Plan
from hierocode.broker.dispatcher import run_plan

# Fixtures
def _profile() -> CapacityProfile:
    return CapacityProfile(
        drafter_model="test",
        context_window=1000,
        host_ram_gb=1,
        host_cpu_cores=1,
        tier="micro",
        max_input_tokens=100,
        max_output_tokens=100,
        max_files_per_unit=1,
    )

def test_planner_retry_on_parse_error():
    provider = MagicMock()
    # First call: bad json
    # Second call: good json
    provider.generate.side_effect = [
        "not json",
        '{"task": "task", "units": [{"id": "1", "goal": "g", "target_files": ["a.py"], "context_files": [], "acceptance": "", "est_input_tokens": 10}]}'
    ]
    
    plan = generate_plan("t", "s", _profile(), provider, "model")
    
    assert provider.generate.call_count == 2
    assert len(plan.units) == 1
    
    args1, kwargs1 = provider.generate.call_args_list[0]
    args2, kwargs2 = provider.generate.call_args_list[1]
    
    # Assert retry prompt
    assert "Previous attempt failed" in kwargs2["prompt"]
    assert "not json" not in kwargs1["prompt"]

def test_planner_retry_fails_eventually():
    provider = MagicMock()
    provider.generate.side_effect = ["not json", "still not json"]
    
    with pytest.raises(PlanParseError):
        generate_plan("t", "s", _profile(), provider, "model")
    assert provider.generate.call_count == 2

def test_qa_retry_on_parse_error():
    provider = MagicMock()
    provider.generate.side_effect = [
        "not json",
        '{"action": "accept"}'
    ]
    
    unit = TaskUnit(id="1", goal="g", target_files=["a.py"])
    verdict = review_draft(provider, "model", unit, "diff")
    
    assert provider.generate.call_count == 2
    assert verdict.action == "accept"
    
    args1, kwargs1 = provider.generate.call_args_list[0]
    args2, kwargs2 = provider.generate.call_args_list[1]
    assert "Previous attempt failed" in kwargs2["prompt"]

# 2f: Same-file units: diff against evolving state
def test_same_file_units_evolving_state(tmp_path):
    (tmp_path / "a.py").write_text("v1\n")
    
    unit1 = TaskUnit(id="1", goal="g1", target_files=["a.py"])
    unit2 = TaskUnit(id="2", goal="g2", target_files=["a.py"])
    
    plan = Plan(task="t", units=[unit1, unit2])
    
    planner = MagicMock()
    drafter = MagicMock()
    
    # QA always accepts
    planner.generate.return_value = '{"action": "accept"}'
    
    # Drafter emits new state
    def mock_draft(*args, **kwargs):
        if "g1" in kwargs["prompt"]:
            return "v1\nv2\n"
        else:
            # At this point, the file should be v1\nv2\n
            assert "v1\nv2\n" in kwargs["prompt"]
            return "v1\nv2\nv3\n"
            
    drafter.generate.side_effect = mock_draft
    
    result = run_plan(plan, _profile(), planner, "p_m", drafter, "d_m", tmp_path)
    
    assert len(result.units) == 2
    assert result.units[0].status == "completed"
    assert result.units[1].status == "completed"
    
    # The first diff should go from v1 to v1\nv2
    assert "-v1" in result.units[0].diff or "+v2" in result.units[0].diff
    # The second diff should go from v1\nv2 to v1\nv2\nv3
    assert "+v3" in result.units[1].diff

# 2g: Bound the split loop
def test_bound_split_loop():
    unit1 = TaskUnit(id="1", goal="g1", target_files=["a.py"])
    plan = Plan(task="t", units=[unit1])
    
    planner = MagicMock()
    drafter = MagicMock()
    
    # Drafter just returns "output"
    drafter.generate.return_value = "output"
    
    # Reviewer always splits
    def mock_review(*args, **kwargs):
        # We need to return a split verdict with sub_units
        return '{"action": "split", "sub_units": [{"id": "s1", "goal": "s", "target_files": ["a.py"]}]}'
        
    planner.generate.side_effect = mock_review
    
    result = run_plan(plan, _profile(), planner, "p_m", drafter, "d_m", "/repo")
    
    # It should terminate, and remaining units should be marked failed.
    failed_units = [u for u in result.units if u.status == "failed"]
    assert len(failed_units) > 0
    assert "unit budget exhausted" in failed_units[-1].reason
