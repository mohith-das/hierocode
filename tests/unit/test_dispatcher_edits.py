from unittest.mock import MagicMock

from hierocode.broker.dispatcher import run_plan
from hierocode.broker.plan_schema import Plan, TaskUnit, CapacityProfile

def test_dispatcher_edit_blocks(tmp_path):
    # Setup
    (tmp_path / "a.py").write_text("def foo():\n    pass\n")
    
    unit = TaskUnit(id="1", goal="g", target_files=["a.py"])
    plan = Plan(task="t", units=[unit])
    
    planner = MagicMock()
    planner.generate.return_value = '{"action": "accept"}'
    
    drafter = MagicMock()
    
    # Drafter emits edit block
    drafter.generate.return_value = """
<<<<<<< SEARCH
def foo():
    pass
=======
def foo():
    return 1
>>>>>>> REPLACE
"""
    profile = CapacityProfile(
        drafter_model="test",
        context_window=1000,
        host_ram_gb=1,
        host_cpu_cores=1,
        tier="micro",
        max_input_tokens=100,
        max_output_tokens=100,
        max_files_per_unit=1,
    )
    
    result = run_plan(plan, profile, planner, "p", drafter, "d", tmp_path)
    
    assert len(result.units) == 1
    assert result.units[0].status == "completed"
    assert "+    return 1" in result.units[0].diff

def test_dispatcher_edit_blocks_broken_search_triggers_revision(tmp_path):
    # Setup
    (tmp_path / "a.py").write_text("def foo():\n    pass\n")
    
    unit = TaskUnit(id="1", goal="g", target_files=["a.py"])
    plan = Plan(task="t", units=[unit])
    
    planner = MagicMock()
    # If QA runs, accept. But pre_review_verdict will trigger a revision first.
    planner.generate.return_value = '{"action": "accept"}'
    
    drafter = MagicMock()
    
    # First emit a broken search, then a correct one
    drafter.generate.side_effect = [
        """
<<<<<<< SEARCH
def foo():
    pass_broken
=======
def foo():
    return 1
>>>>>>> REPLACE
""",
        """
<<<<<<< SEARCH
def foo():
    pass
=======
def foo():
    return 1
>>>>>>> REPLACE
"""
    ]
    
    profile = CapacityProfile(
        drafter_model="test",
        context_window=1000,
        host_ram_gb=1,
        host_cpu_cores=1,
        tier="micro",
        max_input_tokens=100,
        max_output_tokens=100,
        max_files_per_unit=1,
    )
    
    result = run_plan(plan, profile, planner, "p", drafter, "d", tmp_path)
    print("DRAFTER CALLS", drafter.generate.call_count)
    print("UNITS", result.units)
    
    assert len(result.units) == 1
    assert result.units[0].status == "completed"
    assert result.units[0].revision_count == 1
    assert "+    return 1" in result.units[0].diff
