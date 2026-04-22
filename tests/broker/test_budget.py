"""Tests for hierocode.broker.budget."""


from hierocode.broker.budget import estimate_tokens, pack_context
from hierocode.broker.plan_schema import CapacityProfile, TaskUnit


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _profile(**kwargs) -> CapacityProfile:
    defaults = dict(
        drafter_model="test-model",
        context_window=4096,
        host_ram_gb=16.0,
        host_cpu_cores=8,
        tier="standard",
        max_input_tokens=4096,
        max_output_tokens=1024,
        max_files_per_unit=10,
    )
    defaults.update(kwargs)
    return CapacityProfile(**defaults)


def _unit(target_files=None, context_files=None, **kwargs) -> TaskUnit:
    defaults = dict(
        id="u1",
        goal="Test goal",
        target_files=target_files or ["a.py"],
        context_files=context_files or [],
    )
    defaults.update(kwargs)
    return TaskUnit(**defaults)


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

def test_estimate_tokens_char_div_4():
    assert estimate_tokens("a" * 100) == 25
    assert estimate_tokens("") == 0


# ---------------------------------------------------------------------------
# pack_context tests
# ---------------------------------------------------------------------------

def test_includes_target_file_within_budget(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    unit = _unit(target_files=["a.py"])
    result = pack_context(unit, _profile(), tmp_path)
    assert "a.py" in result.included_files
    assert result.skipped_files == []
    assert result.truncated_files == []


def test_respects_max_files_per_unit(tmp_path):
    for name in ("a.py", "b.py", "c.py"):
        (tmp_path / name).write_text("x = 1\n")
    unit = _unit(target_files=["a.py", "b.py", "c.py"])
    result = pack_context(unit, _profile(max_files_per_unit=1), tmp_path)
    assert result.included_files == ["a.py"]
    assert "b.py" in result.skipped_files
    assert "c.py" in result.skipped_files


def test_target_file_truncated_when_oversized(tmp_path):
    big_content = "x = 1\n" * 5000  # ~30 000 chars → ~7 500 tokens > 300 - 800 budget
    (tmp_path / "big.py").write_text(big_content)
    unit = _unit(target_files=["big.py"])
    profile = _profile(max_input_tokens=300)
    result = pack_context(unit, profile, tmp_path)
    assert "big.py" in result.truncated_files
    assert "... [truncated," in result.content


def test_context_file_skipped_when_oversized(tmp_path):
    (tmp_path / "small.py").write_text("x = 1\n")
    big_content = "y = 2\n" * 5000
    (tmp_path / "big.py").write_text(big_content)
    unit = _unit(target_files=["small.py"], context_files=["big.py"])
    result = pack_context(unit, _profile(), tmp_path)
    assert "small.py" in result.included_files
    assert "big.py" in result.skipped_files


def test_missing_file_uses_placeholder(tmp_path):
    unit = _unit(target_files=["missing.py"])
    result = pack_context(unit, _profile(), tmp_path)
    assert "missing.py" in result.included_files
    assert "[file not found]" in result.content


def test_priority_target_before_context(tmp_path):
    (tmp_path / "a.py").write_text("a_content\n")
    (tmp_path / "b.py").write_text("b_content\n")
    unit = _unit(target_files=["a.py"], context_files=["b.py"])
    result = pack_context(unit, _profile(), tmp_path)
    assert "a.py" in result.included_files
    assert "b.py" in result.included_files
    assert result.content.index("a.py") < result.content.index("b.py")


def test_formatted_headers_present(tmp_path):
    (tmp_path / "a.py").write_text("content\n")
    unit = _unit(target_files=["a.py"])
    result = pack_context(unit, _profile(), tmp_path)
    assert "--- a.py ---" in result.content
    assert "--- end a.py ---" in result.content


def test_empty_target_and_context_returns_empty_content(tmp_path):
    unit = TaskUnit.model_construct(
        id="u1",
        goal="Test",
        target_files=[],
        context_files=[],
        acceptance="",
        est_input_tokens=0,
    )
    result = pack_context(unit, _profile(), tmp_path)
    assert result.content == ""
    assert result.estimated_tokens == 0
    assert result.included_files == []
    assert result.skipped_files == []
    assert result.truncated_files == []
