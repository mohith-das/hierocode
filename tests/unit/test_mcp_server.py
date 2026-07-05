import json
from unittest.mock import patch, MagicMock

from hierocode.mcp_server import draft_code, drafter_info, usage_summary, mcp

def test_tools_registered():
    tools = [t.name for t in mcp._tool_manager.list_tools()]
    assert "draft_code" in tools
    assert "drafter_info" in tools
    assert "usage_summary" in tools

@patch("hierocode.mcp_server.draft_unit")
@patch("hierocode.mcp_server._get_config")
def test_draft_code_happy_path(mock_get_config, mock_draft_unit):
    mock_get_config.return_value = MagicMock()
    mock_draft_unit.return_value = MagicMock(status="ok", diff="abc")
    # Need asdict to work on MagicMock, so let's mock it properly or use DraftResult
    from hierocode.engine import DraftResult
    mock_draft_unit.return_value = DraftResult(status="ok", diff="abc")
    
    res = draft_code("goal", "t.py")
    parsed = json.loads(res)
    assert parsed["status"] == "ok"
    assert parsed["diff"] == "abc"

@patch("hierocode.mcp_server._get_config")
def test_draft_code_invalid_repo_root(mock_get_config):
    mock_get_config.return_value = MagicMock()
    res = draft_code("goal", "t.py", repo_root="/invalid/path/that/does/not/exist")
    parsed = json.loads(res)
    assert parsed["status"] == "error"
    assert parsed["error_type"] == "config"

@patch("hierocode.mcp_server._get_config")
def test_draft_code_config_missing_does_not_raise(mock_get_config):
    mock_get_config.side_effect = Exception("Config error")
    res = draft_code("goal", "t.py")
    parsed = json.loads(res)
    assert parsed["status"] == "error"
    assert "Config error" in parsed["message"]

@patch("hierocode.mcp_server._get_config")
@patch("hierocode.mcp_server.get_route")
@patch("hierocode.mcp_server.get_provider")
@patch("hierocode.mcp_server.build_capacity_profile")
def test_drafter_info_happy_path(mock_build, mock_get_provider, mock_route, mock_get_config):
    mock_get_config.return_value = MagicMock()
    mock_route.return_value = ("prov", "mod")
    provider = MagicMock()
    provider.healthcheck.return_value = True
    mock_get_provider.return_value = provider
    profile = MagicMock()
    profile.drafter_model = "mod"
    profile.param_count_b = 3.0
    profile.quantization = "Q4"
    profile.context_window = 4000
    profile.max_input_tokens = 2000
    profile.max_output_tokens = 1000
    mock_build.return_value = profile
    
    res = drafter_info()
    parsed = json.loads(res)
    assert parsed["status"] == "ok"
    assert parsed["drafter_model"] == "mod"
    assert parsed["reachable"] is True

def test_usage_summary():
    res = usage_summary()
    parsed = json.loads(res)
    assert parsed["status"] == "ok"
    assert isinstance(parsed["usage"], dict)
