from typer.testing import CliRunner
import json
from unittest.mock import patch

from hierocode.cli import app
from hierocode.engine import DraftResult

runner = CliRunner()

@patch("hierocode.engine.draft_unit")
def test_cli_draft_json(mock_draft_unit):
    mock_draft_unit.return_value = DraftResult(status="ok", diff="--- a\n+++ b\n@@ -1 +1 @@\n-a\n+b")
    
    result = runner.invoke(app, ["draft", "--goal", "g", "--target", "t.py", "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed["status"] == "ok"
    assert "diff" in parsed
    
@patch("hierocode.engine.draft_unit")
def test_cli_draft_error_json(mock_draft_unit):
    mock_draft_unit.return_value = DraftResult(status="error", error_type="config", error_message="foo")
    
    result = runner.invoke(app, ["draft", "--goal", "g", "--target", "t.py", "--json"])
    assert result.exit_code == 1
    parsed = json.loads(result.stdout)
    assert parsed["status"] == "error"
    assert parsed["error_type"] == "config"
