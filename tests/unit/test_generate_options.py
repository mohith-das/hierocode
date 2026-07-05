import pytest
from hierocode.providers.options import parse_options
from pydantic import ValidationError

def test_generate_options_defaults():
    opts = parse_options({})
    assert opts.max_tokens is None
    assert opts.system is None
    assert opts.json_mode is False
    assert opts.temperature is None
    assert opts.timeout is None
    assert opts.cwd is None
    assert opts.exploration == "passive"
    assert opts.allowed_tools is None

def test_generate_options_forbids_unknown():
    with pytest.raises(ValidationError):
        parse_options({"unknown_key": "value"})

def test_generate_options_parses_known():
    opts = parse_options({
        "max_tokens": 100,
        "system": "sys",
        "json_mode": True,
        "temperature": 0.5,
        "timeout": 10.0,
        "cwd": "/tmp",
        "exploration": "active",
        "allowed_tools": ["Bash"],
    })
    assert opts.max_tokens == 100
    assert opts.system == "sys"
    assert opts.json_mode is True
    assert opts.temperature == 0.5
    assert opts.timeout == 10.0
    assert opts.cwd == "/tmp"
    assert opts.exploration == "active"
    assert opts.allowed_tools == ["Bash"]
