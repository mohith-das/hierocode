from hierocode.models.schemas import HierocodeConfig
import pytest
from pydantic import ValidationError

def test_config_parsing_defaults():
    config = HierocodeConfig()
    assert config.default_provider == "local_default"
    assert config.default_model == "model-large"
    assert config.parallelization.default_strategy == "balanced"
    assert config.routing.draft_model == "model-small"

def test_config_parsing_custom():
    data = {"default_provider": "remote_one"}
    config = HierocodeConfig(**data)
    assert config.default_provider == "remote_one"
