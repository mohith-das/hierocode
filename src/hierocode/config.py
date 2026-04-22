import yaml

from hierocode.exceptions import ConfigError
from hierocode.models.schemas import HierocodeConfig
from hierocode.utils.paths import get_config_path

DEFAULT_CONFIG_YAML = """\
default_provider: local_ollama

providers:
  local_ollama:
    type: ollama
    base_url: http://localhost:11434
    auth:
      type: none

routing:
  planner:
    provider: local_ollama
    model: llama3.2:3b
  drafter:
    provider: local_ollama
    model: llama3.2:3b

policy:
  max_revisions_per_unit: 2
  max_escalations_per_task: 3
  warn_before_escalation: true

tui:
  interaction_mode: prompt

tasks: []

parallelization:
  default_strategy: balanced
  max_local_workers: 4
  max_remote_workers: 8
"""


def load_config() -> HierocodeConfig:
    """Loads configuration from ~/.hierocode.yaml. Raises ConfigError if not found."""
    path = get_config_path()
    if not path.exists():
        raise ConfigError(f"Configuration file not found at {path}. Run 'hierocode init' to create one.")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    try:
        return HierocodeConfig(**data)
    except Exception as e:
        raise ConfigError(f"Invalid configuration format in {path}: {e}")


def create_default_config(force: bool = False):
    """Creates a default ~/.hierocode.yaml configuration file."""
    path = get_config_path()
    if path.exists() and not force:
        raise ConfigError(f"Configuration file already exists at {path}. Use --force to overwrite.")

    with open(path, "w", encoding="utf-8") as f:
        f.write(DEFAULT_CONFIG_YAML)
