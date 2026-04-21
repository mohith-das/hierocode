import yaml
from hierocode.models.schemas import HierocodeConfig
from hierocode.utils.paths import get_config_path
from hierocode.exceptions import ConfigError

DEFAULT_CONFIG_YAML = """\
default_provider: local_default
default_model: model-large
small_model: model-small

providers:
  local_default:
    type: ollama
    base_url: http://localhost:11434
    auth:
      type: none

  local_network_backend:
    type: openai_compatible
    base_url: http://<local-ip-or-hostname>:<port>/v1
    auth:
      type: bearer_env
      env_var: LOCAL_BACKEND_API_KEY

  remote_backend:
    type: openai_compatible
    base_url: https://<your-domain-or-endpoint>/v1
    auth:
      type: bearer_env
      env_var: HIEROCODE_API_KEY

parallelization:
  default_strategy: balanced
  max_local_workers: 4
  max_remote_workers: 8

routing:
  draft_model: model-small
  review_model: model-large
"""

def load_config() -> HierocodeConfig:
    """Loads configuration from ~/.hierocode.yaml. Raises ConfigError if not found."""
    path = get_config_path()
    if not path.exists():
        raise ConfigError(f"Configuration file not found at {path}. Run 'hierocode init' to create one.")
    
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        if not data:
            data = {}
        
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
