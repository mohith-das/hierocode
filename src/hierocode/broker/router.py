from hierocode.config import HierocodeConfig
from hierocode.exceptions import ConfigError
from typing import Tuple

def get_draft_model_route(config: HierocodeConfig) -> Tuple[str, str]:
    """Returns the (provider_name, model_name) to use for drafting."""
    provider_name = config.default_provider
    if provider_name not in config.providers:
        raise ConfigError(f"Default provider '{provider_name}' not configured.")
    return provider_name, config.routing.draft_model

def get_review_model_route(config: HierocodeConfig) -> Tuple[str, str]:
    """Returns the (provider_name, model_name) to use for review/planning."""
    provider_name = config.default_provider
    if provider_name not in config.providers:
        raise ConfigError(f"Default provider '{provider_name}' not configured.")
    return provider_name, config.routing.review_model
