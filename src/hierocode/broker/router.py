"""Role-to-provider routing for the planner / drafter / reviewer pipeline."""

from typing import Tuple

from hierocode.exceptions import ConfigError
from hierocode.models.schemas import HierocodeConfig


def get_route(config: HierocodeConfig, role: str) -> Tuple[str, str]:
    """Resolve (provider_name, model_name) for a role: 'planner', 'drafter', or 'reviewer'.

    Reviewer falls back to planner when not explicitly set. All other roles require
    an explicit `routing.<role>` block; users upgrading from v0.1 should re-run
    `hierocode init --wizard --force` to regenerate their config.
    """
    routing = config.routing
    if role == "planner":
        role_cfg = routing.planner
    elif role == "drafter":
        role_cfg = routing.drafter
    elif role == "reviewer":
        role_cfg = routing.reviewer or routing.planner
    else:
        raise ConfigError(f"Unknown routing role: {role}")

    if role_cfg is None:
        raise ConfigError(
            f"No provider/model configured for role '{role}'. "
            f"Run `hierocode init --wizard --force` to regenerate your config."
        )
    if role_cfg.provider not in config.providers:
        raise ConfigError(
            f"Role '{role}' references provider '{role_cfg.provider}' which is not configured."
        )
    return role_cfg.provider, role_cfg.model
