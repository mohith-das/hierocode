import os
from hierocode.models.schemas import AuthConfig

def resolve_auth_token(config: AuthConfig) -> str | None:
    """Resolve an authentication token based on the AuthConfig."""
    if config.type == "none":
        return None
    elif config.type == "bearer_env":
        if not config.env_var:
            return None
        return os.environ.get(config.env_var)
    return None
