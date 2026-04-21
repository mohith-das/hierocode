from typing import Optional, Literal
from pydantic import BaseModel, Field

class AuthConfig(BaseModel):
    type: Literal["none", "bearer_env"] = "none"
    env_var: Optional[str] = None

class ProviderConfig(BaseModel):
    type: Literal["ollama", "openai_compatible", "lmstudio", "transformers_local"]
    base_url: Optional[str] = None
    auth: AuthConfig = Field(default_factory=AuthConfig)

class ParallelizationConfig(BaseModel):
    default_strategy: Literal["safe", "balanced", "aggressive"] = "balanced"
    max_local_workers: int = 4
    max_remote_workers: int = 8

class RoutingConfig(BaseModel):
    draft_model: str = "model-small"
    review_model: str = "model-large"

class HierocodeConfig(BaseModel):
    default_provider: str = "local_default"
    default_model: str = "model-large"
    small_model: str = "model-small"

    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    parallelization: ParallelizationConfig = Field(default_factory=ParallelizationConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
