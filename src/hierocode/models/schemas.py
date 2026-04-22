from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class AuthConfig(BaseModel):
    type: Literal["none", "bearer_env"] = "none"
    env_var: Optional[str] = None


class ProviderConfig(BaseModel):
    type: Literal[
        "ollama",
        "openai_compatible",
        "lmstudio",
        "transformers_local",
        "anthropic",
        "claude_code_cli",
        "codex_cli",
    ]
    base_url: Optional[str] = None
    auth: AuthConfig = Field(default_factory=AuthConfig)


class ParallelizationConfig(BaseModel):
    default_strategy: Literal["safe", "balanced", "aggressive"] = "balanced"
    max_local_workers: int = 4
    max_remote_workers: int = 8


class RoleRouting(BaseModel):
    provider: str
    model: str


class RoutingConfig(BaseModel):
    """Per-role planner/drafter/reviewer bindings. Extra keys (e.g. legacy `draft_model`)
    from older configs are silently ignored — upgrade with `hierocode init --wizard --force`."""

    model_config = ConfigDict(extra="ignore")

    planner: Optional[RoleRouting] = None
    drafter: Optional[RoleRouting] = None
    reviewer: Optional[RoleRouting] = None


class PolicyConfig(BaseModel):
    max_revisions_per_unit: int = 2
    max_escalations_per_task: int = 3
    warn_before_escalation: bool = True


class HierocodeConfig(BaseModel):
    """Top-level hierocode config. Unknown keys from v0.1 configs are tolerated."""

    model_config = ConfigDict(extra="ignore")

    default_provider: str = "local_default"

    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    parallelization: ParallelizationConfig = Field(default_factory=ParallelizationConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
