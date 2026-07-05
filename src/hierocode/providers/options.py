"""Typed generation options shared by all providers."""

from typing import Literal, Optional

from pydantic import BaseModel


class GenerateOptions(BaseModel):
    """Normalized options for BaseProvider.generate; providers translate explicitly."""

    max_tokens: Optional[int] = None
    system: Optional[str] = None
    json_mode: bool = False
    temperature: Optional[float] = None
    timeout: Optional[float] = None
    cwd: Optional[str] = None
    exploration: Literal["passive", "active"] = "passive"
    allowed_tools: Optional[list[str]] = None

    model_config = {"extra": "forbid"}

def parse_options(options: dict) -> GenerateOptions:
    """Validate raw generate() kwargs; unknown keys raise immediately (fail loud)."""
    return GenerateOptions(**options)
