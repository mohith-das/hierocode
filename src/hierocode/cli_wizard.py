"""Auto-detect available planners/drafters and write a tailored ~/.hierocode.yaml."""

from __future__ import annotations

import os
import shutil
import subprocess
import warnings
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import httpx

from hierocode.exceptions import ConfigError

try:
    from hierocode.utils.paths import get_config_path
except ImportError:
    def get_config_path() -> Path:  # type: ignore[misc]
        """Fallback if utils.paths is unavailable."""
        return Path.home() / ".hierocode.yaml"

try:
    from hierocode.runtime.resources import get_total_ram_gb
except ImportError:
    def get_total_ram_gb() -> float:  # type: ignore[misc]
        """Fallback RAM probe."""
        return 16.0

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Aggregated snapshot of the host environment."""

    claude_cli_available: bool
    codex_cli_available: bool
    anthropic_api_key_present: bool
    ollama_reachable: bool
    ollama_models: list[str] = field(default_factory=list)
    total_ram_gb: float = 0.0
    recommended_drafter_model: Optional[str] = None
    recommended_planner_type: Optional[str] = None


# ---------------------------------------------------------------------------
# Planner / drafter model tables
# ---------------------------------------------------------------------------

_PLANNER_MODEL: dict[str, str] = {
    "claude_code_cli": "claude-sonnet-4-6",
    "codex_cli": "gpt-5-codex",
    "anthropic": "claude-haiku-4-5",
    # "ollama" is filled dynamically (same as drafter)
}

# (max_ram_exclusive, preferred_model) — ordered from smallest to largest
_RAM_DRAFTER_TABLE: list[tuple[float, str]] = [
    (12.0, "llama3.2:1b"),
    (24.0, "llama3.2:3b"),
    (48.0, "qwen2.5-coder:7b"),
    (float("inf"), "qwen2.5-coder:14b"),
]

_RAM_FALLBACK_TABLE: list[tuple[float, str]] = [
    (12.0, "llama3.2:1b"),
    (24.0, "llama3.2:3b"),
    (48.0, "llama3.2:8b"),
    (float("inf"), "qwen2.5-coder:7b"),
]

_GOOD_SUBSTRINGS = ("coder", "llama", "qwen")


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _cli_available(binary: str) -> bool:
    """Return True if binary is on PATH and responds to --version with rc==0."""
    if shutil.which(binary) is None:
        return False
    try:
        result = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            timeout=5,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def _probe_ollama() -> tuple[bool, list[str]]:
    """Return (reachable, model_names)."""
    try:
        client = httpx.Client(timeout=2)
        resp = client.get("http://localhost:11434/")
        if resp.status_code != 200:
            return False, []
        tags_resp = client.get("http://localhost:11434/api/tags")
        data = tags_resp.json()
        models = [m["name"] for m in data.get("models", [])]
        return True, models
    except Exception:
        return False, []


def _pick_drafter(ram_gb: float, ollama_models: list[str]) -> str:
    """Choose the best drafter model given RAM and already-pulled models."""
    preferred = _RAM_DRAFTER_TABLE[0][1]
    fallback_preferred = _RAM_FALLBACK_TABLE[0][1]
    for max_ram, model in _RAM_DRAFTER_TABLE:
        if ram_gb < max_ram:
            preferred = model
            break
    for max_ram, model in _RAM_FALLBACK_TABLE:
        if ram_gb < max_ram:
            fallback_preferred = model
            break

    # If preferred is already pulled, use it exactly.
    if preferred in ollama_models:
        return preferred

    # Otherwise try the fallback preferred.
    if fallback_preferred in ollama_models:
        return fallback_preferred

    # Try any model with a good substring.
    for name in ollama_models:
        if any(sub in name for sub in _GOOD_SUBSTRINGS):
            return name

    # Nothing installed — return the preferred name so the user knows what to pull.
    return preferred


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_environment() -> DetectionResult:
    """Probe the host for available planners and drafters."""
    claude_ok = _cli_available("claude")
    codex_ok = _cli_available("codex")
    api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    ollama_ok, models = _probe_ollama()
    ram = get_total_ram_gb()

    # Planner priority
    if claude_ok:
        planner_type = "claude_code_cli"
    elif codex_ok:
        planner_type = "codex_cli"
    elif api_key:
        planner_type = "anthropic"
    else:
        warnings.warn(
            "No planner found (claude CLI, codex CLI, or ANTHROPIC_API_KEY). "
            "Falling back to degraded local-only mode.",
            RuntimeWarning,
            stacklevel=2,
        )
        planner_type = "ollama"

    drafter = _pick_drafter(ram, models)

    return DetectionResult(
        claude_cli_available=claude_ok,
        codex_cli_available=codex_ok,
        anthropic_api_key_present=api_key,
        ollama_reachable=ollama_ok,
        ollama_models=models,
        total_ram_gb=ram,
        recommended_drafter_model=drafter,
        recommended_planner_type=planner_type,
    )


# Provider-name → canonical YAML key mapping
_PROVIDER_YAML_KEY: dict[str, str] = {
    "claude_code_cli": "claude_pro",
    "codex_cli": "codex_plus",
    "anthropic": "anthropic_api",
    "ollama": "local_ollama",
}


def build_config_yaml(detection: DetectionResult) -> str:
    """Return a ready-to-write ~/.hierocode.yaml string based on detection."""
    drafter = detection.recommended_drafter_model or "llama3.2:3b"
    planner_type = detection.recommended_planner_type or "ollama"

    planner_provider_key = _PROVIDER_YAML_KEY.get(planner_type, "local_ollama")

    if planner_type == "ollama":
        planner_model = drafter
    else:
        planner_model = _PLANNER_MODEL.get(planner_type, "claude-sonnet-4-6")

    today = date.today().isoformat()

    return f"""\
# Generated by hierocode init --wizard on {today}
default_provider: local_ollama

providers:
  local_ollama:
    type: ollama
    base_url: http://localhost:11434
    auth:
      type: none
  claude_pro:
    type: claude_code_cli
    auth:
      type: none
  codex_plus:
    type: codex_cli
    auth:
      type: none
  anthropic_api:
    type: anthropic
    auth:
      type: bearer_env
      env_var: ANTHROPIC_API_KEY

routing:
  planner:
    provider: {planner_provider_key}
    model: {planner_model}
  drafter:
    provider: local_ollama
    model: {drafter}
  reviewer:
    provider: {planner_provider_key}
    model: {planner_model}

policy:
  max_revisions_per_unit: 2
  max_escalations_per_task: 3
  warn_before_escalation: true

parallelization:
  default_strategy: balanced
  max_local_workers: 4
  max_remote_workers: 8
"""


def run_wizard(write: bool = True, force: bool = False) -> tuple[DetectionResult, str]:
    """Main entry: detect, build config, optionally write. Returns (detection, yaml_string)."""
    detection = detect_environment()
    yaml_str = build_config_yaml(detection)

    if not write:
        return detection, yaml_str

    config_path = get_config_path()
    if config_path.exists() and not force:
        raise ConfigError(
            f"Config already exists at {config_path}. Pass force=True to overwrite."
        )

    config_path.write_text(yaml_str, encoding="utf-8")
    return detection, yaml_str
