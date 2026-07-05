"""MCP server exposing hierocode's local drafter to MCP-capable coding agents."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from hierocode.broker.capacity import build_capacity_profile
from hierocode.broker.router import get_route
from hierocode.broker.usage import UsageAccumulator
from hierocode.config import HierocodeConfig, load_config
from hierocode.engine import draft_unit
from hierocode.exceptions import ConfigError
from hierocode.providers import get_provider

mcp = FastMCP("hierocode")

_config: Optional[HierocodeConfig] = None
_usage = UsageAccumulator()

def _get_config() -> HierocodeConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


@mcp.tool()
def draft_code(
    goal: str,
    target_file: str,
    context_files: list[str] | None = None,
    acceptance: str = "",
    repo_root: str = ".",
) -> str:
    """Delegate a bounded, single-file code edit to the local drafter model and
    get back a unified diff. Costs $0 — the draft runs on the user's own hardware.

    WHEN TO USE: mechanical, well-specified edits confined to one file —
    boilerplate, renames, adding a function/test/endpoint that follows an obvious
    pattern, filling in a stub. The more precise `goal` and `acceptance` are, the
    better the draft.

    WHEN NOT TO USE: multi-file changes, edits needing cross-file reasoning,
    architectural decisions, or anything you'd want a strong model to write.
    Do those yourself.

    YOU are the reviewer: read the returned diff critically before applying it
    (e.g. `git apply` or your own edit tools). The diff is generated against the
    file's current on-disk content. Paths are relative to repo_root, which
    defaults to the server's working directory.

    Returns JSON: {status, diff, drafter_model, warnings, input_tokens,
    output_tokens, error_type?, message?, suggestion?}.
    """
    try:
        config = _get_config()
        if not Path(repo_root).is_dir():
            return json.dumps({
                "status": "error",
                "error_type": "config",
                "message": f"repo_root '{repo_root}' is not a valid directory",
                "suggestion": "provide a valid repository root directory"
            })
            
        result = draft_unit(
            goal=goal,
            target_file=target_file,
            repo_root=repo_root,
            context_files=context_files,
            acceptance=acceptance,
            config=config,
            usage=_usage
        )
        return json.dumps(asdict(result))
    except Exception as exc:
        return json.dumps({
            "status": "error",
            "error_type": "unknown",
            "message": str(exc),
            "suggestion": "Check server logs or configuration"
        })


@mcp.tool()
def drafter_info() -> str:
    """Report the local drafter's capability envelope: model name, parameter
    count, context window, max input/output tokens per draft, and whether the
    provider is currently reachable. Call this once before delegating to size
    your requests (a 3B model needs smaller, more precise goals than a 14B one)."""
    try:
        config = _get_config()
        provider_name, model_name = get_route(config, "drafter")
        provider_config = config.providers[provider_name]
        provider = get_provider(provider_name, provider_config)
        profile = build_capacity_profile(provider, model_name)
        
        info = {
            "status": "ok",
            "drafter_model": profile.drafter_model,
            "param_count_b": profile.param_count_b,
            "quantization": profile.quantization,
            "context_window": profile.context_window,
            "max_input_tokens": profile.max_input_tokens,
            "max_output_tokens": profile.max_output_tokens,
            "reachable": provider.healthcheck(),
        }
        return json.dumps(info)
    except Exception as exc:
        return json.dumps({
            "status": "error",
            "error_type": "config" if isinstance(exc, ConfigError) else "unknown",
            "message": str(exc),
            "suggestion": "Check provider logs or configuration"
        })


@mcp.tool()
def usage_summary() -> str:
    """Token usage accumulated by this server session, by role. Use it to report
    to the user how much drafting was delegated to the local model."""
    try:
        totals = {}
        for role in ["planner", "drafter", "reviewer"]:
            summary = getattr(_usage, role)
            totals[role] = {
                "input_tokens": summary.input_tokens,
                "output_tokens": summary.output_tokens,
                "cache_creation_input_tokens": summary.cache_creation_input_tokens,
                "cache_read_input_tokens": summary.cache_read_input_tokens,
                "messages": summary.messages,
            }
        return json.dumps({"status": "ok", "usage": totals})
    except Exception as exc:
        return json.dumps({
            "status": "error",
            "error_type": "unknown",
            "message": str(exc)
        })


def main() -> None:
    """Run the stdio MCP server (entry point for `hierocode mcp`)."""
    mcp.run()
