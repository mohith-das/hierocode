"""Headless engine exposing hierocode's core capabilities without the TUI."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from hierocode.broker.budget import pack_context
from hierocode.broker.capacity import build_capacity_profile
from hierocode.broker.edits import EditApplyError
from hierocode.broker.plan_schema import TaskUnit
from hierocode.broker.prompts import build_drafter_prompt, build_drafter_revision_prompt
from hierocode.broker.router import get_route
from hierocode.broker.usage import UsageAccumulator, UsageInfo
from hierocode.config import HierocodeConfig, load_config
from hierocode.exceptions import ConfigError, ModelNotFoundError, ProviderConnectionError
from hierocode.providers import get_provider


@dataclass
class DraftResult:
    status: Literal["ok", "error"]
    diff: Optional[str] = None
    drafter_model: str = ""
    included_files: list[str] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    error_type: Optional[str] = None  # "config" | "provider" | "budget" | "empty" | "edit_apply"
    error_message: Optional[str] = None
    suggestion: Optional[str] = None


def draft_unit(
    goal: str,
    target_file: str,
    repo_root: Path | str,
    context_files: Optional[list[str]] = None,
    acceptance: str = "",
    config: Optional[HierocodeConfig] = None,
    usage: Optional[UsageAccumulator] = None,
) -> DraftResult:
    """Execute a single-file edit draft silently, returning a structured result."""
    try:
        config = config or load_config()
    except Exception as exc:
        return DraftResult(
            status="error",
            error_type="config",
            error_message=str(exc),
            suggestion="Fix your hierocode config file",
        )

    try:
        provider_name, model_name = get_route(config, "drafter")
        provider_config = config.providers[provider_name]
        provider = get_provider(provider_name, provider_config)
    except ConfigError as exc:
        return DraftResult(
            status="error",
            error_type="config",
            error_message=str(exc),
            suggestion="Check your router configuration",
        )

    profile = build_capacity_profile(provider, model_name)
    
    unit = TaskUnit(
        id="mcp-1",
        goal=goal,
        target_files=[target_file],
        context_files=context_files or [],
        acceptance=acceptance,
    )
    
    packed = pack_context(unit, profile, repo_root)
    
    if packed.infeasible_targets:
        return DraftResult(
            status="error",
            error_type="budget",
            error_message=f"target file(s) {packed.infeasible_targets} exceed the drafter's input budget",
            suggestion="split the change into smaller goals or reduce context_files",
        )

    warnings = []
    for skipped in packed.skipped_files:
        warnings.append(f"context file {skipped} skipped: over budget")

    # Mode logic
    original = ""
    # In engine, file_state is empty since it's a one-off run
    file_state: dict[str, str] = {}
    from hierocode.broker.dispatcher import _normalize_target, _current_content
    target = _normalize_target(unit.target_files[0], repo_root)
    original = _current_content(target, repo_root, file_state)

    mode: Literal["whole_file", "edit_blocks"] = "whole_file" if original == "" else "edit_blocks"
    prompt = build_drafter_prompt(unit, packed.content, mode=mode)
    
    from hierocode.broker.edits import parse_edit_blocks, apply_edit_blocks
    from hierocode.repo.diffing import generate_unified_diff
    from hierocode.broker.dispatcher import _strip_code_fences

    def _do_draft(p: str) -> tuple[str, str, bool]:
        drafted = provider.generate(prompt=p, model=model_name, max_tokens=profile.max_output_tokens)
        cleaned = _strip_code_fences(drafted)
        blocks = parse_edit_blocks(cleaned)
        used_blocks = len(blocks) > 0
        if blocks:
            new_content = apply_edit_blocks(original, blocks)
        else:
            new_content = cleaned
        diff_str = generate_unified_diff(original, new_content, target)
        return diff_str or "", new_content, used_blocks

    def _consume_usage() -> tuple[int, int]:
        # Records the most recent provider call; when a retry happened, the first
        # call's usage is dropped (same accepted limitation as qa.review_draft).
        if isinstance(provider.last_usage, UsageInfo):
            if usage is not None:
                usage.record("drafter", provider.last_usage)
            return provider.last_usage.input_tokens, provider.last_usage.output_tokens
        return 0, 0

    try:
        try:
            diff, new_content, used_blocks = _do_draft(prompt)
        except EditApplyError as exc:
            revision_prompt = build_drafter_revision_prompt(
                unit, packed.content, "", f"Your edit blocks could not be applied: {exc}. Re-emit the blocks with SEARCH text copied exactly from the file.", mode=mode
            )
            try:
                diff, new_content, used_blocks = _do_draft(revision_prompt)
            except EditApplyError as exc_retry:
                input_tokens, output_tokens = _consume_usage()
                return DraftResult(
                    status="error",
                    error_type="edit_apply",
                    drafter_model=model_name,
                    included_files=packed.included_files,
                    skipped_files=packed.skipped_files,
                    warnings=warnings,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    error_message=(
                        f"the drafter's edit blocks could not be applied after one retry: {exc_retry}"
                    ),
                    suggestion="handle this edit yourself or configure a larger drafter model",
                )

        if mode == "edit_blocks" and not used_blocks:
            warnings.append("drafter ignored edit-block format; whole-file replacement used")

        input_tokens, output_tokens = _consume_usage()

        if diff is None or not diff.strip():
            return DraftResult(
                status="error",
                error_type="empty",
                drafter_model=model_name,
                included_files=packed.included_files,
                skipped_files=packed.skipped_files,
                warnings=warnings,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                error_message="The drafter returned an empty diff",
                suggestion="the drafter produced no change; rephrase the goal or handle this edit yourself",
            )
            
        return DraftResult(
            status="ok",
            diff=diff,
            drafter_model=model_name,
            included_files=packed.included_files,
            skipped_files=packed.skipped_files,
            warnings=warnings,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    except (ProviderConnectionError, ModelNotFoundError, ConfigError) as exc:
        err_type = "config" if isinstance(exc, ConfigError) else "provider"
        return DraftResult(
            status="error",
            error_type=err_type,
            error_message=str(exc),
            suggestion="Check provider logs or configuration",
        )
