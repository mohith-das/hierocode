"""Context budgeter: pack TaskUnit files into the drafter's input-token budget."""

from pathlib import Path

from pydantic import BaseModel

from hierocode.broker.plan_schema import CapacityProfile, TaskUnit
from hierocode.repo.files import read_file_safe

_PROMPT_OVERHEAD = 800


class PackedContext(BaseModel):
    """Result of packing a TaskUnit's files into the drafter's token budget."""

    content: str
    included_files: list[str]
    truncated_files: list[str]
    skipped_files: list[str]
    estimated_tokens: int


def estimate_tokens(text: str) -> int:
    """Char/4 heuristic for drafter-side token estimation."""
    return len(text) // 4


def _format_block(path: str, content: str) -> str:
    """Wrap file content in standard header/footer markers."""
    return f"--- {path} ---\n{content}\n--- end {path} ---\n"


def pack_context(
    unit: TaskUnit,
    profile: CapacityProfile,
    repo_root: Path | str,
) -> PackedContext:
    """Pack target and context files into the drafter's available token budget."""
    repo_root = Path(repo_root)

    available = profile.max_input_tokens - _PROMPT_OVERHEAD
    if available <= 0:
        available = max(256, profile.max_input_tokens // 2)

    slots = profile.max_files_per_unit

    target_files: list[str] = list(unit.target_files or [])
    context_files: list[str] = list(unit.context_files or [])

    included: list[str] = []
    truncated: list[str] = []
    skipped: list[str] = []
    blocks: list[str] = []

    def _process(path: str, *, is_target: bool) -> None:
        nonlocal available, slots

        if slots <= 0:
            skipped.append(path)
            return

        raw = read_file_safe(repo_root / path)
        content = raw if raw != "" else "[file not found]"
        block = _format_block(path, content)
        tokens = estimate_tokens(block)

        if tokens <= available:
            blocks.append(block)
            included.append(path)
            available -= tokens
            slots -= 1
            return

        # Does not fit within budget.
        if not is_target:
            skipped.append(path)
            return

        # Target file: truncate to fit.
        char_budget = available * 4
        if char_budget <= 0:
            skipped.append(path)
            return

        overhead_block = _format_block(path, "")
        overhead_chars = len(overhead_block)
        content_chars = max(0, char_budget - overhead_chars)
        truncated_content = content[:content_chars]
        leftover = len(content) - content_chars
        suffix = f"\n... [truncated, {leftover} more chars] ...\n"
        block = _format_block(path, truncated_content + suffix)
        blocks.append(block)
        truncated.append(path)
        available -= estimate_tokens(block)
        slots -= 1

    for path in target_files:
        _process(path, is_target=True)

    for path in context_files:
        _process(path, is_target=False)

    packed_content = "\n".join(blocks)
    return PackedContext(
        content=packed_content,
        included_files=included,
        truncated_files=truncated,
        skipped_files=skipped,
        estimated_tokens=estimate_tokens(packed_content),
    )
