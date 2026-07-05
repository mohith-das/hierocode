"""Parse and apply SEARCH/REPLACE edit blocks emitted by the drafter.

Format:
<<<<<<< SEARCH
<exact lines copied from the current file>
=======
<replacement lines>
>>>>>>> REPLACE
"""

import re
from dataclasses import dataclass


@dataclass
class EditBlock:
    search: str
    replace: str


class EditApplyError(ValueError):
    """Raised when edit blocks are malformed or cannot be applied to the file."""


_SEARCH_MARKER = re.compile(r"^<{5,9}\s*SEARCH\s*$")
_DIVIDER = re.compile(r"^={5,9}$")
_REPLACE_MARKER = re.compile(r"^>{5,9}\s*REPLACE\s*$")


def parse_edit_blocks(text: str) -> list[EditBlock]:
    """Parse SEARCH/REPLACE blocks from drafter output.

    Marker lines are hard boundaries — a block's SEARCH or REPLACE text can never
    span into a neighbouring block. Returns [] when no SEARCH markers are present
    (callers fall back to whole-file output). A block that opens but is malformed
    (missing its ``=======`` divider or ``>>>>>>> REPLACE`` end marker) raises
    EditApplyError so the drafter can be asked to re-emit it.
    """
    blocks: list[EditBlock] = []
    state = "text"  # "text" | "search" | "replace"
    search_lines: list[str] = []
    replace_lines: list[str] = []

    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if _SEARCH_MARKER.match(stripped):
            if state != "text":
                raise EditApplyError(
                    f"malformed edit block at line {lineno}: new '<<<<<<< SEARCH' opened"
                    " before the previous block was closed with '>>>>>>> REPLACE'"
                )
            state = "search"
            search_lines, replace_lines = [], []
        elif _REPLACE_MARKER.match(stripped):
            if state != "replace":
                raise EditApplyError(
                    f"malformed edit block at line {lineno}: '>>>>>>> REPLACE' without"
                    " a '=======' divider between SEARCH and REPLACE text"
                )
            blocks.append(
                EditBlock(search="\n".join(search_lines), replace="\n".join(replace_lines))
            )
            state = "text"
        elif state == "search" and _DIVIDER.match(stripped):
            state = "replace"
        elif state == "search":
            search_lines.append(line)
        elif state == "replace":
            replace_lines.append(line)

    if state != "text":
        raise EditApplyError(
            "malformed edit block: output ended before '>>>>>>> REPLACE'"
        )
    return blocks


def apply_edit_blocks(original: str, blocks: list[EditBlock]) -> str:
    """Apply blocks in order: exact unique match, then a trailing-whitespace-insensitive
    fallback; an empty SEARCH with non-empty REPLACE appends to the end of the file."""
    content = original
    for block in blocks:
        if not block.search and block.replace:
            if content and not content.endswith("\n"):
                content += "\n"
            content += block.replace
            if not content.endswith("\n"):
                content += "\n"
            continue

        count = content.count(block.search)
        if count == 1:
            content = content.replace(block.search, block.replace)
            continue
        elif count > 1:
            snippet = block.search[:200]
            raise EditApplyError(f"SEARCH text not unique:\n{snippet}")

        # Fallback: match ignoring trailing whitespace on each line, replacing the
        # original (whitespace-preserving) span.
        regex_parts = [re.escape(line.rstrip()) + r"[ \t]*" for line in block.search.splitlines()]
        regex = re.compile("\n".join(regex_parts))
        matches = list(regex.finditer(content))
        if len(matches) == 1:
            content = content[: matches[0].start()] + block.replace + content[matches[0].end():]
            continue
        elif len(matches) > 1:
            snippet = block.search[:200]
            raise EditApplyError(f"SEARCH text not unique:\n{snippet}")

        snippet = block.search[:200]
        raise EditApplyError(f"SEARCH text not found in file / not unique:\n{snippet}")

    return content
