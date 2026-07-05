"""Parse and apply SEARCH/REPLACE edit blocks emitted by the drafter.

Format:
<<<<<<< SEARCH
<exact lines copied from the current file>
=======
<replacement lines>
>>>>>>> REPLACE
"""

from dataclasses import dataclass
import re

@dataclass
class EditBlock:
    search: str
    replace: str

class EditApplyError(ValueError):
    pass

def parse_edit_blocks(text: str) -> list[EditBlock]:
    """Parse SEARCH/REPLACE blocks from drafter output. Returns [] if none found."""
    # We use a regex that is somewhat tolerant of trailing whitespace or exact marker casing.
    pattern = re.compile(
        r"<<<<<<<\s*SEARCH\n(.*?)\n?=======\n(.*?)\n?>>>>>>>\s*REPLACE",
        re.DOTALL
    )
    blocks = []
    for match in pattern.finditer(text):
        blocks.append(EditBlock(search=match.group(1), replace=match.group(2)))
    return blocks

def apply_edit_blocks(original: str, blocks: list[EditBlock]) -> str:
    """Apply blocks in order."""
    content = original
    for block in blocks:
        if not block.search and block.replace:
            # Append to end of file
            if content and not content.endswith("\n"):
                content += "\n"
            content += block.replace
            if not content.endswith("\n"):
                content += "\n"
            continue
            
        # 1. Exact match
        count = content.count(block.search)
        if count == 1:
            content = content.replace(block.search, block.replace)
            continue
        elif count > 1:
            snippet = block.search[:200]
            raise EditApplyError(f"SEARCH text not unique:\n{snippet}")
            
        # 2. Fallback: match ignoring trailing whitespace on each line
        # We must find the normalized search in the normalized content,
        # but we want to replace the ORIGINAL lines. This is tricky.
        # Instead, let's just use re.sub with optional trailing whitespace.
        regex_parts = []
        for line in block.search.splitlines():
            regex_parts.append(re.escape(line.rstrip()) + r"[ \t]*")
            
        regex = re.compile("\n".join(regex_parts), re.MULTILINE)
        matches = list(regex.finditer(content))
        if len(matches) == 1:
            content = content[:matches[0].start()] + block.replace + content[matches[0].end():]
            continue
        elif len(matches) > 1:
            snippet = block.search[:200]
            raise EditApplyError(f"SEARCH text not unique:\n{snippet}")
            
        snippet = block.search[:200]
        raise EditApplyError(f"SEARCH text not found in file / not unique:\n{snippet}")
        
    return content
