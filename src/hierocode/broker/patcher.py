"""Parse unified diffs and apply them to disk via `git apply`."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class PatchAction(str, Enum):
    """The kind of change a FilePatch represents."""

    MODIFY = "modify"
    CREATE = "create"
    DELETE = "delete"


@dataclass
class FilePatch:
    """A single-file unit extracted from a unified diff."""

    path: str  # relative path as it appears in the diff header
    action: PatchAction
    line_count_added: int = 0
    line_count_removed: int = 0
    raw_diff: str = field(default="")  # the chunk for THIS file (--- and +++ lines included)


@dataclass
class ApplyResult:
    """Result of applying one FilePatch."""

    path: str
    status: str  # "applied" | "skipped" | "error"
    message: Optional[str] = None  # error text when status == "error"


class PatchParseError(ValueError):
    """Raised when a diff cannot be cleanly parsed."""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_DEV_NULL = "/dev/null"


def _strip_diff_path(raw_path: str) -> str:
    """Strip the `a/` / `b/` prefixes that `git diff` prepends, if present."""
    for prefix in ("a/", "b/"):
        if raw_path.startswith(prefix):
            return raw_path[len(prefix):]
    return raw_path


def parse_diff(raw_diff: str) -> list[FilePatch]:
    """Parse a unified diff into a list of FilePatch objects.

    Supports diffs produced by difflib.unified_diff and git diff.
    Raises PatchParseError for malformed or unsupported (e.g. binary) diffs.
    """
    if not raw_diff or not raw_diff.strip():
        return []

    patches: list[FilePatch] = []
    lines = raw_diff.splitlines(keepends=True)

    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # Skip index / diff --git header lines
        if line.startswith("diff --git") or line.startswith("index "):
            i += 1
            continue

        if not line.startswith("--- "):
            i += 1
            continue

        # We found the start of a file block.
        from_line = line.rstrip("\n\r")
        if i + 1 >= n or not lines[i + 1].startswith("+++ "):
            raise PatchParseError(
                f"Expected '+++ ' after '--- ' at line {i + 1}, got: {lines[i + 1]!r}"
            )
        to_line = lines[i + 1].rstrip("\n\r")

        from_path = from_line[4:].split("\t")[0].strip()
        to_path = to_line[4:].split("\t")[0].strip()

        # Reject binary diffs
        if from_path == "/dev/null" and to_path == "/dev/null":
            raise PatchParseError("Both --- and +++ point to /dev/null; cannot parse.")

        # Determine action
        if from_path == _DEV_NULL or from_path.endswith("/dev/null"):
            action = PatchAction.CREATE
            canonical_path = _strip_diff_path(to_path)
        elif to_path == _DEV_NULL or to_path.endswith("/dev/null"):
            action = PatchAction.DELETE
            canonical_path = _strip_diff_path(from_path)
        else:
            action = PatchAction.MODIFY
            canonical_path = _strip_diff_path(to_path)

        # Consume the hunk lines for this file.
        chunk_lines: list[str] = [lines[i], lines[i + 1]]
        added = 0
        removed = 0
        j = i + 2

        # Walk forward until the next file header or EOF.
        while j < n:
            curr = lines[j]
            if curr.startswith("--- ") and j + 1 < n and lines[j + 1].startswith("+++ "):
                # Next file block starts here.
                break
            if curr.startswith("diff --git") or curr.startswith("index "):
                break

            # Check for binary diff marker
            if "binary files" in curr.lower() and "differ" in curr.lower():
                raise PatchParseError(
                    f"Binary diff detected for {canonical_path!r}; /apply does not support binary files."
                )

            chunk_lines.append(curr)

            # Require at least one @@ hunk header
            if curr.startswith("+") and not curr.startswith("+++ "):
                added += 1
            elif curr.startswith("-") and not curr.startswith("--- "):
                removed += 1

            j += 1

        raw_chunk = "".join(chunk_lines)

        # Validate: must contain at least one @@ hunk header (unless it's a pure
        # delete with no hunks — but standard unified diffs always have @@ lines).
        if "@@" not in raw_chunk:
            raise PatchParseError(
                f"No hunk header (@@ ... @@) found for file {canonical_path!r}."
            )

        patches.append(
            FilePatch(
                path=canonical_path,
                action=action,
                line_count_added=added,
                line_count_removed=removed,
                raw_diff=raw_chunk,
            )
        )

        i = j  # advance past the consumed lines

    return patches


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def apply_patch(patch: FilePatch, repo_root: Path | str) -> ApplyResult:
    """Apply a single FilePatch under repo_root using `git apply`.

    Steps:
    1. Verify git is available on PATH.
    2. Write patch.raw_diff to a temp file.
    3. Run `git apply --check` (dry-run).
    4. If check passes, run `git apply`.

    Returns ApplyResult with status "applied" or "error".
    """
    repo_root = Path(repo_root)

    if shutil.which("git") is None:
        return ApplyResult(
            path=patch.path,
            status="error",
            message="git not found; /apply requires git on PATH.",
        )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".patch", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(patch.raw_diff)
        tmp_path = tmp.name

    try:
        # Dry-run first.
        check_result = subprocess.run(
            ["git", "apply", "--check", tmp_path],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
        if check_result.returncode != 0:
            stderr = (check_result.stderr or check_result.stdout or "").strip()
            return ApplyResult(
                path=patch.path,
                status="error",
                message=f"git apply --check failed: {stderr}",
            )

        # Real apply.
        apply_result = subprocess.run(
            ["git", "apply", tmp_path],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
        if apply_result.returncode != 0:
            stderr = (apply_result.stderr or apply_result.stdout or "").strip()
            return ApplyResult(
                path=patch.path,
                status="error",
                message=f"git apply failed: {stderr}",
            )

        return ApplyResult(path=patch.path, status="applied")

    finally:
        Path(tmp_path).unlink(missing_ok=True)
