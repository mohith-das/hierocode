import difflib


def _ensure_trailing_newline(text: str) -> str:
    """Append '\\n' if missing. Prevents git apply from rejecting the diff with
    'corrupt patch' when difflib emits a final '+line' without a trailing
    newline and no '\\ No newline at end of file' marker."""
    if not text:
        return text
    return text if text.endswith("\n") else text + "\n"


def generate_unified_diff(original: str, modified: str, filename: str) -> str:
    """Generate a unified diff representation of code changes.

    Both `original` and `modified` are normalized to end with a newline before
    diffing — required for `git apply` compatibility on new/modified files.
    """
    original = _ensure_trailing_newline(original)
    modified = _ensure_trailing_newline(modified)
    original_lines = original.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile=filename,
        tofile=filename,
        n=3,
    )
    return "".join(diff)
