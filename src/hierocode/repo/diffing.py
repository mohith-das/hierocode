import difflib

def generate_unified_diff(original: str, modified: str, filename: str) -> str:
    """Generate a unified diff representation of code changes."""
    original_lines = original.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)
    
    diff = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile=filename,
        tofile=filename,
        n=3
    )
    return "".join(diff)
