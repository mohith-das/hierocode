from pathlib import Path

def read_file_safe(path: Path | str) -> str:
    """Safely read a file, catching typical encoding issues."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        return "[Binary or non-UTF-8 content]"
    except FileNotFoundError:
        return ""
