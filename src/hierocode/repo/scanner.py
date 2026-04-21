import os
from pathlib import Path
from typing import List

def find_files(root: str, extensions: List[str] = None) -> List[Path]:
    """Find files in the repository, respecting simple ignores."""
    found = []
    ignore_dirs = {".git", "venv", ".venv", "__pycache__", "node_modules", "dist", "build"}
    for p, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for f in files:
            if extensions and not any(f.endswith(ext) for ext in extensions):
                continue
            path = Path(p) / f
            found.append(path)
    return found
