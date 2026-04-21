from hierocode.repo.files import read_file_safe
from hierocode.repo.slicer import slice_context

def build_file_context(filepath: str, max_chars: int = 15000) -> str:
    """Read a file and optionally slice/trim it if it's too big."""
    content = read_file_safe(filepath)
    return slice_context(content, max_chars)
