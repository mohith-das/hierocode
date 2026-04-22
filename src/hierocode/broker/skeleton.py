"""Build a compact AST-based repo skeleton for planner prompts."""

import ast
from pathlib import Path
from typing import Optional

_DEFAULT_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    ".venv312",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".tox",
}

_DUNDER_WHITELIST = {"__init__", "__call__"}


def _kb(size_bytes: int) -> int:
    """Round file size to KB; 0 for empty files, ceil-to-1 for non-empty."""
    if size_bytes == 0:
        return 0
    kb = size_bytes // 1024
    return kb if kb > 0 else 1


def _format_args(args: ast.arguments) -> str:
    """Format an ast.arguments node to a compact signature string."""
    try:
        return ast.unparse(args)
    except Exception:
        return "..."


def _format_returns(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Return ' -> Type' string if a return annotation exists, else empty string."""
    if node.returns is not None:
        try:
            return f" -> {ast.unparse(node.returns)}"
        except Exception:
            pass
    return ""


def _is_visible_method(name: str) -> bool:
    """Return True if the method name should appear in the skeleton."""
    if name.startswith("_"):
        return name in _DUNDER_WHITELIST
    return True


def _extract_symbols(source: str, indent: str) -> list[str]:
    """Parse Python source and return skeleton lines for top-level symbols."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _is_visible_method(node.name):
                continue
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            sig = _format_args(node.args)
            ret = _format_returns(node)
            lines.append(f"{indent}{prefix} {node.name}({sig}){ret}")
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases) if node.bases else ""
            class_header = f"{indent}class {node.name}({bases}):" if bases else f"{indent}class {node.name}:"
            lines.append(class_header)
            method_indent = indent + "  "
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not _is_visible_method(child.name):
                        continue
                    prefix = "async def" if isinstance(child, ast.AsyncFunctionDef) else "def"
                    sig = _format_args(child.args)
                    ret = _format_returns(child)
                    lines.append(f"{method_indent}{prefix} {child.name}({sig}){ret}: ...")
    return lines


def _walk_dir(
    directory: Path,
    repo_root: Path,
    depth: int,
    skip_dirs: set[str],
    file_counter: list[int],
    max_files: int,
    max_bytes: int,
    output: list[str],
) -> bool:
    """Recursively walk directory, appending to output. Returns True if truncated by max_bytes."""
    indent = "  " * depth

    try:
        entries = sorted(directory.iterdir(), key=lambda p: (0 if p.is_dir() else 1, p.name.lower()))
    except PermissionError:
        return False

    for entry in entries:
        if entry.is_dir():
            if entry.name in skip_dirs:
                continue
            dir_line = f"{indent}{entry.name}/"
            output.append(dir_line)
            if sum(len(ln) + 1 for ln in output) > max_bytes:
                output.append("... (output truncated)")
                return True
            if _walk_dir(entry, repo_root, depth + 1, skip_dirs, file_counter, max_files, max_bytes, output):
                return True
        elif entry.is_file():
            if file_counter[0] >= max_files:
                return False  # signal truncation upstream
            file_counter[0] += 1
            size_bytes = entry.stat().st_size
            file_line = f"{indent}{entry.name}  ({_kb(size_bytes)} KB)"
            output.append(file_line)
            if sum(len(ln) + 1 for ln in output) > max_bytes:
                output.append("... (output truncated)")
                return True

            if entry.suffix == ".py":
                try:
                    source = entry.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    source = ""
                sym_indent = indent + "  "
                symbols = _extract_symbols(source, sym_indent)
                output.extend(symbols)
                if sum(len(ln) + 1 for ln in output) > max_bytes:
                    output.append("... (output truncated)")
                    return True

    return False


def build_skeleton(
    repo_root: Path | str,
    max_files: int = 200,
    max_bytes: int = 20_000,
    skip_dirs: Optional[set[str]] = None,
) -> str:
    """Build a compact repo skeleton for planner prompts."""
    root = Path(repo_root).resolve()
    effective_skip = _DEFAULT_SKIP_DIRS | (skip_dirs or set())

    output: list[str] = [f"{root.name}/"]
    file_counter = [0]  # mutable int via list

    try:
        entries = sorted(root.iterdir(), key=lambda p: (0 if p.is_dir() else 1, p.name.lower()))
    except PermissionError:
        return "\n".join(output)

    truncated_by_bytes = False
    truncated_by_files = False

    for entry in entries:
        if entry.is_dir():
            if entry.name in effective_skip:
                continue
            dir_line = f"  {entry.name}/"
            output.append(dir_line)
            if sum(len(ln) + 1 for ln in output) > max_bytes:
                output.append("... (output truncated)")
                truncated_by_bytes = True
                break
            did_truncate = _walk_dir(
                entry, root, 2, effective_skip, file_counter, max_files, max_bytes, output
            )
            if did_truncate:
                truncated_by_bytes = True
                break
            # check file cap after walking sub-dir
            if file_counter[0] >= max_files:
                truncated_by_files = True
                break
        elif entry.is_file():
            if file_counter[0] >= max_files:
                truncated_by_files = True
                break
            file_counter[0] += 1
            size_bytes = entry.stat().st_size
            file_line = f"  {entry.name}  ({_kb(size_bytes)} KB)"
            output.append(file_line)
            if sum(len(ln) + 1 for ln in output) > max_bytes:
                output.append("... (output truncated)")
                truncated_by_bytes = True
                break
            if entry.suffix == ".py":
                try:
                    source = entry.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    source = ""
                symbols = _extract_symbols(source, "    ")
                output.extend(symbols)
                if sum(len(ln) + 1 for ln in output) > max_bytes:
                    output.append("... (output truncated)")
                    truncated_by_bytes = True
                    break

    if truncated_by_files and not truncated_by_bytes:
        # count remaining
        remaining = 0
        for entry in root.rglob("*"):
            if entry.is_file():
                rel = entry.relative_to(root)
                if any(part in effective_skip for part in rel.parts):
                    continue
                remaining += 1
        remaining -= file_counter[0]
        if remaining > 0:
            output.append(f"  ... (truncated: {remaining} more files)")

    return "\n".join(output)
