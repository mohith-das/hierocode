"""Tests for hierocode.broker.skeleton."""

from pathlib import Path

from hierocode.broker.skeleton import build_skeleton


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, rel: str, content: str = "") -> Path:
    """Write content to a file at tmp_path/rel, creating parents."""
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEmptyRepo:
    def test_empty_repo(self, tmp_path: Path):
        result = build_skeleton(tmp_path)
        assert result.strip() == f"{tmp_path.name}/"


class TestFileWithSimpleFunction:
    def test_file_with_simple_function(self, tmp_path: Path):
        _write(tmp_path, "foo.py", "def hello(x: int) -> str:\n    return str(x)\n")
        result = build_skeleton(tmp_path)
        assert "foo.py" in result
        assert "def hello(x: int) -> str" in result


class TestFileWithClassAndMethods:
    def test_file_with_class_and_methods(self, tmp_path: Path):
        src = (
            "class Widget:\n"
            "    def __init__(self, name: str) -> None:\n"
            "        self.name = name\n"
            "    def render(self) -> str:\n"
            "        return self.name\n"
            "    def _private(self) -> None:\n"
            "        pass\n"
        )
        _write(tmp_path, "widget.py", src)
        result = build_skeleton(tmp_path)
        assert "class Widget" in result
        assert "__init__" in result
        assert "render" in result
        assert "_private" not in result


class TestSyntaxErrorFallback:
    def test_syntax_error_fallback(self, tmp_path: Path):
        _write(tmp_path, "broken.py", "def (((broken syntax !!!")
        # Must not raise
        result = build_skeleton(tmp_path)
        assert "broken.py" in result
        # File size info should still appear
        assert "KB" in result
        # No extracted symbols should appear (no "def " lines beyond the file entry)
        lines_with_def = [ln for ln in result.splitlines() if ln.strip().startswith("def ")]
        assert lines_with_def == []


class TestNonPythonFilesListedOnly:
    def test_non_python_files_listed_only(self, tmp_path: Path):
        _write(tmp_path, "README.md", "# Hello\n")
        result = build_skeleton(tmp_path)
        assert "README.md" in result
        # Should show KB size notation
        assert "KB" in result
        # Should NOT show any code symbols
        assert "def " not in result
        assert "class " not in result


class TestSkipDefaultDirs:
    def test_skip_default_dirs(self, tmp_path: Path):
        _write(tmp_path, ".git/config", "[core]\n    bare = false\n")
        _write(tmp_path, "__pycache__/foo.pyc", b"\x00\x01\x02".decode("latin-1"))
        result = build_skeleton(tmp_path)
        assert ".git" not in result
        assert "__pycache__" not in result
        assert "foo.pyc" not in result


class TestSkipDirsMergedWithDefaults:
    def test_skip_dirs_merged_with_defaults(self, tmp_path: Path):
        _write(tmp_path, "docs/index.rst", "Title\n=====\n")
        _write(tmp_path, ".git/config", "[core]\n")
        _write(tmp_path, "src/main.py", "x = 1\n")
        result = build_skeleton(tmp_path, skip_dirs={"docs"})
        assert "docs" not in result
        assert ".git" not in result
        assert "main.py" in result


class TestMaxFilesTruncates:
    def test_max_files_truncates(self, tmp_path: Path):
        for i in range(10):
            _write(tmp_path, f"file{i:02d}.py", f"x = {i}\n")
        result = build_skeleton(tmp_path, max_files=3)
        # Count how many .py file entries appear (lines containing ".py")
        py_lines = [ln for ln in result.splitlines() if ".py" in ln and "KB" in ln]
        assert len(py_lines) == 3
        assert "truncated" in result


class TestFileSizeInKb:
    def test_file_2048_bytes_shows_2_kb(self, tmp_path: Path):
        _write(tmp_path, "big.py", "x" * 2048)
        result = build_skeleton(tmp_path)
        assert "(2 KB)" in result

    def test_empty_file_shows_0_kb(self, tmp_path: Path):
        _write(tmp_path, "empty.py", "")
        result = build_skeleton(tmp_path)
        assert "(0 KB)" in result

    def test_small_nonempty_file_shows_at_least_1_kb(self, tmp_path: Path):
        # 500 bytes is non-empty but < 1 KB; our rule: ceil to 1 for non-empty
        _write(tmp_path, "small.py", "x" * 500)
        result = build_skeleton(tmp_path)
        assert "(1 KB)" in result


class TestSortedDirsBeforeFiles:
    def test_sorted_dirs_before_files(self, tmp_path: Path):
        _write(tmp_path, "zebra.py", "z = 1\n")
        _write(tmp_path, "aaa/init.py", "a = 1\n")
        result = build_skeleton(tmp_path)
        aaa_pos = result.index("aaa/")
        zebra_pos = result.index("zebra.py")
        assert aaa_pos < zebra_pos


class TestRespectsMaxBytes:
    def test_respects_max_bytes(self, tmp_path: Path):
        for i in range(20):
            _write(tmp_path, f"module{i:02d}.py", f"def func_{i}(): pass\n" * 10)
        result = build_skeleton(tmp_path, max_bytes=200)
        # Allow a small buffer for the truncation marker line itself
        assert len(result) <= 300
        assert "truncated" in result


class TestAsyncFunctionsIncluded:
    def test_async_functions_included(self, tmp_path: Path):
        _write(tmp_path, "fetcher.py", "async def fetch() -> None:\n    pass\n")
        result = build_skeleton(tmp_path)
        assert "async def fetch" in result


class TestRepoRootAcceptsStringOrPath:
    def test_repo_root_accepts_string_or_path(self, tmp_path: Path):
        _write(tmp_path, "mod.py", "def foo(): pass\n")
        result_path = build_skeleton(tmp_path)
        result_str = build_skeleton(str(tmp_path))
        assert result_path == result_str
