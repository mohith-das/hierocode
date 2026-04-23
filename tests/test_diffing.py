import os
import shutil
import subprocess
import tempfile

from hierocode.repo.diffing import generate_unified_diff


def test_generate_diff_same():
    orig = "a\nb\nc"
    diff = generate_unified_diff(orig, orig, "file.txt")
    assert not diff.strip()


def test_generate_diff_changed():
    orig = "a\nb\nc"
    mod = "a\nx\nc"
    diff = generate_unified_diff(orig, mod, "file.txt")
    assert "-b" in diff
    assert "+x" in diff


def test_generate_diff_ends_with_newline_when_content_has_none():
    """Regression: difflib.unified_diff emits no trailing newline when input lacks
    one, making `git apply` reject the patch as corrupt. generate_unified_diff
    must guarantee trailing newlines on both sides before diffing."""
    orig = ""
    mod = "<html><body></body></html>"  # deliberately no trailing newline
    diff = generate_unified_diff(orig, mod, "index.html")
    assert diff.endswith("\n"), f"Diff should end with newline; got {diff!r}"


def test_generated_diff_applies_cleanly_with_git_apply():
    """End-to-end: a diff from no-trailing-newline content should be accepted by
    `git apply --check` against an empty placeholder file."""
    git = shutil.which("git")
    if git is None:
        import pytest
        pytest.skip("git not installed")

    orig = ""
    mod = "line one\nline two\nline three"  # no trailing newline
    diff = generate_unified_diff(orig, mod, "file.txt")

    with tempfile.TemporaryDirectory() as td:
        # Placeholder file so git apply has a target to resolve.
        (open(os.path.join(td, "file.txt"), "w")).close()
        diff_path = os.path.join(td, "x.diff")
        with open(diff_path, "w") as f:
            f.write(diff)
        result = subprocess.run(
            [git, "apply", "--check", diff_path],
            cwd=td, capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, f"git apply --check failed: {result.stderr}"
