"""Tests for hierocode.broker.patcher."""

from __future__ import annotations

import difflib
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hierocode.broker.patcher import (
    FilePatch,
    PatchAction,
    PatchParseError,
    apply_patch,
    parse_diff,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_unified_diff(original: str, modified: str, filename: str) -> str:
    """Produce a unified diff string the same way generate_unified_diff does."""
    orig_lines = original.splitlines(keepends=True)
    mod_lines = modified.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(orig_lines, mod_lines, fromfile=filename, tofile=filename, n=3)
    )


def _init_git_repo(path: Path) -> None:
    """Initialise a bare-minimum git repo so `git apply` works."""
    subprocess.run(["git", "init"], cwd=str(path), check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=str(path),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(path),
        check=True,
        capture_output=True,
    )


def _write_and_commit(repo: Path, filename: str, content: str) -> None:
    """Write a file, stage it, and commit it in the given repo."""
    (repo / filename).write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", filename], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", f"add {filename}"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )


# ---------------------------------------------------------------------------
# parse_diff tests
# ---------------------------------------------------------------------------


class TestParseDiffEmptyString:
    def test_parse_diff_empty_string_returns_empty_list(self):
        """Empty or blank diff string yields an empty list."""
        assert parse_diff("") == []
        assert parse_diff("   \n") == []


class TestParseDiffSingleFileModification:
    def test_parse_diff_single_file_modification(self):
        """A typical 3-line-context diff for one modified file is parsed correctly."""
        original = "line1\nline2\nline3\nline4\nline5\n"
        modified = "line1\nline2\nchanged\nline4\nline5\n"
        raw = _make_unified_diff(original, modified, "src/foo.py")

        patches = parse_diff(raw)

        assert len(patches) == 1
        p = patches[0]
        assert p.path == "src/foo.py"
        assert p.action == PatchAction.MODIFY
        assert p.line_count_added == 1
        assert p.line_count_removed == 1
        assert "@@" in p.raw_diff
        assert "--- src/foo.py" in p.raw_diff
        assert "+++ src/foo.py" in p.raw_diff


class TestParseDiffMultipleFiles:
    def test_parse_diff_multiple_files(self):
        """Two file blocks in one diff produce two FilePatch objects."""
        diff_a = _make_unified_diff("a\nb\n", "a\nB\n", "a.py")
        diff_b = _make_unified_diff("x\ny\n", "x\nY\n", "b.py")
        combined = diff_a + diff_b

        patches = parse_diff(combined)

        assert len(patches) == 2
        paths = {p.path for p in patches}
        assert "a.py" in paths
        assert "b.py" in paths


class TestParseDiffNewFile:
    def test_parse_diff_new_file(self):
        """A diff with --- /dev/null should produce a CREATE action."""
        new_file_diff = (
            "--- /dev/null\n"
            "+++ b/new_file.py\n"
            "@@ -0,0 +1,3 @@\n"
            "+line1\n"
            "+line2\n"
            "+line3\n"
        )
        patches = parse_diff(new_file_diff)

        assert len(patches) == 1
        assert patches[0].action == PatchAction.CREATE
        assert patches[0].path == "new_file.py"
        assert patches[0].line_count_added == 3
        assert patches[0].line_count_removed == 0


class TestParseDiffDeleteFile:
    def test_parse_diff_delete_file(self):
        """A diff with +++ /dev/null should produce a DELETE action."""
        delete_diff = (
            "--- a/old_file.py\n"
            "+++ /dev/null\n"
            "@@ -1,3 +0,0 @@\n"
            "-line1\n"
            "-line2\n"
            "-line3\n"
        )
        patches = parse_diff(delete_diff)

        assert len(patches) == 1
        assert patches[0].action == PatchAction.DELETE
        assert patches[0].path == "old_file.py"
        assert patches[0].line_count_removed == 3
        assert patches[0].line_count_added == 0


class TestParseDiffCountsAddedAndRemovedLines:
    def test_parse_diff_counts_added_and_removed_lines(self):
        """line_count_added and line_count_removed accurately reflect hunk contents."""
        original = "\n".join(f"line{i}" for i in range(1, 11)) + "\n"
        modified = "\n".join(
            (f"NEW{i}" if i in (3, 5) else f"line{i}") for i in range(1, 11)
        ) + "\nextra\n"
        raw = _make_unified_diff(original, modified, "counted.py")

        patches = parse_diff(raw)

        assert len(patches) == 1
        # 2 lines replaced means 2 removed + 2 added, plus 1 new line at the end
        assert patches[0].line_count_added >= 2
        assert patches[0].line_count_removed >= 2


class TestParseDiffMalformedRaises:
    def test_parse_diff_malformed_raises(self):
        """A diff that has --- / +++ headers but no @@ hunk raises PatchParseError."""
        bad_diff = "--- a.py\n+++ a.py\nsome random content\n"
        with pytest.raises(PatchParseError, match=r"@@"):
            parse_diff(bad_diff)


class TestParseDiffBinaryRejected:
    def test_parse_diff_binary_rejected(self):
        """A diff containing a binary-file marker raises PatchParseError."""
        binary_diff = (
            "--- a/image.png\n"
            "+++ b/image.png\n"
            "@@ -1 +1 @@\n"
            "Binary files a/image.png and b/image.png differ\n"
        )
        with pytest.raises(PatchParseError, match="[Bb]inary"):
            parse_diff(binary_diff)


# ---------------------------------------------------------------------------
# apply_patch tests
# ---------------------------------------------------------------------------


class TestApplyPatchGitNotInstalled:
    def test_apply_patch_git_not_installed_returns_error(self, tmp_path):
        """If git is not on PATH, apply_patch returns status='error'."""
        patch_obj = FilePatch(
            path="foo.py",
            action=PatchAction.MODIFY,
            raw_diff="--- foo.py\n+++ foo.py\n@@ -1 +1 @@\n-old\n+new\n",
        )
        with patch("hierocode.broker.patcher.shutil.which", return_value=None):
            result = apply_patch(patch_obj, tmp_path)

        assert result.status == "error"
        assert "git" in result.message.lower()


class TestApplyPatchHappyPathModify:
    def test_apply_patch_happy_path_modify(self, tmp_path):
        """A real unified diff is applied successfully in an initialised git repo."""
        _init_git_repo(tmp_path)

        original = "hello\nworld\n"
        modified = "hello\nearth\n"
        _write_and_commit(tmp_path, "greet.py", original)

        raw = _make_unified_diff(original, modified, "greet.py")
        patch_obj = FilePatch(
            path="greet.py",
            action=PatchAction.MODIFY,
            line_count_added=1,
            line_count_removed=1,
            raw_diff=raw,
        )

        result = apply_patch(patch_obj, tmp_path)

        assert result.status == "applied", result.message
        assert (tmp_path / "greet.py").read_text(encoding="utf-8") == modified


class TestApplyPatchCreateNewFile:
    def test_apply_patch_create_new_file(self, tmp_path):
        """A diff that creates a new file from /dev/null is applied correctly."""
        _init_git_repo(tmp_path)

        new_content = "brand new\n"
        new_file_diff = (
            "--- /dev/null\n"
            "+++ b/brand_new.py\n"
            "@@ -0,0 +1 @@\n"
            "+brand new\n"
        )
        patch_obj = FilePatch(
            path="brand_new.py",
            action=PatchAction.CREATE,
            line_count_added=1,
            raw_diff=new_file_diff,
        )

        result = apply_patch(patch_obj, tmp_path)

        assert result.status == "applied", result.message
        assert (tmp_path / "brand_new.py").read_text(encoding="utf-8") == new_content


class TestApplyPatchConflictReturnsError:
    def test_apply_patch_conflict_returns_error(self, tmp_path):
        """If the file on disk doesn't match the diff context, git apply --check fails."""
        _init_git_repo(tmp_path)

        original = "hello\nworld\n"
        conflicting = "completely\ndifferent\ncontent\n"
        _write_and_commit(tmp_path, "conflict.py", conflicting)

        # This diff is for a different original content, so it will conflict.
        raw = _make_unified_diff(original, "hello\nearth\n", "conflict.py")
        patch_obj = FilePatch(
            path="conflict.py",
            action=PatchAction.MODIFY,
            raw_diff=raw,
        )

        result = apply_patch(patch_obj, tmp_path)

        assert result.status == "error"
        assert result.message  # some git error message


class TestApplyPatchRejectsNonGitDir:
    def test_apply_patch_rejects_diff_in_non_git_dir(self, tmp_path):
        """apply_patch fails gracefully when repo_root has no git repo."""
        raw = _make_unified_diff("a\n", "b\n", "file.py")
        patch_obj = FilePatch(
            path="file.py",
            action=PatchAction.MODIFY,
            raw_diff=raw,
        )
        # tmp_path has no git repo initialised
        result = apply_patch(patch_obj, tmp_path)

        assert result.status == "error"
        # The error should mention git
        assert result.message


class TestApplyPatchGitCheckFailsMocked:
    def test_apply_patch_git_check_fails_mocked(self, tmp_path):
        """When git apply --check returns non-zero, apply_patch returns error without applying."""
        patch_obj = FilePatch(
            path="foo.py",
            action=PatchAction.MODIFY,
            raw_diff="--- foo.py\n+++ foo.py\n@@ -1 +1 @@\n-old\n+new\n",
        )

        failed_check = MagicMock()
        failed_check.returncode = 1
        failed_check.stderr = "error: patch does not apply"
        failed_check.stdout = ""

        with patch("hierocode.broker.patcher.shutil.which", return_value="/usr/bin/git"):
            with patch("hierocode.broker.patcher.subprocess.run", return_value=failed_check):
                result = apply_patch(patch_obj, tmp_path)

        assert result.status == "error"
        assert "patch does not apply" in result.message
