"""Shared skip markers for hierocode integration tests."""

import os
import shutil
import subprocess

import httpx
import pytest


def _ollama_alive() -> bool:
    try:
        return httpx.Client(timeout=2).get("http://localhost:11434/").status_code == 200
    except Exception:
        return False


def _cli_alive(binary: str) -> bool:
    if shutil.which(binary) is None:
        return False
    try:
        r = subprocess.run([binary, "--version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


skip_no_ollama = pytest.mark.skipif(
    os.environ.get("HIEROCODE_TEST_OLLAMA") != "1" or not _ollama_alive(),
    reason="set HIEROCODE_TEST_OLLAMA=1 and ensure Ollama is running at localhost:11434",
)

skip_no_claude_cli = pytest.mark.skipif(
    os.environ.get("HIEROCODE_TEST_CLAUDE_CLI") != "1" or not _cli_alive("claude"),
    reason="set HIEROCODE_TEST_CLAUDE_CLI=1 and install/authenticate `claude`",
)

skip_no_codex_cli = pytest.mark.skipif(
    os.environ.get("HIEROCODE_TEST_CODEX_CLI") != "1" or not _cli_alive("codex"),
    reason="set HIEROCODE_TEST_CODEX_CLI=1 and install/authenticate `codex`",
)

skip_no_anthropic = pytest.mark.skipif(
    os.environ.get("HIEROCODE_TEST_ANTHROPIC") != "1" or not os.environ.get("ANTHROPIC_API_KEY"),
    reason="set HIEROCODE_TEST_ANTHROPIC=1 and ANTHROPIC_API_KEY",
)

skip_no_pipeline = pytest.mark.skipif(
    os.environ.get("HIEROCODE_TEST_PIPELINE") != "1" or not _ollama_alive(),
    reason=(
        "set HIEROCODE_TEST_PIPELINE=1; Ollama must be running and at least one "
        "planner (HIEROCODE_TEST_CLAUDE_CLI=1 or HIEROCODE_TEST_ANTHROPIC=1 + ANTHROPIC_API_KEY) "
        "must be available"
    ),
)
