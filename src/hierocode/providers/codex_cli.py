import json
import subprocess
from typing import List

from hierocode.broker.usage import UsageInfo
from hierocode.providers._models import CODEX_MODELS
from hierocode.providers.base import BaseProvider
from hierocode.exceptions import ProviderConnectionError


class CodexCliProvider(BaseProvider):
    """Shells out to the user's authenticated `codex` binary (OpenAI Codex CLI)."""

    def healthcheck(self) -> bool:
        """Return True if `codex --version` exits with returncode 0."""
        try:
            result = subprocess.run(
                ["codex", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def list_models(self) -> List[str]:
        """Return the hardcoded list of models accessible via Codex CLI."""
        return list(CODEX_MODELS)

    def generate(self, prompt: str, model: str, **options) -> str:
        """Run `codex exec <prompt>` and return the terminal agent message content.

        codex (>=0.122.0) has no `--system` flag and exec mode is non-interactive by
        default, so the system string (if provided) is prepended to the prompt text.
        We always pass `--skip-git-repo-check` so hierocode works outside a git repo.
        """
        system = options.get("system")
        effective_prompt = f"{system}\n\n{prompt}" if system else prompt

        cmd = ["codex", "exec", "--skip-git-repo-check", "--json"]

        # Pass --model only when the caller wants to override codex's own default.
        # ChatGPT-authenticated codex rejects most named models (gpt-5, gpt-5-codex,
        # o4-mini) — the account's tier determines what's available. Treat "default",
        # "" and None as "let codex decide".
        if model and model.lower() != "default":
            cmd += ["--model", model]

        if options.get("exploration") == "active":
            # Read-only sandbox lets the model use Read/Grep/Glob without letting
            # it edit files — hierocode's drafter owns edits.
            cmd += ["--sandbox", "read-only"]

        cmd.append(effective_prompt)

        timeout = options.get("timeout", 180)
        cwd = options.get("cwd")

        try:
            # stdin=DEVNULL — codex exec reads extra input from stdin if connected
            # to a TTY, which hangs the subprocess even when a prompt arg is given.
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            raise ProviderConnectionError(
                "codex CLI not found on PATH; install Codex CLI and run `codex login`"
            )
        except subprocess.TimeoutExpired:
            raise ProviderConnectionError("codex CLI timed out")

        if result.returncode != 0:
            raise ProviderConnectionError(f"codex CLI failed: {result.stderr[:500]}")

        self.last_usage = None
        input_tokens, output_tokens = self._extract_usage_from_jsonl(result.stdout)
        self.last_usage = UsageInfo(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            messages=1,
            provider_type="codex_cli",
            model=model or "",
        )
        return self._parse_jsonl(result.stdout)

    def is_local(self) -> bool:
        """Return True — usage is covered by the user's ChatGPT Plus subscription."""
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_usage_from_jsonl(stdout: str) -> tuple[int, int]:
        """Best-effort token extraction. Returns (input_tokens, output_tokens)."""
        for raw_line in reversed(stdout.splitlines()):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Try several known shapes
            if "usage" in obj and isinstance(obj["usage"], dict):
                u = obj["usage"]
                return (
                    int(u.get("input_tokens", u.get("prompt_tokens", 0)) or 0),
                    int(u.get("output_tokens", u.get("completion_tokens", 0)) or 0),
                )
            if "input_tokens" in obj and "output_tokens" in obj:
                return (int(obj["input_tokens"] or 0), int(obj["output_tokens"] or 0))
        return (0, 0)

    @staticmethod
    def _parse_jsonl(stdout: str) -> str:
        """Parse Codex JSONL output and return the terminal agent-message text.

        Codex 0.122.0 emits events of shape:
            {"type": "item.completed", "item": {"type": "agent_message", "text": "..."}}
        Older/alternate shapes may have `agent_message` or `message` at the top level
        with `content` / `text` / `message` as the payload — supported as fallbacks."""
        _flat_terminal_types = {"agent_message", "message"}
        terminal_content: str | None = None
        fallback_parts: list[str] = []

        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Shape 1: nested item.completed → item.agent_message (codex 0.122+)
            item = obj.get("item")
            if isinstance(item, dict) and item.get("type") in _flat_terminal_types:
                text = item.get("text") or item.get("content") or item.get("message")
                if text:
                    terminal_content = str(text)
                    continue

            # Shape 2: flat agent_message / message at top level
            event_type = obj.get("type", "")
            text = obj.get("content") or obj.get("text") or obj.get("message")

            if event_type in _flat_terminal_types and text:
                terminal_content = str(text)
                continue

            # Collect incidental text for fallback
            if text and event_type not in _flat_terminal_types:
                fallback_parts.append(str(text))

        if terminal_content is not None:
            return terminal_content

        if fallback_parts:
            return "\n".join(fallback_parts)

        return stdout
