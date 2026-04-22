import json
import subprocess
from typing import List

from hierocode.broker.usage import UsageInfo
from hierocode.exceptions import ProviderConnectionError
from hierocode.providers._models import ANTHROPIC_MODELS
from hierocode.providers.base import BaseProvider

_DEFAULT_ACTIVE_TOOLS = ["Read", "Grep", "Glob"]


class ClaudeCodeCliProvider(BaseProvider):
    """Planner provider that delegates to the user's locally-authenticated `claude` binary."""

    def healthcheck(self) -> bool:
        """Return True if the `claude` binary is on PATH and responds to --version."""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def list_models(self) -> List[str]:
        """Return hardcoded model list; the CLI has no list endpoint."""
        return list(ANTHROPIC_MODELS)

    def generate(self, prompt: str, model: str, **options) -> str:
        """Invoke `claude -p <prompt> --output-format json` and return the text result."""
        timeout = options.get("timeout", 180)
        cwd = options.get("cwd")
        system = options.get("system")

        cmd = ["claude", "-p", prompt, "--output-format", "json"]

        if system:
            cmd += ["--append-system-prompt", system]

        if model:
            cmd += ["--model", model]

        if options.get("exploration") == "active":
            tools = options.get("allowed_tools") or _DEFAULT_ACTIVE_TOOLS
            cmd += ["--allowedTools", ",".join(tools)]
            cmd += ["--disallowedTools", "Write,Edit,Bash"]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
        except FileNotFoundError:
            raise ProviderConnectionError(
                "claude CLI not found on PATH; install Claude Code and run `claude /login`"
            )
        except subprocess.TimeoutExpired:
            raise ProviderConnectionError("claude CLI timed out")

        if result.returncode != 0:
            raise ProviderConnectionError(f"claude CLI failed: {result.stderr[:500]}")

        raw = result.stdout
        self.last_usage = None
        try:
            parsed = json.loads(raw)
            usage_block = parsed.get("usage", {}) or {}
            self.last_usage = UsageInfo(
                input_tokens=usage_block.get("input_tokens", 0) or 0,
                output_tokens=usage_block.get("output_tokens", 0) or 0,
                cache_creation_input_tokens=usage_block.get("cache_creation_input_tokens", 0) or 0,
                cache_read_input_tokens=usage_block.get("cache_read_input_tokens", 0) or 0,
                messages=1,
                provider_type="claude_code_cli",
                model=model or "",
            )
            if "result" in parsed:
                return parsed["result"]
            if "content" in parsed:
                return parsed["content"]
            return json.dumps(parsed)
        except (json.JSONDecodeError, TypeError):
            return raw.rstrip()

    def is_local(self) -> bool:
        """Return True — subscription covers cost, treated as a free/local resource for routing."""
        return True
