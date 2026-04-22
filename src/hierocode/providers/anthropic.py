import os
from typing import List, Optional

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

from hierocode.providers._models import ANTHROPIC_MODELS
from hierocode.providers.base import BaseProvider
from hierocode.exceptions import ProviderConnectionError, ModelNotFoundError

_HEALTHCHECK_MODEL = "claude-haiku-4-5"
_SDK_ERROR_MSG = "anthropic package not installed; pip install anthropic>=0.39.0"
_CACHE_MIN_CHARS = 1024


class AnthropicProvider(BaseProvider):
    """Planner provider backed by the Anthropic Messages API."""

    def __init__(self, name: str, config, **kwargs):
        super().__init__(name, config)

    def _get_client(self):
        if Anthropic is None:
            raise ProviderConnectionError(_SDK_ERROR_MSG)
        env_var = (self.config.auth.env_var or "ANTHROPIC_API_KEY")
        api_key = os.environ.get(env_var)
        if not api_key:
            raise ProviderConnectionError(f"{env_var} not set")
        return Anthropic(api_key=api_key)

    def _build_create_kwargs(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        system_text: Optional[str],
        cache: bool,
        cache_user_prefix: Optional[str],
    ) -> dict:
        """Build the kwargs dict for client.messages.create with optional prompt caching."""
        create_kwargs: dict = {"model": model, "max_tokens": max_tokens}

        # System block
        if system_text:
            if cache and len(system_text) >= _CACHE_MIN_CHARS:
                create_kwargs["system"] = [
                    {"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}
                ]
            else:
                create_kwargs["system"] = system_text

        # Messages block
        if cache and cache_user_prefix is not None and len(cache_user_prefix) >= _CACHE_MIN_CHARS:
            create_kwargs["messages"] = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": cache_user_prefix,
                            "cache_control": {"type": "ephemeral"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            create_kwargs["messages"] = [{"role": "user", "content": prompt}]

        return create_kwargs

    def healthcheck(self) -> bool:
        try:
            client = self._get_client()
            client.messages.create(
                model=_HEALTHCHECK_MODEL,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except ProviderConnectionError:
            raise
        except Exception:
            return False

    def list_models(self) -> List[str]:
        if Anthropic is None:
            raise ProviderConnectionError(_SDK_ERROR_MSG)
        return list(ANTHROPIC_MODELS)

    def generate(
        self,
        prompt: str,
        model: str,
        *,
        cache: bool = True,
        cache_user_prefix: Optional[str] = None,
        **options,
    ) -> str:
        """Generate a response; supports prompt caching via cache_control blocks."""
        client = self._get_client()
        max_tokens = options.get("max_tokens", 4096)
        system_parts = []

        system_kwarg = options.get("system")
        if system_kwarg:
            system_parts.append(system_kwarg)

        if options.get("json_mode"):
            system_parts.append("Respond with valid JSON only. No prose, no code fences.")

        system_text = "\n\n".join(system_parts) if system_parts else None

        create_kwargs = self._build_create_kwargs(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            system_text=system_text,
            cache=cache,
            cache_user_prefix=cache_user_prefix,
        )

        try:
            response = client.messages.create(**create_kwargs)
            return response.content[0].text
        except Exception as e:
            err = str(e)
            if "model" in err.lower() and ("not found" in err.lower() or "invalid" in err.lower()):
                raise ModelNotFoundError(f"Model '{model}' not found on Anthropic: {e}")
            raise ProviderConnectionError(f"Anthropic API error: {e}")

    def is_local(self) -> bool:
        return False
