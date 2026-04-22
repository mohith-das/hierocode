from hierocode.providers.base import BaseProvider
from hierocode.providers.ollama import OllamaProvider
from hierocode.providers.openai_compatible import OpenAICompatibleProvider
from hierocode.providers.lmstudio import LMStudioProvider
from hierocode.providers.transformers_local import TransformersLocalProvider
from hierocode.providers.anthropic import AnthropicProvider
from hierocode.providers.claude_code_cli import ClaudeCodeCliProvider
from hierocode.providers.codex_cli import CodexCliProvider
from hierocode.models.schemas import ProviderConfig

def get_provider(name: str, config: ProviderConfig) -> BaseProvider:
    """Factory to instantiate a provider based on its type constraint."""
    if config.type == "ollama":
        return OllamaProvider(name=name, config=config)
    elif config.type == "openai_compatible":
        return OpenAICompatibleProvider(name=name, config=config)
    elif config.type == "lmstudio":
        return LMStudioProvider(name=name, config=config)
    elif config.type == "transformers_local":
        return TransformersLocalProvider(name=name, config=config)
    elif config.type == "anthropic":
        return AnthropicProvider(name=name, config=config)
    elif config.type == "claude_code_cli":
        return ClaudeCodeCliProvider(name=name, config=config)
    elif config.type == "codex_cli":
        return CodexCliProvider(name=name, config=config)
    else:
        raise ValueError(f"Unknown provider type: {config.type}")
