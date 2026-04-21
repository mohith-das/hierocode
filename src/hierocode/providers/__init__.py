from hierocode.providers.base import BaseProvider
from hierocode.providers.ollama import OllamaProvider
from hierocode.providers.openai_compatible import OpenAICompatibleProvider
from hierocode.providers.lmstudio import LMStudioProvider
from hierocode.providers.transformers_local import TransformersLocalProvider
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
    else:
        raise ValueError(f"Unknown provider type: {config.type}")
