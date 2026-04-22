from abc import ABC, abstractmethod
from typing import List, Optional
from hierocode.models.schemas import ProviderConfig
from hierocode.broker.usage import UsageInfo

class BaseProvider(ABC):
    """Abstract base provider for all LLM backends."""

    def __init__(self, name: str, config: ProviderConfig):
        self.name = name
        self.config = config
        self.last_usage: Optional[UsageInfo] = None

    @abstractmethod
    def healthcheck(self) -> bool:
        """Verify the provider is reachable and responsive."""
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """Return a list of available model names."""
        pass

    @abstractmethod
    def generate(self, prompt: str, model: str, **options) -> str:
        """Generate a response synchronously."""
        pass

    def is_local(self) -> bool:
        """Heuristic to determine if the provider is hosted locally."""
        if not self.config.base_url:
            return True
        return "127.0.0.1" in self.config.base_url or "localhost" in self.config.base_url
