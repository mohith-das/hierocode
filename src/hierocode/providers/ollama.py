import httpx
from typing import List
from hierocode.providers.base import BaseProvider
from hierocode.exceptions import ProviderConnectionError, ModelNotFoundError

class OllamaProvider(BaseProvider):
    """Provides access to Ollama via typical port endpoints."""

    def __init__(self, name: str, config, **kwargs):
        super().__init__(name, config)
        self.base_url = self.config.base_url or "http://localhost:11434"
        self.client = httpx.Client()

    def healthcheck(self) -> bool:
        try:
            r = self.client.get(f"{self.base_url}/")
            return r.status_code == 200
        except httpx.RequestError:
            return False

    def list_models(self) -> List[str]:
        try:
            r = self.client.get(f"{self.base_url}/api/tags")
            r.raise_for_status()
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
        except httpx.RequestError as e:
            raise ProviderConnectionError(f"Failed to reach Ollama at {self.base_url}: {e}")
        except httpx.HTTPStatusError as e:
            raise ProviderConnectionError(f"Ollama returned an error status: {e.response.status_code}")

    def generate(self, prompt: str, model: str, **options) -> str:
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            if options:
                payload["options"] = options
                
            r = self.client.post(f"{self.base_url}/api/generate", json=payload)
            r.raise_for_status()
            return r.json().get("response", "")
        except httpx.RequestError as e:
            raise ProviderConnectionError(f"Failed to reach Ollama at {self.base_url}: {e}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ModelNotFoundError(f"Model {model} not found on Ollama instance {self.name}.")
            raise ProviderConnectionError(f"Ollama returned HTTP error: {e.response.status_code}")
