import httpx
from typing import List
from hierocode.providers.base import BaseProvider
from hierocode.exceptions import ProviderConnectionError, ModelNotFoundError
from hierocode.auth.helpers import resolve_auth_token

class OpenAICompatibleProvider(BaseProvider):
    """Provides access to OpenAI-compatible generic endpoints."""

    def __init__(self, name: str, config, **kwargs):
        super().__init__(name, config)
        self.base_url = self.config.base_url
        if not self.base_url:
            raise ValueError(f"OpenAI compatible provider '{name}' requires a 'base_url'")
        
        # Ensure trailing slash missing for path joins
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]
            
        self.auth_token = resolve_auth_token(self.config.auth)
        
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
            
        self.client = httpx.Client(headers=headers)

    def healthcheck(self) -> bool:
        try:
            r = self.client.get(f"{self.base_url}/models")
            return r.status_code == 200
        except httpx.RequestError:
            return False

    def list_models(self) -> List[str]:
        try:
            r = self.client.get(f"{self.base_url}/models")
            r.raise_for_status()
            data = r.json()
            return [m["id"] for m in data.get("data", [])]
        except httpx.RequestError as e:
            raise ProviderConnectionError(f"Failed to reach provider {self.name} at {self.base_url}: {e}")
        except httpx.HTTPStatusError as e:
            raise ProviderConnectionError(f"Provider {self.name} returned an error status: {e.response.status_code}")

    def generate(self, prompt: str, model: str, **options) -> str:
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                **options
            }
                
            r = self.client.post(f"{self.base_url}/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return ""
        except httpx.RequestError as e:
            raise ProviderConnectionError(f"Failed to reach provider {self.name}: {e}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ModelNotFoundError(f"Model {model} not found on provider {self.name}.")
            raise ProviderConnectionError(f"Provider {self.name} returned HTTP error: {e.response.status_code}")
