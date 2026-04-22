import httpx
from typing import List
from hierocode.broker.usage import UsageInfo
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
        self.last_usage = None
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
            data = r.json()
            self.last_usage = UsageInfo(
                input_tokens=int(data.get("prompt_eval_count", 0) or 0),
                output_tokens=int(data.get("eval_count", 0) or 0),
                messages=0,
                provider_type="ollama",
                model=model,
            )
            return data.get("response", "")
        except httpx.RequestError as e:
            raise ProviderConnectionError(f"Failed to reach Ollama at {self.base_url}: {e}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ModelNotFoundError(f"Model {model} not found on Ollama instance {self.name}.")
            raise ProviderConnectionError(f"Ollama returned HTTP error: {e.response.status_code}")

    def get_model_info(self, model: str) -> dict:
        """Query /api/show for model metadata. Returns num_ctx, param_count_b, quantization."""
        try:
            r = self.client.post(f"{self.base_url}/api/show", json={"name": model})
            r.raise_for_status()
            data = r.json()
        except httpx.RequestError as e:
            raise ProviderConnectionError(f"Failed to reach Ollama at {self.base_url}: {e}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ModelNotFoundError(f"Model {model} not found on Ollama instance {self.name}.")
            raise ProviderConnectionError(f"Ollama returned HTTP error: {e.response.status_code}")

        details = data.get("details", {}) or {}
        model_info = data.get("model_info", {}) or {}
        num_ctx = None
        for key, val in model_info.items():
            if key.endswith(".context_length") and isinstance(val, int):
                num_ctx = val
                break
        if num_ctx is None:
            num_ctx = data.get("parameters_num_ctx") or 8192

        param_count_b = None
        param_size = details.get("parameter_size") or ""
        if isinstance(param_size, str) and param_size:
            import re
            m = re.match(r"([\d.]+)\s*([BMK]?)", param_size.strip().upper())
            if m:
                try:
                    value = float(m.group(1))
                    unit = m.group(2)
                    if unit == "B" or unit == "":
                        param_count_b = value
                    elif unit == "M":
                        param_count_b = value / 1000.0
                except ValueError:
                    pass

        quantization = details.get("quantization_level") or None
        return {"num_ctx": int(num_ctx), "param_count_b": param_count_b, "quantization": quantization}
