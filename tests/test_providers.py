from unittest.mock import patch

import httpx

from hierocode.models.schemas import ProviderConfig
from hierocode.providers.ollama import OllamaProvider


def test_provider_initialization():
    conf = ProviderConfig(type="ollama", base_url="http://test")
    p = OllamaProvider("test", conf)
    assert p.base_url == "http://test"


def test_local_heuristic():
    conf_local = ProviderConfig(type="ollama", base_url="http://localhost:1234")
    p1 = OllamaProvider("test", conf_local)
    assert p1.is_local()

    conf_remote = ProviderConfig(type="openai_compatible", base_url="https://api.openai.com")
    p2 = OllamaProvider("test", conf_remote)  # any provider class will do
    assert not p2.is_local()


def test_ollama_generate_uses_long_timeout():
    """Regression: default httpx timeout (5s) is too short for local model generation.
    The provider must pass a generous per-request timeout on /api/generate."""
    conf = ProviderConfig(type="ollama", base_url="http://localhost:11434")
    p = OllamaProvider("test", conf)

    captured = {}

    def _fake_post(url, **kwargs):
        captured["url"] = url
        captured["timeout"] = kwargs.get("timeout")
        resp = httpx.Response(200, json={"response": "ok", "prompt_eval_count": 1, "eval_count": 1})
        resp._request = httpx.Request("POST", url)
        return resp

    with patch.object(p.client, "post", side_effect=_fake_post):
        p.generate("hi", "llama3.2:3b")

    assert captured["url"].endswith("/api/generate")
    # Must be at least a minute — realistic cold-load + generation budget.
    assert captured["timeout"] is not None
    assert float(captured["timeout"]) >= 60.0
