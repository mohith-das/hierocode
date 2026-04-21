import pytest
from hierocode.providers.ollama import OllamaProvider
from hierocode.models.schemas import ProviderConfig, AuthConfig
import httpx

def test_provider_initialization():
    conf = ProviderConfig(type="ollama", base_url="http://test")
    p = OllamaProvider("test", conf)
    assert p.base_url == "http://test"
    
def test_local_heuristic():
    conf_local = ProviderConfig(type="ollama", base_url="http://localhost:1234")
    p1 = OllamaProvider("test", conf_local)
    assert p1.is_local()
    
    conf_remote = ProviderConfig(type="openai_compatible", base_url="https://api.openai.com")
    p2 = OllamaProvider("test", conf_remote) # any provider class will do
    assert not p2.is_local()
