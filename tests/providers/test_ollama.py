from unittest.mock import MagicMock
from hierocode.models.schemas import ProviderConfig
from hierocode.providers.ollama import OllamaProvider

def test_ollama_generate_payload_options(monkeypatch):
    conf = ProviderConfig.model_construct(type="ollama", base_url="http://localhost:11434")
    provider = OllamaProvider("test", conf)
    
    mock_post = MagicMock()
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"response": "test_response"}
    monkeypatch.setattr(provider.client, "post", mock_post)

    provider.generate(
        "test_prompt",
        "test_model",
        max_tokens=500,
        system="test_system",
        json_mode=True,
        temperature=0.8,
        timeout=123.4
    )
    
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["timeout"] == 123.4
    payload = kwargs["json"]
    assert payload["model"] == "test_model"
    assert payload["prompt"] == "test_prompt"
    assert payload["system"] == "test_system"
    assert payload["format"] == "json"
    assert "options" in payload
    assert payload["options"]["num_predict"] == 500
    assert payload["options"]["temperature"] == 0.8
