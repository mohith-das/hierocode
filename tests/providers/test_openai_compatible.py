from unittest.mock import MagicMock
from hierocode.models.schemas import AuthConfig, ProviderConfig
from hierocode.providers.openai_compatible import OpenAICompatibleProvider

def test_openai_compatible_generate_payload_options(monkeypatch):
    conf = ProviderConfig.model_construct(type="openai_compatible", base_url="http://test_url", auth=AuthConfig(type="none"))
    provider = OpenAICompatibleProvider("test", conf)
    
    mock_post = MagicMock()
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"choices": [{"message": {"content": "test"}}]}
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
    assert payload["max_tokens"] == 500
    assert payload["temperature"] == 0.8
    assert "system" not in payload # should be in messages
    
    messages = payload["messages"]
    assert len(messages) == 3
    assert messages[0] == {"role": "system", "content": "test_system"}
    assert messages[1] == {"role": "system", "content": "Respond with valid JSON only. No prose, no code fences."}
    assert messages[2] == {"role": "user", "content": "test_prompt"}
