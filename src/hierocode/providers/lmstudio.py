from hierocode.broker.usage import UsageInfo
from hierocode.providers.openai_compatible import OpenAICompatibleProvider


class LMStudioProvider(OpenAICompatibleProvider):
    """
    Subclass of the generic OpenAI API adapter explicitly tracking LM Studio heuristics.
    Configured similarly, often defaults to localhost:1234/v1.
    """

    def __init__(self, name: str, config, **kwargs):
        if not config.base_url:
            config.base_url = "http://localhost:1234/v1"
        super().__init__(name, config, **kwargs)

    def generate(self, prompt: str, model: str, **options) -> str:
        result = super().generate(prompt=prompt, model=model, **options)
        # Fix provider_type to reflect lmstudio rather than the parent's openai_compatible.
        if self.last_usage is not None:
            self.last_usage = UsageInfo(
                input_tokens=self.last_usage.input_tokens,
                output_tokens=self.last_usage.output_tokens,
                cache_creation_input_tokens=self.last_usage.cache_creation_input_tokens,
                cache_read_input_tokens=self.last_usage.cache_read_input_tokens,
                messages=self.last_usage.messages,
                provider_type="lmstudio",
                model=self.last_usage.model,
            )
        return result
