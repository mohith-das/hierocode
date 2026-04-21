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
