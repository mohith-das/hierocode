from typing import List
from hierocode.broker.usage import UsageInfo
from hierocode.providers.base import BaseProvider


class TransformersLocalProvider(BaseProvider):
    """
    Planned local provider to run HuggingFace models strictly in process.
    Not currently shipped in the v0.1 artifact.
    """

    def __init__(self, name: str, config, **kwargs):
        super().__init__(name, config)
        raise NotImplementedError("HuggingFace local inference is planned but not yet implemented.")

    def healthcheck(self) -> bool:
        return False

    def list_models(self) -> List[str]:
        return []

    def generate(self, prompt: str, model: str, **options) -> str:
        # Zero-token stub so the interface is uniform even though no real
        # inference happens here yet.
        self.last_usage = UsageInfo(
            input_tokens=0,
            output_tokens=0,
            messages=0,
            provider_type="transformers_local",
            model=model,
        )
        return ""
