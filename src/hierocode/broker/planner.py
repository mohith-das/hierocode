from hierocode.providers.base import BaseProvider
from hierocode.repo.prompts import build_plan_prompt

def generate_plan(task: str, provider: BaseProvider, model: str) -> str:
    """Generate a high level task breakdown."""
    prompt = build_plan_prompt(task)
    return provider.generate(prompt=prompt, model=model, temperature=0.7)
