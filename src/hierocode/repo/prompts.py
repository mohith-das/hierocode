def build_draft_prompt(task: str, filename: str, content: str) -> str:
    return f"""You are a drafting helper model.
Your task is to propose modifications for the file "{filename}" to satisfy the goal: {task}.

Current File Content:
```
{content}
```

Return ONLY the proposed new file content. No extra explanations.
"""

def build_plan_prompt(task: str) -> str:
    return f"""You are a planning model.
Break down the following task into a structured plan:
TASK: {task}

Return a markdown list of required changes.
"""

def build_review_prompt(task: str, filename: str, content: str) -> str:
    return f"""You are a code review model.
Task: {task}
File: {filename}

Content:
```
{content}
```

Provide a strict code review pointing out any correctness issues related to the task.
"""
