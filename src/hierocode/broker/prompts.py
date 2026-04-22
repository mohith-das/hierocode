"""Prompt builders for the hierocode broker: planner, drafter, drafter-revision, and QA."""

from typing import Optional

from hierocode.broker.plan_schema import CapacityProfile, TaskUnit


def build_planner_system_prompt() -> str:
    """Return the system prompt that defines the planner agent's role and output contract."""
    return (
        "You are a senior software engineer acting as a task planner for a hierarchical coding"
        " pipeline.\n\n"
        "Your job is to decompose a user's coding task into a sequence of small, concrete"
        " TaskUnit work items that a local drafter model (3B–8B parameters) can execute one at"
        " a time. Each unit must have a single, focused goal that the drafter can accomplish"
        " with only the listed files in front of it.\n\n"
        "The drafter's capability envelope (max_input_tokens, max_output_tokens,"
        " max_files_per_unit) will be supplied in the user prompt. You MUST respect these"
        " limits when sizing each unit — never assign more files than max_files_per_unit, and"
        " keep est_input_tokens within max_input_tokens.\n\n"
        "OUTPUT CONTRACT: respond with a single valid JSON object that matches the Plan schema."
        " No prose before or after it. No markdown code fences. No comments. JSON only.\n\n"
        "The JSON must follow this exact structure:\n"
        '{\n'
        '  "task": "<restate the user task>",\n'
        '  "units": [\n'
        '    {\n'
        '      "id": "u1",\n'
        '      "goal": "<single concrete objective>",\n'
        '      "target_files": ["path/to/file.py"],\n'
        '      "context_files": ["path/to/reference.py"],\n'
        '      "acceptance": "<how to tell it worked>",\n'
        '      "est_input_tokens": 1200\n'
        '    }\n'
        '  ]\n'
        '}'
    )


def build_planner_user_prompt(
    task: str,
    skeleton: str,
    profile: CapacityProfile,
) -> str:
    """Return the user prompt for the planner with the task, repo skeleton, and drafter profile."""
    param_str = (
        f"{profile.param_count_b}B" if profile.param_count_b is not None else "unknown"
    )
    return f"""## User Task

{task}

## Repository Skeleton

{skeleton}

## Drafter Capability Profile

| Property            | Value                     |
|---------------------|---------------------------|
| tier                | {profile.tier}            |
| param_count_b       | {param_str}               |
| context_window      | {profile.context_window}  |
| max_input_tokens    | {profile.max_input_tokens}|
| max_output_tokens   | {profile.max_output_tokens}|
| max_files_per_unit  | {profile.max_files_per_unit}|

Design each TaskUnit so that:
- `est_input_tokens` ≤ {profile.max_input_tokens}
- `len(target_files) + len(context_files)` ≤ {profile.max_files_per_unit}
- The goal is achievable by a {param_str} local model with no external context.

## Required JSON Schema

Your response must match this structure exactly:

{{
  "task": "<restate the user task>",
  "units": [
    {{
      "id": "u1",
      "goal": "<single concrete objective>",
      "target_files": ["path/to/file.py"],
      "context_files": ["path/to/reference.py"],
      "acceptance": "<how to tell it worked>",
      "est_input_tokens": 1200
    }}
  ]
}}

Respond with the JSON object only. No markdown, no commentary."""


def build_drafter_prompt(
    unit: TaskUnit,
    packed_context: str,
) -> str:
    """Return the prompt sent to the drafter for a fresh code generation attempt."""
    primary_target = unit.target_files[0] if unit.target_files else "(no target file specified)"

    acceptance_section = ""
    if unit.acceptance:
        acceptance_section = f"\n## Acceptance Criteria\n\n{unit.acceptance}\n"

    return f"""## Goal

{unit.goal}
{acceptance_section}
## Primary Target

Primary target: {primary_target}

## Context Files

{packed_context}

## Instructions

Return ONLY the new full contents of the primary target file (first entry in target_files).\
 No explanations, no diff markers, no code fences."""


def build_drafter_revision_prompt(
    unit: TaskUnit,
    packed_context: str,
    prior_diff: str,
    feedback: str,
) -> str:
    """Return the prompt for a drafter revision attempt, including the prior diff and feedback."""
    primary_target = unit.target_files[0] if unit.target_files else "(no target file specified)"

    acceptance_section = ""
    if unit.acceptance:
        acceptance_section = f"\n## Acceptance Criteria\n\n{unit.acceptance}\n"

    return f"""## Goal

{unit.goal}
{acceptance_section}
## Primary Target

Primary target: {primary_target}

## Context Files

{packed_context}

## Previous Attempt (diff):

{prior_diff}

## Reviewer Feedback:

{feedback}

## Instructions

Address ALL of the reviewer's feedback. Return the full new file contents.\
 No explanations, no diff markers, no code fences."""


def build_qa_prompt(
    unit: TaskUnit,
    diff: str,
    test_output: Optional[str] = None,
    original_task: str = "",
) -> str:
    """Return the QA review prompt for the planner to evaluate a drafter's diff."""
    test_output_section = ""
    if test_output is not None:
        test_output_section = f"\n## Test Output\n\n{test_output}\n"

    acceptance_section = ""
    if unit.acceptance:
        acceptance_section = f"\n## Acceptance Criteria\n\n{unit.acceptance}\n"

    original_task_section = ""
    if original_task:
        original_task_section = f"\n## Original Task\n\n{original_task}\n"

    return f"""You are a strict code reviewer. Evaluate the drafter's diff below against the unit \
goal and acceptance criteria. Return a single JSON verdict — no prose, no fences.
{original_task_section}
## Unit Goal

{unit.goal}
{acceptance_section}
## Diff

{diff}
{test_output_section}
## Verdict Schema

{{
  "action": "accept" | "revise" | "split" | "escalate",
  "feedback": "<required if revise>",
  "sub_units": [...TaskUnit objects...],
  "reason": "<optional, explains escalate>"
}}

## Action Rules

- **accept**: the diff correctly satisfies the acceptance criteria with no correctness issues.
- **revise**: minor fixable issues exist; provide specific, actionable feedback in `feedback`.
- **split**: the unit was too large for the drafter; provide 2+ smaller `sub_units` covering the \
same goal.
- **escalate**: the task is beyond the drafter's capability; a stronger model must take over; \
explain in `reason`.

Respond with the JSON object only."""
