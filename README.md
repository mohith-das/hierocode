# Hierocode

**Hierarchical coding with cost-aware model delegation.**

Frontier models plan and review; local models write the code. A $17/mo Claude Pro
subscription paired with a laptop running Ollama gives you a full agentic coding
workflow at $0 incremental cost per task.

## Why hierocode

Frontier-model APIs charge per token, and code generation is token-heavy. A single
non-trivial refactor can consume thousands of output tokens across multiple retries,
driving API costs into the dollars-per-task range quickly. Subscription plans (Claude
Pro, ChatGPT Plus) are cheaper on a per-message basis but have soft quotas — burning
them on code generation leaves nothing for the rest of your workday.

Hierocode splits the work by role. The planner (your Claude Pro or ChatGPT Plus
subscription, or an API key) turns your task into a structured JSON work breakdown
and QAs the result. The drafter (a local Ollama model) does the actual code writing
for each bounded unit. The reviewer defaults to the planner tier. Planner calls are
short — 5–20 k input tokens, 1–3 k output — so two or three of them barely dent
a subscription quota. Drafter calls run entirely on your own hardware at $0.

## Before you install

Hierocode needs three things on your machine. `pip install hierocode` is the last
step, not the first.

### 1. Python ≥ 3.10

```bash
python3 --version
```

If older: `brew install python@3.12` (macOS), `sudo apt install python3.12`
(Ubuntu/Debian), or https://www.python.org/downloads/ (Windows).

### 2. A local drafter — Ollama + at least one model

This is where the actual code gets written. Runs entirely on your hardware, $0 per call.

```bash
# Install Ollama (one-time):
#   macOS / Windows:  https://ollama.com/download   (GUI installer)
#   Linux:            curl -fsSL https://ollama.com/install.sh | sh

# Pull a drafter model (~2 GB for llama3.2:3b)
ollama pull llama3.2:3b
```

Model size the wizard picks for you based on RAM:

| RAM          | Recommended drafter         |
|--------------|-----------------------------|
| ≤ 12 GB      | `llama3.2:1b`               |
| 12–24 GB     | `llama3.2:3b`               |
| 24–48 GB     | `qwen2.5-coder:7b`          |
| ≥ 48 GB      | `qwen2.5-coder:14b`         |

### 3. A planner — pick ONE

The planner breaks your task into sized units and QAs the drafter's output. You need
exactly one of these paths configured:

| Path                   | Cost per task       | Setup                                                                                          |
|------------------------|---------------------|------------------------------------------------------------------------------------------------|
| **Claude Pro** ($17/mo) | $0 incremental     | Install [Claude Code](https://docs.anthropic.com/claude/docs/claude-code), run `claude /login` |
| **ChatGPT Plus** ($20/mo) | $0 incremental   | Install Codex CLI, run `codex login`                                                           |
| **Anthropic API key**  | ~$0.01–0.15/task    | `export ANTHROPIC_API_KEY=sk-...`                                                              |

**No subscription or API key?** Hierocode falls back to using Ollama as the planner
too. It works, but plan quality drops noticeably on non-trivial tasks. The wizard
warns when this happens. Get a subscription or API key for any real use.

## Quickstart — the $17 journey

Assumes the three prereqs above are satisfied and `claude /login` has been run.

```bash
pip install hierocode
```

### First run

```bash
# Detect your environment and write a tailored config.
hierocode init --wizard

# Confirm roles and providers were detected correctly.
hierocode models show

# Sanity-check that providers are reachable.
hierocode doctor

# Estimate cost and message quota before committing.
hierocode run -t "add input validation to the login endpoint" --estimate

# Run the full plan → draft → QA pipeline.
hierocode run -t "add input validation to the login endpoint"
```

The wizard detects `claude` on PATH, picks it as the planner, reads your RAM to
choose the right Ollama drafter model, and writes `~/.hierocode.yaml`. Nothing is
sent to an API until you run `hierocode run` (or `hierocode plan`).

## Quickstart — the API journey

If you have `ANTHROPIC_API_KEY` set and no `claude` or `codex` CLI installed, the
wizard still works: `hierocode init --wizard` detects the env var and routes the
planner through the Anthropic Messages API using `claude-haiku-4-5` by default.
Run `hierocode models set planner claude-sonnet-4-6` afterward if you want a stronger
planner. All other steps are identical to the $17 journey above.

## Architecture

```
hierocode run
    |
    +-- skeleton.py       AST symbol index of the repo (Python) / file list (other)
    |
    +-- capacity.py       Probe drafter model: num_ctx, RAM, GPU, tier
    |
    +-- Planner           (claude_code_cli / codex_cli / anthropic / ollama)
    |       Receives: task + skeleton + capacity profile
    |       Returns:  JSON Plan — list of TaskUnits
    |       Cache:    plan_cache keyed on (task, skeleton, planner_model, drafter_model)
    |
    +-- Dispatcher        Iterates TaskUnits
            |
            +-- Budget    Pack target + context files into drafter context window
            |
            +-- Drafter   (Ollama) — writes code for one bounded unit
            |       Returns: unified diff
            |
            +-- QA        (Planner tier) — reviews diff against acceptance criteria
                    verdict:
                      accept   -> emit diff to stdout
                      revise   -> loop back to Drafter (cap: max_revisions_per_unit)
                      split    -> re-plan the unit into sub-units
                      escalate -> planner drafts the unit directly (cap: max_escalations_per_task)
```

Diffs are printed to stdout; they are not applied automatically.

## Supported providers

| Type | Auth required | Typical role |
|---|---|---|
| `claude_code_cli` | `claude` binary, logged in to Pro | Planner, Reviewer |
| `codex_cli` | `codex` binary, logged in to Plus | Planner, Reviewer |
| `anthropic` | `ANTHROPIC_API_KEY` env var | Planner, Reviewer |
| `ollama` | none | Drafter (and fallback Planner) |
| `openai_compatible` | bearer token env var | Planner or Drafter |
| `lmstudio` | none | Drafter |
| `transformers_local` | none | Drafter |

Model IDs in each provider:

- `claude_code_cli` / `anthropic`: `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5`
- `codex_cli`: `gpt-5`, `gpt-5-codex`, `o4-mini`
- `ollama`: whatever you have pulled; wizard recommends based on RAM

## Cost model

The drafter is always local ($0). Only planner calls have a cost.

| Planner path | Cost per task | Notes |
|---|---|---|
| `claude_code_cli` (Pro, $17/mo) | $0 incremental | Counts against subscription quota |
| `codex_cli` (Plus, $20/mo) | $0 incremental | Counts against subscription quota |
| `anthropic` Haiku API | ~$0.01–0.03 | 4 calls x ~10 k input + ~2 k output |
| `anthropic` Sonnet API | ~$0.05–0.15 | 4 calls x ~10 k input + ~2 k output |

The 4-call estimate breaks down as:

- Plan cache **hit**: 0 planner calls.
- Plan cache **miss** + typical 3 QA rounds: 1 planning call + 3 QA calls = 4 calls.

Token counts use a char/4 heuristic (not tiktoken) with a 20% safety margin.

## Commands

### Setup

```bash
hierocode init                  # Write default ~/.hierocode.yaml
hierocode init --wizard         # Auto-detect environment, write tailored config
hierocode init --force          # Overwrite existing config
hierocode doctor                # Validate config and ping all providers
```

### Inspection

```bash
hierocode providers list        # List configured providers and their types
hierocode models list           # Query each provider for available model names
hierocode models show           # Show planner / drafter / reviewer role bindings
hierocode resources             # Print CPU count and RAM (total / available)
hierocode workers suggest       # Suggest safe / balanced / aggressive worker counts
```

### Execution

```bash
hierocode plan -t "..."                      # Generate and print a JSON plan (no drafting)
hierocode plan -t "..." --planner-model <m>  # Override planner model
hierocode run -t "..."                       # Full pipeline: plan -> draft -> QA
hierocode run -t "..." --estimate            # Print cost/quota estimate, then exit
hierocode run -t "..." --drafter-model <m>   # Override drafter model
hierocode run -t "..." --repo ./path         # Set repo root for skeleton builder
hierocode draft -t "..." --file path/to/file # Draft a patch for a single file
hierocode review -t "..." --file path/to/file # Review a single file
```

### Maintenance

```bash
hierocode models set planner claude-sonnet-4-6          # Set planner model
hierocode models set drafter llama3.2:3b                # Set drafter model
hierocode models set planner claude-haiku-4-5 -p claude_pro  # With explicit provider
hierocode cache clear                                   # Delete all cached plans
hierocode version                                       # Print installed version
```

## Configuration reference

`hierocode init --wizard` writes `~/.hierocode.yaml`. A representative example for
a Claude Pro user with 16 GB RAM:

```yaml
# Generated by hierocode init --wizard
default_provider: local_ollama

providers:
  local_ollama:
    type: ollama
    base_url: http://localhost:11434
    auth:
      type: none
  claude_pro:
    type: claude_code_cli
    auth:
      type: none
  codex_plus:
    type: codex_cli
    auth:
      type: none
  anthropic_api:
    type: anthropic
    auth:
      type: bearer_env
      env_var: ANTHROPIC_API_KEY

routing:
  planner:
    provider: claude_pro
    model: claude-sonnet-4-6
  drafter:
    provider: local_ollama
    model: llama3.2:3b
  reviewer:
    provider: claude_pro
    model: claude-sonnet-4-6

policy:
  max_revisions_per_unit: 2
  max_escalations_per_task: 3
  warn_before_escalation: true

parallelization:
  default_strategy: balanced
  max_local_workers: 4
  max_remote_workers: 8
```

`hierocode init` (no wizard) writes a minimal config that points every role at
`local_ollama` with `llama3.2:3b`. That works out of the box for pure-local setups
but runs in degraded mode (Ollama-as-planner). Use `hierocode init --wizard` for
any real setup so the three prereqs above are actually wired up.

v0.1 YAML files still parse — legacy keys like `default_model`, `small_model`,
`routing.draft_model`, and `routing.review_model` are silently ignored. Run
`hierocode init --wizard --force` once to upgrade cleanly.

## Current limitations

- **Not an autonomous agent.** Hierocode does not apply patches, run tests, or iterate
  on failures without user intervention. Every diff must be reviewed and applied manually.
- **Diffs are printed, not applied.** The `run` and `draft` commands emit unified diffs
  to stdout. There is no `--apply` flag in v0.2.
- **AST skeleton is Python-only.** Non-Python files are listed by name and size; their
  symbols are not extracted. The planner sees less context for non-Python repos.
- **Token counting uses a char/4 heuristic.** The budget module does not use tiktoken
  for Ollama-side estimation; a 20% safety margin compensates but is not exact.
- **Single-file `draft` and `review` commands do not use the QA loop.** They call the
  provider once and return raw output. Use `hierocode run` for the full pipeline.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, linting, and test
instructions.

## Acknowledgements

Built with the help of Claude Opus 4.7 (Anthropic).

## License

MIT
