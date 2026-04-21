# Hierocode

**Hierarchical coding with cost-aware model delegation.**

*Make local models actually useful for coding.*

Hierocode is a local-first hierarchical coding orchestrator that reduces expensive LLM token usage by delegating narrow coding tasks to cheaper local or self-hosted models, then escalating only when needed. It is a broker/orchestrator that helps you structure context and prompts safely without requiring remote high-cost APIs for every tiny change.

## Key Idea

Use cheap local models for most of the work; use stronger models only when needed. Hierocode is **not** an autonomous coding agent; rather, it is a tool that slices your codebase into manageable work units, lets local models draft candidate patches, and optionally uses a larger model for review and ranking.

## Installation

```bash
pip install hierocode
```
Or, for development:
```bash
git clone https://github.com/example/hierocode.git
cd hierocode
pip install -e .[dev]
```

## Quickstart

Initialize configuration:
```bash
hierocode init
```

Check your configuration and model visibility:
```bash
hierocode doctor
```

List discovered models:
```bash
hierocode models list
```

## Configuration

A configuration file is generated at `~/.hierocode.yaml`.

Example:

```yaml
default_provider: local_default
default_model: llama3

providers:
  local_default:
    type: ollama
    base_url: http://localhost:11434
    auth:
      type: none

  remote_backend:
    type: openai_compatible
    base_url: https://<your-domain-or-endpoint>/v1
    auth:
      type: bearer_env
      env_var: HIEROCODE_API_KEY
```

## Commands

- `hierocode init` - Initialize config.
- `hierocode doctor` - Verify backend connectivity.
- `hierocode providers list` - List configured providers.
- `hierocode models list` - Query backend for available models.
- `hierocode resources` - Show system resources and backend types.
- `hierocode workers suggest` - View recommended parallel worker counts.
- `hierocode plan --task "..."` - Generate a coding plan.
- `hierocode draft --task "..." --file path/to/file` - Generate a draft candidate patch using safe local modeling.
- `hierocode review --task "..." --file path/to/file` - Review a given file.

## Supported Providers

- `ollama`: Standard Ollama HTTP endpoints.
- `openai_compatible`: Standard OpenAI-style API endpoints (e.g. OpenAI, groq, together, etc.).
- `lmstudio`: A streamlined wrapper for LM Studio HTTP servers.
- `transformers_local`: (Planned) Run huggingface transformers strictly locally without server.

## Current Limitations

- **Not an agent**: It does not recursively apply patches or run external scripts to resolve dependencies.
- Patch application logic in v0.1 only outputs diffs to the console rather than silently appending/writing to files.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features and architecture progression.

## License

MIT
