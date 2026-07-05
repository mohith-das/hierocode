# Roadmap

## Shipped

### v0.1
* MVP of provider abstractions (Ollama, OpenAI Compatible).
* Configuration using YAML.
* Heuristic based recommendations of resource boundaries.
* Core CLI for drafting single files and presenting a diff.

### v0.2
* Hierarchical pipeline: planner → drafter → QA with JSON Plan/TaskUnit schemas.
* Subscription-mode planners (`claude_code_cli`, `codex_cli`) and the Anthropic API provider with prompt caching.
* Capacity profiler, context budgeter, plan cache, cost estimator, first-run wizard.

### v0.3
* Persistent TUI via `prompt_toolkit` with 21+ slash commands.
* `/apply` patcher, named task aliases, usage tracking, live progress panel.
* Permission dialogs, opt-in active exploration for planner/reviewer, quota tracking.

### v0.4
* **MCP server** (`hierocode mcp`): any MCP-capable coding agent — Claude Code, Codex CLI, opencode, Cursor — can delegate bounded single-file drafts to the local model. Read-only by design; the host agent reviews and applies diffs.
* Headless draft engine and `hierocode draft --goal/--target` CLI.
* SEARCH/REPLACE edit blocks for the drafter (no more whole-file regeneration on existing files).
* Typed provider options layer and eight pipeline correctness fixes (see release notes).

## Planned

### v0.5
* Optional acceptance-criteria execution (run tests inside the pipeline) behind a sandbox boundary.
* Smarter drafter routing: pick between multiple local models by task size.

### Later
* Parallel unit dispatch.
* Streaming draft output in the TUI.
* Provider auto-discovery.
