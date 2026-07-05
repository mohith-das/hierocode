# Hierocode — Living Roadmap

**Status as of 2026-04-22.** This is the authoritative plan. Wave-specific drafts
(e.g. `plan-v0.3.md`) are historical artifacts kept for reference.

## Vision

Open-source hierarchical coding orchestrator. Frontier models (Claude Opus/Sonnet,
GPT-5) plan and review; local Ollama models write the code. A developer with a
**$17/mo Claude Pro** or **$20/mo ChatGPT Plus** subscription and a laptop running
Ollama gets a full agentic coding workflow at **$0 incremental cost per task**.

Not a fully autonomous agent. A broker: bounded pipeline, explicit confirmations
for destructive actions, predictable cost envelope.

## Architecture in one paragraph

`hierocode run -t "..."` (or the TUI equivalent) builds a repo skeleton, profiles
the drafter's capacity, and asks the planner (Claude/Codex via CLI subscription or
Anthropic API) for a JSON Plan of TaskUnits sized to the drafter's context window.
The dispatcher iterates units through draft → QA → {accept | revise | split |
escalate}, using Ollama for the token-heavy drafting and the planner tier for
review. The plan cache (24h TTL) skips re-planning identical tasks. Config and
task aliases persist in `~/.hierocode.yaml`.

## Shipped waves

| Wave | Version | Headline | Tests |
|---|---|---|---|
| 1 | 0.2.0-internal | 4 new providers (`anthropic`, `claude_code_cli`, `codex_cli`, updates); JSON plan schema | 87 |
| 2+3 | 0.2.0-internal | Pipeline: capacity profiler, context budgeter, QA loop, dispatcher, prompts | 162 |
| 4 | 0.2.0-internal | AST skeleton, Anthropic prompt caching, first-run wizard, cost estimator, integration tests | 227 |
| 5 | 0.2.0-internal | GPU/VRAM probe, plan cache, `models set/show` + `cache clear` subcommands | 270 |
| Cleanup | 0.2.0 | Removed v0.1 dead code (ranking, workers, escalation stub, legacy prompts/context/slicer), legacy schema fields; README rewritten for v0.2 | 271 |
| v0.3 | 0.3.0 | Persistent TUI via `prompt_toolkit`, 21+ slash commands, `/apply` patcher, named task aliases | 384 |
| v0.3.1 + v0.3.2 (combined) | **0.3.2** | Usage tracking + `/usage`, live progress panel, permission dialogs with `warn_before_escalation`, opt-in active exploration for planner/reviewer, file-based pricing override + QuotaTracker with warning thresholds | 504 |
| v0.4 | 0.4.0 | MCP server + provider option layer + edit blocks + 8 correctness fixes | 597 |

## Current state snapshot (v0.3.0)

### Providers — 7 types

- `ollama` — local drafter; tier heuristic from param count; exposes `get_model_info` via `/api/show`.
- `anthropic` — Anthropic API with prompt caching on long system prompts.
- `claude_code_cli` — subscription-mode planner via `claude` binary. $0 incremental.
- `codex_cli` — subscription-mode planner via `codex` binary. $0 incremental.
- `openai_compatible`, `lmstudio`, `transformers_local` — extras for flexibility.

### Pipeline

- `broker/skeleton.py` — AST-based repo summary (Python symbols + file sizes for others).
- `broker/capacity.py` — profiles drafter: num_ctx, param count, tier, input/output budget.
- `broker/budget.py::pack_context` — packs TaskUnit files to fit drafter's context window.
- `broker/planner.py` — asks the planner for JSON Plan with validated TaskUnits.
- `broker/prompts.py` — planner/drafter/QA prompt templates.
- `broker/dispatcher.py` — the loop: draft → QA → {accept/revise/split/escalate} with hard caps.
- `broker/qa.py` — asks the planner to verdict a drafter's diff.
- `broker/plan_cache.py` — 24h file cache keyed on `(task, skeleton, planner_model, drafter_model)`.
- `broker/estimator.py` — pre-flight cost/quota prediction (projection, not actuals).
- `broker/patcher.py` — parses unified diffs, applies via `git apply` with per-file confirmation.
- `broker/aliases.py` — named task aliases persisted in YAML.

### CLI (one-shot)

```
hierocode init [--wizard] [--force]
hierocode doctor
hierocode resources
hierocode providers list
hierocode models list / show / set <role> <model> [--provider P]
hierocode workers suggest
hierocode plan -t "..." [--planner-model ...] [--drafter-model ...] [--repo P]
hierocode run -t "..." [--planner-model ...] [--drafter-model ...] [--repo P] [--estimate]
hierocode draft -t "..." -f <file> [--drafter-model ...]
hierocode review -t "..." -f <file> [--reviewer-model ...]
hierocode cache clear
hierocode version
```

### TUI (bare `hierocode` → REPL)

Slash commands registered at REPL start:

```
/run /plan /estimate /draft /review
/apply /diff
/models /models set /cache clear /config edit
/doctor /resources /repo /history
/task save|list|delete|<name>
/help /clear /exit /quit
```

Interaction modes (config: `tui.interaction_mode`):
- `prompt` (default, safer): plain text → "Run as task? [y/N/e(dit)]"
- `immediate`: plain text → `/run <text>` directly

### Config schema (`HierocodeConfig`)

- `default_provider: str`
- `providers: dict[str, ProviderConfig]`
- `parallelization: ParallelizationConfig`
- `routing: RoutingConfig` — per-role `RoleRouting(provider, model)` for planner/drafter/reviewer
- `policy: PolicyConfig` — `max_revisions_per_unit`, `max_escalations_per_task`, `warn_before_escalation`
- `tui: TUIConfig` — `interaction_mode`
- `tasks: list[TaskAlias]` — named task shortcuts

`extra="ignore"` on both top-level and routing — v0.1 YAMLs with `default_model`,
`small_model`, `routing.draft_model`, `routing.review_model` still parse (fields dropped).

## v0.3.1 — "Feel" wave (in flight)

Goal: hierocode feels closer to Claude Code without becoming Claude Code.

Phase A (sequential — starts now, touches too many shared files for parallel split):

- **W25 — Usage tracking foundation + `/usage` command.** Extract actual token/message usage from every provider's response (Ollama `/api/generate`, Anthropic SDK `response.usage`, Claude CLI `--output-format json` usage field, Codex JSONL events, OpenAI-compat standard). Accumulate into `SessionState` by role. `/usage` handler prints session totals.

Phase B (parallel after W25 lands):

- **W26 — Live progress panel during `/run`.** `rich.live.Live` rendering plan state (units queued/drafting/QAing/done), drafter model, elapsed time, live token counts, message/quota counter.
- **W27 — Permission dialogs.** Replace `/apply`'s plain `[y/N/s/q]` with a `prompt_toolkit` radiolist-style prompt. Wire up the currently-ignored `policy.warn_before_escalation` to actually pause and ask before escalating an expensive retry.
- **W28 — Active exploration mode (opt-in).** Add `routing.<role>.exploration: "passive" | "active"` config field (default `passive`). When active, pass `--allowedTools Read,Grep,Glob` (and similar for Codex) so the planner/reviewer uses its own tools to explore the repo. Wizard asks once during setup. Preserves the predictable-cheap default; unlocks better plan quality for users who opt in.

**Ships as:** v0.3.1 on PyPI.

## v0.3.2 — "Dynamic pricing + quota awareness" wave (planned)

Goal: make the cost-aware story robust to model releases and subscription-limit
changes without requiring a hierocode release, and surface quota pressure
inline in `/usage` so users don't burn through their window blind.

### W29 — Dynamic pricing (Option A — file-based override)

The hardcoded `ANTHROPIC_PRICING` table in `broker/usage.py` goes stale the
moment Anthropic changes prices or ships a new model. Option A lets users
override the table via a local config file.

New file (user-editable): `~/.hierocode/pricing.yaml`

```yaml
anthropic_models:
  # [input $/M tokens, output $/M tokens]
  claude-haiku-4-5: [0.25, 1.25]
  claude-sonnet-4-6: [3.0, 15.0]
  claude-opus-4-7: [15.0, 75.0]
  claude-some-future-model: [2.0, 10.0]   # users can add ahead of hierocode releases

openai_models:
  gpt-5: [5.0, 15.0]           # future — once we track OpenAI API costs

subscription_quotas:
  claude_pro:
    messages_per_window: 40
    window_hours: 5
  chatgpt_plus:
    messages_per_window: 50
    window_hours: 3
```

Behavior:
- On app startup, load `~/.hierocode/pricing.yaml` if present.
- Merge over hardcoded defaults — user values win, unknown-to-hierocode models
  are additive.
- Malformed file → log a warning, fall back to defaults (never crash startup).
- New helper `broker/pricing.py` owning the load + merge logic; `broker/usage.py`
  imports from there instead of hardcoding.

**Option C (remote pricing fetch) is shelved.** Only graduate if real users
report stale-pricing issues — adds a network dependency and privacy surface
that the file-based approach avoids.

### W30 — QuotaTracker

Ships alongside W29. Consumes `subscription_quotas` from the pricing config and
surfaces inline inside `/usage` output:

```
Session usage
─────────────────────────────────────────────
Planner  (claude_code_cli / claude-sonnet-4-6)
  calls: 4
  messages billed: 4

Total messages billed against subscription: 4 / 40 (Pro, 5h window)
▰▰▱▱▱▱▱▱▱▱  10% of window used
```

Warning thresholds (display-only):
- ≥ 50%: dim line "approaching half of your window"
- ≥ 75%: yellow "75% — consider `--estimate` before next `/run`"
- ≥ 90%: red "over 90% — next task may exceed quota"

**Scope for v0.3.2** — session-scoped tracking only. `UsageAccumulator.messages`
resets when the TUI exits. Truly rolling 5-hour windowed tracking across
sessions (persistent state file + timestamp-per-message) is v0.4+ if users
demand it. The display keeps the "window" label so users know what it's
approximating.

### Files

- New: `src/hierocode/broker/pricing.py`, `src/hierocode/broker/quota.py`
- Modified: `src/hierocode/broker/usage.py` (pull pricing from broker/pricing),
  `src/hierocode/shell_handlers/usage_cmd.py` (bar + warnings).

**Ships as:** v0.3.2 on PyPI.

## v0.3.3 + v0.3.4 — bug fixes + apply-UX rework (in flight)

### v0.3.3 (already committed to `origin/main` at `ac1d4a2`, not yet tagged)

Bug-fix release from live integration testing:

- **ollama**: default 5s httpx timeout was too short — 10-minute budget on
  `/api/generate`, 10s on metadata endpoints.
- **codex_cli**: three issues against codex 0.122.0 — `--system` flag was
  removed upstream (system now prepended to prompt), `--approval-policy` was
  removed (dropped from active-exploration mode), hardcoded model names
  (gpt-5, gpt-5-codex, o4-mini) are rejected on ChatGPT-authenticated Codex
  (provider now honors `model="default"` → omit `--model`; wizard ships
  codex_cli with that). `--skip-git-repo-check` always passed.
- **codex_cli JSONL**: codex 0.122+ emits `item.completed → item.agent_message`
  instead of flat `agent_message`. Parser handles both.
- **cli doctor**: used to crash when a provider raised on healthcheck (e.g.
  anthropic without API key). Now reports per-provider and continues.
- 7 regression tests added.

### v0.3.4 — apply-UX rework (in flight)

Per-file confirmation on `/apply` was interrogation for multi-file runs.
This wave makes the applied-diff flow feel like Claude Code:

- **Batch prompt for `/apply`**: one "Apply N files? [yes / review / no]"
  panel instead of N prompts. `review` drops into today's per-file flow when
  cherry-picking is needed.
- **Session-sticky auto-apply**: on the first "yes" the prompt offers "...and
  don't ask again this session." Subsequent `/run` → `/apply` within the
  same TUI session write silently.
- **`hierocode run --apply` / `hierocode draft --apply`**: writes drafter
  output to disk immediately, no prompt. For scripts, CI, and trust-by-default
  workflows. Default behavior unchanged — explicit flag required.
- **`policy.auto_apply: true` config**: standing preference. Wizard asks
  once during `init --wizard`. Off by default.

Ships as **v0.3.4** on PyPI bundled with v0.3.3's bug fixes. v0.3.3 never
becomes a PyPI version — same pattern as v0.3.1 → v0.3.2.

## v0.4 — "Interactive" wave (planned)

- **Streaming output** from all providers. Tokens appear live instead of behind a spinner. Ollama supports it natively; Anthropic SDK has `messages.stream`; Claude CLI supports `--output-format stream-json`; Codex `exec --json` already streams. `BaseProvider.generate_stream` as a new optional method.
- **Interruption.** Ctrl-C cancels the in-flight call and returns to the prompt, does not exit the TUI. Needs careful thread/signal handling per provider.

## v0.5 — "Conversational" wave (planned)

- **Session conversation history.** Follow-up tasks (`"that didn't work, try with exponential backoff"`) see prior `Plan`, diff, and verdict. Planner prompt gains a "conversation context" section.
- **`/clear`, `/compact` equivalents** for managing conversation length.
- **Per-project memory file** (`.hierocode/memory.md`) — accumulated user preferences and project facts, prepended to planner prompts.

## Out of scope (explicit)

- **Fully autonomous agent loop.** Hierocode is a broker, not an agent. The drafter writes what the planner hands it; it does not run bash, grep, or decide on its own which files to touch. Going agentic would blow up the cost story — the cheap-by-default pitch depends on a bounded pipeline.
- **Streaming to apply files as the drafter writes.** Diffs are produced then confirmed; we don't watch a file change under the user's cursor.
- **Running tests automatically.** The dispatcher doesn't invoke pytest or similar. A future `policy.run_tests: <cmd>` could feed test output to QA, but that's not planned for v0.5.

## Release pipeline

- **Triggered by tag push** (`v*`) via `.github/workflows/release.yml`.
- Builds wheel + sdist via `python -m build`, publishes with `pypa/gh-action-pypi-publish` using `${{ secrets.PYPI_API_TOKEN }}`.
- CI on every push to `main` / PR runs pytest + ruff on Python 3.10.

**Current pushed tag:** none yet (v0.2.0 was never tagged; we're shipping v0.3.0 as the first PyPI release of the new architecture).

## Known gaps / TODOs

- `policy.warn_before_escalation: true` is set in wizard YAML but not honored by the
  dispatcher. W27 (v0.3.1) wires it up.
- `--estimate` is projective only; no actuals until W25 lands.
- `broker/skeleton.py` is Python-only — other languages show name + size only.
- Token budgeting uses char/4 heuristic; replace with `tiktoken` if it becomes a pain point.
- No multi-language AST parsing (out of scope unless users complain).
- Integration tests are env-gated; CI skips them. Manual run required before a release
  where provider behavior may have shifted.

## Release checklist (template — for v0.3.1 and beyond)

1. `pytest --ignore=tests/integration` → all green.
2. `ruff check src tests` → all green.
3. Bump `pyproject.toml::version` and `src/hierocode/__init__.py::__version__`.
4. Update this plan.md: move in-flight wave to shipped.
5. Consider README updates if user-visible features changed.
6. `git add src/ tests/ pyproject.toml .gitignore README.md`.
7. `git commit -m "vX.Y.Z: <headline>"` (no `Co-Authored-By` trailer — user preference).
8. `git push origin main`.
9. Wait for CI green on the push commit.
10. `git tag -a vX.Y.Z -m "vX.Y.Z"`.
11. `git push origin vX.Y.Z`.
12. Watch Actions for the release workflow; verify PyPI publish succeeded.
