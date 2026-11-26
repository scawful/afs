# AFS Integration & Best Practices Guide

Canonical operator and multi-harness playbook for running **AFS (`afs-lite`)** across macOS/Linux workstations and syncing your agent rules, MCP setup, hooks, local embeddings, and evals to another machine (such as a work laptop).

---

## 1. Core Design: `afs-lite` & the Frozen 16-Tool MCP Surface

Empirical telemetry across 14 months of AFS usage (`70,063` MCP calls) established that **2 tools (`context.read`, `context.write`) cover 74.1%** of invocations and **5 tools (`context.*` + `messages.*`) cover 97.2%**. Expose only the frozen slim MCP surface (`FROZEN_SLIM_MCP_TOOLS`, 16 tools) to coding agents:

- **Context & discovery**: `context.status`, `context.query`, `context.read`, `context.list`, `context.write`
- **Hivemind coordination**: `messages.send`, `messages.read`, `messages.ack`
- **Notes, skills, & handoffs**: `note.create`, `note.read`, `note.list`, `skill.match`, `skill.read`, `handoff.create`, `handoff.read`, `handoff.list`

Keep `AFS_MCP_TOOL_CATALOG=slim`, `AFS_NATIVE_SKILLS=1`, and `AFS_PROMPT_SCAFFOLDING=minimal`. Avoid deprecated `fs.*` aliases (`removal scheduled in 0.3`).

---

## 2. Cross-Harness Integration Matrix

Portable templates for all clients live in [`docs/agent-harness/`](agent-harness/README.md). Never check personal credentials or machine-specific paths into git; render templates with `scripts/harness/install-harness-kit.sh`.

| Harness | Global Rules Template | Hook / MCP Config Template | Native Skills Path |
|---|---|---|---|
| **Claude Code** | `instructions/CLAUDE.md` -> `~/.claude/CLAUDE.md` | `hooks/claude-settings.hooks.json` (`SessionStart`, `PreToolUse`, `PostToolUse`, `UserPromptSubmit`, `Stop`) | `~/.claude/skills/` -> `~/.agents/skills/` |
| **OpenAI Codex** | `instructions/CODEX_AGENTS.md` -> `~/.codex/AGENTS.md` | `hooks/codex-config.toml.snippet` (`[mcp_servers.afs]`) | `~/.codex/skills/` -> `~/.agents/skills/` |
| **Antigravity (`agy`) & Gemini SDK** | `instructions/GEMINI.md` -> `~/.gemini/GEMINI.md` | `hooks/antigravity-mcp_config.json` -> `~/.gemini/config/mcp_config.json` | `~/.gemini/config/skills/` -> `~/.agents/skills/` |
| **Cursor** | `instructions/CURSOR_RULES.mdc` -> `~/.cursor/rules/afs-agent.mdc` | `hooks/cursor-hooks.json` + `hooks/cursor-mcp.json` -> `~/.cursor/` | Reads `AGENTS.md` / `.cursor/rules/` |
| **Legacy `gemini-cli` (`gemini`)** | *Deprecated* — migrate to `agy` or Gemini SDK | `hooks/gemini-settings.hooks.json` + `hooks/gemini-afs_start.sh` | Do not add new config to `~/.gemini/skills/` |

---

## 3. Session Workflow, Storage Hygiene, & Security Gates

1. **One compact bootstrap**: Let the `SessionStart` hook (`scripts/harness/afs-session-start`) run once, or invoke `<AFS_ROOT>/scripts/afs session bootstrap --short --path <workspace> --json`.
2. **Use the right mount**:
   - Ephemeral drafts, checkpoints, eval logs -> `scratchpad/`
   - Project notes / handoffs -> `note.*`, `handoff.*`
   - Durable `memory/` or `knowledge/` edits -> only when the user asks.
3. **Compare-and-swap on shared files**: Always supply `--if-match <sha256>` on shared writes to prevent silent overwrites.
4. **Approval gates & loopjacking defense (`ApprovalGate` v2)**:
   - Every queued gate binds `payload_sha256` at request time and re-verifies it on `check()`. Swapping payload arguments fails closed and logs `payload_mismatch` in `approval_audit_log`.
   - Inspect audit history anytime with:
     ```bash
     afs approvals audit --limit 25 --json
     ```
5. **Retention & health**:
   ```bash
   afs history reap --dry-run   # preview aged history/scratchpad compression
   afs history reap --apply     # apply configured 30d/14d retention
   afs doctor                   # verify slim catalog census, skills, and indexes
   ```

---

## 4. Encouraging Local-First Semantic Search (`--provider auto`)

See [`docs/EMBEDDINGS_AND_EVALS.md`](EMBEDDINGS_AND_EVALS.md) for the model table and batching internals.

1. Run `afs embeddings probe` to inspect ready backends (LM Studio, Ollama, Apple Silicon MLX, FastEmbed, Gemini, OpenAI) without triggering billable API calls.
2. Build or query your scoped index using local-first auto-selection:
   ```bash
   afs search --semantic --rebuild --provider auto "<query>"
   ```
   `auto` prioritizes loaded local endpoints (`lmstudio` -> `ollama` -> `mlx` -> `fastembed`) before `gemini` (`gemini-embedding-2`) or `openai`, and automatically applies asymmetric prompt instructions (`nomic-embed`, `qwen3-embedding`, `embeddinggemma`, `bge-*`).

---

## 5. Agentic Evals & CI Regression Gating

Use [`src/afs/evals.py`](../src/afs/evals.py) via `afs eval` to catch regressions across retrieval, context packing, safety gates, and agent trajectories:

```bash
# Run the starter multi-surface suite and persist metrics to the context ledger
afs eval run --suite configs/evals/afs_core_eval.jsonl --path . --modes text,hybrid --provider auto --rebuild --save

# List recent runs and gate CI on regressions
afs eval list --limit 10
afs eval compare <baseline_run_id> <candidate_run_id> --fail-on-regression
```

---

## 6. Upgrading Your Work Computer (Step-by-Step)

On a secondary / work workstation after pulling `main`:

```bash
# 1. Pull latest AFS and refresh the local virtualenv
cd ~/src/lab/afs && git pull && ./scripts/afs-venv

# 2. Preview what will be linked/installed (safe dry-run first)
./scripts/harness/install-harness-kit.sh \
  --workspace-root "$HOME/src" \
  --render-configs /tmp/afs-harness-rendered

# 3. Apply skills, guides symlink (~src/docs/guides/ai-coder), and rendered templates
#    Add --instructions (and --force if replacing older copies) when ready
./scripts/harness/install-harness-kit.sh \
  --apply \
  --skills \
  --instructions \
  --workspace-root "$HOME/src" \
  --render-configs /tmp/afs-harness-rendered

# 4. Upgrade project hooks / MCP registrations cleanly
./scripts/afs-upgrade-agent-setup --workspace "$PWD" --apply --all

# 5. Verify health, local embeddings, and starter evals
./scripts/afs doctor
./scripts/afs embeddings probe
./scripts/afs eval run --suite configs/evals/afs_core_eval.jsonl --path "$PWD"
```

Review the rendered snippets in `/tmp/afs-harness-rendered/` before merging into `~/.claude/settings.json`, `~/.codex/config.toml`, `~/.gemini/config/mcp_config.json`, or `~/.cursor/` (`mcp.json` / `hooks.json`), and avoid committing personal work directories into shared configs.
