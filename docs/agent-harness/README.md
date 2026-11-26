# Agent Harness Kit

Portable shared rules, skills, instruction templates, and hooks used by Claude Code, OpenAI Codex, Cursor, and Antigravity (`agy`) / Gemini SDK agents running on top of AFS. Refreshed on 2026-11-26 from the canonical cross-harness setup; zero personal paths or credentials are checked into this directory.

## Contents

| Path | Purpose |
| --- | --- |
| `AGENT_HARNESS_CONTRACT.md` | Shared contract for all harnesses: act-and-finish, debugging, handoffs, messaging, tool selection, usage checkpoints. |
| `AGENT_COMMUNICATION_STANDARD.md` | Response shape, parallel items, judgment calls, exact-text exceptions, handoff payload. |
| `ENGINEERING_PRACTICE_SOURCES.md` | Provenance and upstream notices for domain-modeling, debugging, and verification rules. |
| `REASONING_EFFORT.md` | Task-based reasoning effort guide. |
| `AGENT_STORAGE_HYGIENE.md` | Pre-authorized build-artifact cleanup (`agent-delete`) and worktree preservation rules. |
| `AGENT_AUTOMATION_PLAYBOOK.md` | Host entry points and runtime automation patterns. |
| `skills/` | Six portable `SKILL.md` skills (`focused-verification`, `handoff-writer`, `code-review-findings`, `config-change-safety`, `stale-knowledge-auditor`, `skill-maintainer`). |
| `instructions/` | Global templates: `CLAUDE.md`, `CODEX_AGENTS.md`, `GEMINI.md` (Antigravity `agy` + Gemini SDK), `CURSOR_RULES.mdc`, `HOME_AGENTS.md`. |
| `hooks/` | Hook and MCP config templates: `claude-settings.hooks.json`, `codex-config.toml.snippet`, `cursor-hooks.json`, `cursor-mcp.json`, `antigravity-mcp_config.json`, `gemini-settings.hooks.json`, `gemini-afs_start.sh`. |
| `../../scripts/harness/` | `install-harness-kit.sh`, `afs-session-start`, `agent-pr-gate`, `worktree-preserve`. |

See [`docs/AFS_BEST_PRACTICES.md`](../AFS_BEST_PRACTICES.md) for the full cross-harness integration guide and work-machine upgrade playbook.

## Install on Another Machine

`<AFS_ROOT>` is this checkout (for example `~/src/lab/afs`).

1. Pull and refresh the Python environment:

   ```sh
   cd <AFS_ROOT> && git pull && ./scripts/afs-venv
   ```

2. Run the portable installer (dry-run first, then `--apply`):

   ```sh
   ./scripts/harness/install-harness-kit.sh --render-configs /tmp/afs-rendered
   ./scripts/harness/install-harness-kit.sh --apply --skills --instructions --render-configs /tmp/afs-rendered
   ```

   This links `<AFS_ROOT>/docs/agent-harness` to `~/src/docs/guides/ai-coder`, installs the 6 portable skills into `~/.agents/skills/` with symlinks into `~/.codex/skills/`, `~/.claude/skills/`, and `~/.gemini/config/skills/`, copies instruction files cleanly (without clobbering existing files unless `--force` is passed), and renders path-substituted hook/MCP snippets into `/tmp/afs-rendered/`.

3. Merge rendered hook/MCP configs into your client configs (`~/.claude/settings.json`, `~/.codex/config.toml`, `~/.cursor/mcp.json`, `~/.cursor/hooks.json`, `~/.gemini/config/mcp_config.json`), then verify:

   ```sh
   <AFS_ROOT>/scripts/afs doctor
   <AFS_ROOT>/scripts/afs embeddings probe
   ```

## Notes

- `agent-pr-gate` only gates repos under `~/src/hobby`; elsewhere it is a no-op.
- `worktree-preserve` targets Codex desktop chat/worktree archives. Run its tests with `python3 scripts/harness/test_worktree_preserve.py`.
- Legacy `gemini-cli` (`gemini`) is deprecated; prefer Antigravity (`agy`) and shared `~/.agents/skills/` links.
- Never copy host credentials (`~/.gemini/oauth_creds.json`, tokens, API keys) into git.
