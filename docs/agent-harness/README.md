# Agent harness kit

Shared rules, skills, instruction templates and hooks used by the Claude,
Codex and Gemini/Antigravity agents that run on top of AFS. This folder is
a snapshot taken on 2026-11-04 from the personal Mac, where the canonical
copies live in `~/src/docs/guides/ai-coder/`, `~/.codex/skills/`,
`~/src/scripts/` and each agent's home config. Refresh it from those copies;
do not fork the rules here.

## Contents

| Path | Purpose |
| --- | --- |
| `AGENT_HARNESS_CONTRACT.md` | One contract for every agent: act-and-finish, domain language and debugging, handoffs, messaging, tool selection, usage checkpoints. |
| `AGENT_COMMUNICATION_STANDARD.md` | Response shape, parallel lists, judgment calls, exact-text exceptions, handoff payload. |
| `ENGINEERING_PRACTICE_SOURCES.md` | Provenance for the domain-modeling, systematic-debugging and Ponytail adaptations, with upstream MIT notices. |
| `REASONING_EFFORT.md` | Task-based reasoning effort guide. |
| `AGENT_STORAGE_HYGIENE.md` | Pre-authorized build-output cleanup and the worktree preservation gate. |
| `AGENT_AUTOMATION_PLAYBOOK.md` | Host entry points referenced by the contract (some scripts are personal-Mac only). |
| `skills/focused-verification/`, `skills/handoff-writer/` | Codex-format skills (`SKILL.md` + `agents/openai.yaml`); Claude and Gemini read the same `SKILL.md`. |
| `instructions/` | Templates for `~/.claude/CLAUDE.md`, `~/.gemini/GEMINI.md`, `~/.codex/AGENTS.md` and `~/AGENTS.md`. |
| `hooks/` | Claude Code hooks, Gemini CLI SessionStart hook, Antigravity AFS MCP config (slim tool catalog). |
| `../../scripts/harness/` | `afs-session-start` (SessionStart briefing), `agent-pr-gate` (PR approval gate), `worktree-preserve` (+ its unittest). |

## Install on another machine

`<AFS_ROOT>` is this checkout (for example `~/src/lab/afs`).

1. Pull and set up the runtime:

   ```sh
   cd <AFS_ROOT> && git pull && ./scripts/afs-venv
   ```

2. Point the doc paths the instruction templates use at this folder, if that
   path does not already exist:

   ```sh
   mkdir -p ~/src/docs/guides && ln -s <AFS_ROOT>/docs/agent-harness ~/src/docs/guides/ai-coder
   ```

3. Copy the instruction templates you want, after reviewing any existing file:

   ```sh
   cp docs/agent-harness/instructions/CLAUDE.md ~/.claude/CLAUDE.md
   cp docs/agent-harness/instructions/GEMINI.md ~/.gemini/GEMINI.md
   cp docs/agent-harness/instructions/CODEX_AGENTS.md ~/.codex/AGENTS.md
   ```

4. Merge the `hooks` block from `hooks/claude-settings.hooks.json` into
   `~/.claude/settings.json`, and the one from `hooks/gemini-settings.hooks.json`
   into `~/.gemini/settings.json`. Replace `<AFS_ROOT>` and `<HOME>` with
   absolute paths. Install `hooks/gemini-afs_start.sh` as
   `~/.gemini/hooks/afs_start.sh`.

5. For Antigravity, merge `hooks/antigravity-mcp_config.json` into
   `~/.gemini/config/mcp_config.json` (same `<AFS_ROOT>` replacement).

6. Copy the skills into each host's skill root (for example
   `~/.codex/skills/`, `~/.claude/skills/`).

Verify with `<AFS_ROOT>/scripts/afs session bootstrap --short --path "$PWD"`
and one Claude or Gemini session start: the first turn shows
"AFS short bootstrap for …".

## Notes

- `agent-pr-gate` only gates repos under `~/src/hobby`; elsewhere it is a no-op.
- `worktree-preserve` targets Codex desktop chat/worktree archives. Run its
  tests with `python3 scripts/harness/test_worktree_preserve.py`.
- Never copy host credentials (`~/.gemini/oauth_creds.json` and similar) into
  this repo; templates carry paths and tool lists only.
