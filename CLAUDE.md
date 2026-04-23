# Claude Workspace Bootstrap

See `AGENTS.md` for startup workflow and working rules.

## Session Recovery

If Claude notices MCP sluggishness, session tool timeouts, repeated missing-tool errors, or obvious stale-session buildup:
1. Run `afs claude doctor --json` first to inspect session counts, bridge protection, and recent debug signals.
2. If cleanup is needed, run `afs claude reap --limit 20` as a dry-run before making changes.
3. Claude may run `afs claude reap --limit 20 --apply` to archive stale or zombie sessions in bounded batches.
4. Never reap `protected` sessions or any project with an active `bridge-pointer.json`.
5. Re-run `afs claude doctor --json` after each batch and stop once the blocking condition clears.

## Handoff Protocol

Before ending a session:
1. Use `handoff.create` to record accomplished work, blockers, and next steps.
2. Update scratchpad state if needed.
3. The next session's bootstrap will include the handoff automatically.
