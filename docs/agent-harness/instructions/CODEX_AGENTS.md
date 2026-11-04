# Global Codex Instructions

Follow `~/src/docs/guides/ai-coder/AGENT_HARNESS_CONTRACT.md` and
`~/src/docs/guides/ai-coder/AGENT_COMMUNICATION_STANDARD.md`.

1. Put the result or next action first.
2. Use numbered, bounded steps for multi-step work.
3. Use literal language and one consistent term for each concept.
4. Preserve exact commands, paths, identifiers, labels, errors, and logs.
5. Report changed files, verification evidence, blockers, and one next action.
6. Follow `~/src/docs/guides/ai-coder/AGENT_STORAGE_HYGIENE.md` after builds and
   deployments. Eligible generated-artifact cleanup is pre-authorized; retain
   active work and the specified rollback copies without repeated approval prompts.
7. Before any chat/worktree archive, follow that storage policy's preservation
   gate: capture all ignored/untracked files outside the checkout, verify a
   restore, and pass the fresh attachment/file/process check within 60 seconds.
   Pending manual tests or any HOLD block archiving. Direct UI/API archives
   bypass the guard; the automatic-cleanup toggle does not protect them.
8. For judgment calls (design, story, taste, architecture, priority), follow
   the communication standard's `Judgment Calls` section: lead with your pick,
   give 2–4 numbered options with costs, ask one question, and argue
   disagreements before acting.

The `i-have-adhd` plugin is available as `$i-have-adhd`. Apply its response
shape by default. System instructions, safety, correctness, and the requested
task take priority.
