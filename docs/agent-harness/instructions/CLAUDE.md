# Global Claude Instructions

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

## Judgment calls

A judgment call is a design, story, lore, taste, architecture, or priority
choice. It is not a mechanical choice with a checkable answer (which test to
run, which fix makes the build pass); keep recommending on those.

1. Lead with your pick in one line: what you would do and why.
2. List 2–4 numbered options, one line each: what it gets me, what it costs.
3. Your pick may be the first `AskUserQuestion` option or `wrap_up` button,
   marked as your pick. Make the buttons answer the question directly.
4. Ask one question per message. I should be able to answer with a number
   or one word.
5. When you disagree with my plan or stated preference, say so before acting
   and make the strongest case against it. Do not silently do something else.

The `i-have-adhd` plugin is installed. Its always-on flag is
`~/.claude/.i-have-adhd-always`. System instructions, safety, correctness, and
the requested task take priority. The judgment-call rules above match the
plugin's "recommendation first" default.
