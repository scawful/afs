# GEMINI.md (Compact)

Purpose: concise Gemini-specific routing for this folder.

Gemini Rules
1. Follow local `AGENTS.md` first when both exist.
2. Minimize scope and avoid unnecessary refactors.
3. Reuse existing project commands and scripts.
4. Run targeted verification before completion.
5. Be explicit when checks were skipped or unavailable.
6. Ask for clarification when requirements are ambiguous.
7. Follow `~/src/docs/guides/ai-coder/AGENT_STORAGE_HYGIENE.md` after builds and
   deployments. Eligible generated-artifact cleanup is pre-authorized; retain
   active work and the specified rollback copies without repeated approval prompts.

Communication Rules
1. Follow `~/src/docs/guides/ai-coder/AGENT_COMMUNICATION_STANDARD.md`.
2. Put the result or next action first.
3. Use numbered, bounded steps for multi-step work.
4. Use literal language and one consistent term for each concept.
5. Preserve exact commands, paths, identifiers, labels, errors, and logs.
6. For judgment calls (design, story, taste, architecture, priority), follow
   the communication standard's `Judgment Calls` section: lead with your pick,
   give 2–4 numbered options with costs, ask one question, and argue
   disagreements before acting.

Response Contract
- What changed
- What was checked
- Known limitations or next actions

Reference Material
- Shared harness contract: `~/src/docs/guides/ai-coder/AGENT_HARNESS_CONTRACT.md`.
- Full legacy guidance: `.context/knowledge/agent-reference.md`.
- Use repository docs as source of truth for implementation details.
