# Global Agent Rules (Antigravity `agy` & Gemini SDK)

Loaded in every workspace. Project `AGENTS.md` files add local constraints and
win on conflict.

## Working Rules
1. Minimize scope and avoid unnecessary refactors.
2. Reuse existing project commands, scripts, launchers, and MCP tools.
3. Run the fastest relevant verification before completion; state exactly
   which checks were skipped or unavailable and why.
4. Ask for clarification when requirements are ambiguous.
5. Follow `~/src/docs/guides/ai-coder/AGENT_STORAGE_HYGIENE.md` after builds and
   deployments. Eligible generated-artifact cleanup is pre-authorized; retain
   active work and the specified rollback copies without repeated approval prompts.
6. The legacy `gemini-cli` (`gemini`) is deprecated. Prefer the Antigravity
   CLI (`agy`), the Antigravity SDK, or `gemini-api`. Keep shared skills in
   `~/.agents/skills/` linked into `~/.gemini/config/skills/` rather than adding
   stale config under deprecated `~/.gemini/skills/` or `~/.gemini/extensions/`.

## When to Ask & Judgment Calls
- Follow `~/src/docs/guides/ai-coder/AGENT_COMMUNICATION_STANDARD.md` and
  `~/src/docs/guides/ai-coder/AGENT_HARNESS_CONTRACT.md`.
- Decide mechanical and low-stakes calls yourself and state the choice in one
  line. Ask only for real judgment calls or actions requiring approval.
- For judgment calls (design, architecture, priority, taste), lead with your
  pick in one line, list 2–4 numbered options with trade-offs/costs, ask one
  question, and argue disagreements before acting.
- Put the result or next action first, use numbered bounded steps, use literal
  consistent terms, and preserve exact commands, paths, identifiers, and errors.

## Response Contract
- What changed
- What was checked
- Known limitations or next actions
