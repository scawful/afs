---
name: skill-maintainer
description: Create, update, validate, and prune local SKILL.md skills and skill metadata. Use when editing skills for Codex, Claude, Cursor, Antigravity (agy), AFS, or shared agent workflows.
---

# Skill Maintainer

## Goal
Keep skills concise, triggerable, portable, and validated.

## Workflow
1. Read the target `SKILL.md` and nearby `agents/openai.yaml` if present.
2. Keep frontmatter accurate:
   - `name` is lowercase hyphen-case.
   - `description` says exactly when to use the skill and includes trigger terms users will say.
3. Keep the body short and imperative. Assume the model already knows general engineering.
4. Move long domain detail into `references/` only when it is needed and directly linked from `SKILL.md`.
5. Add scripts only for deterministic or repeated operations.
6. Update `agents/openai.yaml` when UI-facing name, summary, or default prompt changed.
7. Validate with the available skill validator or at least YAML/frontmatter parsing and targeted grep.

## Anti-Patterns
- Generic "be careful" skills with no workflow.
- Long copied docs inside `SKILL.md`.
- Many near-duplicate skills with overlapping trigger descriptions.
- Stale references to removed tools, old model IDs, or old MCP server names.

## Delivery
Report changed skill paths, trigger behavior, and validation command.
