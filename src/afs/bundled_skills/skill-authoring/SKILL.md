---
name: skill-authoring
triggers:
  - skill authoring
  - SKILL.md
  - skill trigger
  - skill frontmatter
  - afs skills
  - write a skill
  - new SKILL.md
  - distinctive trigger
profiles:
  - general
requires:
  - afs
---

# Skill Authoring

Author SKILL.md files that AFS matches and injects into agent sessions.

## Format

```markdown
---
name: my-skill
triggers:
  - distinctive-word
  - two word phrase
profiles:
  - general
requires:
  - afs
enforcement:
  - Hard rule the agent must follow.
verification:
  - Check the agent should run before claiming success.
---

# My Skill
Body: command tables first, prose last.
```

## Injection Budget (write for it)

- Matching is trigger counting: more distinct trigger hits = higher rank
- At most 10 matches retain metadata; at most 3 get instruction bodies
- Each body is capped at 2,000 characters and all bodies share a 6,000-character cap
- A narrow client hook may inject less; front-load the actionable instructions
- Under prompt pressure, bodies drop before compact enforcement, verification,
  and retrieval pointers
- A skill file may contain at most 64,000 characters; names are capped at 256
  characters; each metadata list may contain 16 items of at most 256 characters
  each. Discovery rejects a skill that exceeds these limits rather than silently
  dropping guardrails

## Commands

| Command | Description |
|---------|-------------|
| `afs skills list` | Discovered skills with roots and triggers |
| `afs skills match "<prompt>" --top-k 5` | Preview what a prompt would load |
| `afs skills mine` | Mine repeated session traces into skill candidates |
| `afs skills review` / `reject` / `archive` | Review mined candidates |
| `afs skills promote --candidate <id>` | Promote a candidate to a starter SKILL.md (user-gated) |

## Discovery

Runtime roots merge profile `skill_roots`, optional `<AFS_ROOT>/skills`, and
the packaged core root. Core source lives at
`src/afs/bundled_skills/<name>/SKILL.md`; wheels use the equivalent
`afs/bundled_skills/` path. Explicit `--root` replaces defaults. Triggerless
skills do not auto-match; mid-session agents use `skill.match` / `skill.read`.
