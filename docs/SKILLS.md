# Skills

Skills are reusable instruction files that AFS matches against the current
task and injects into agent sessions. Each skill is one directory containing a
`SKILL.md` with optional frontmatter metadata.

```text
src/afs/bundled_skills/
  health-repair/
    SKILL.md
```

## Quick Commands

```bash
afs skills list --profile work
afs skills match "rebuild a stale context index"
```

MCP tools (in the default slim catalog):

- `skill.match` — rank skills against a task or intent description; set
  `include_bodies: true` to get bounded instructions inline.
- `skill.read` — load one skill's `SKILL.md` body plus metadata by name.
  Reads are root-contained: only files under resolved skill roots are served.

Both MCP tools preserve valid results when another configured skill entry is
malformed or unreadable. Their success payloads include bounded
`diagnostic_count`, `diagnostics_omitted`, and `diagnostics` fields; unknown
`skill.read` errors include a compact warning summary.

Use these mid-session when the task shifts: the launch-time match reflects the
opening prompt, not where the session ends up.

## Discovery

Skill roots are resolved in this order:

1. Explicit overrides (CLI/API callers), which replace everything else.
2. Profile `skill_roots` from the active profile (see `docs/PROFILES.md`),
   merged with the bundled core skills in `src/afs/bundled_skills/`
   (shipped inside the wheel).

Every `SKILL.md` found under those roots is a candidate, bounded to 256 KiB
per file. If a skill declares `profiles:`, it is only visible when the active
profile is listed (or the skill lists `general`). Skills without a `profiles:`
field are visible everywhere. Extensions contribute additional roots via
`skill_roots` in `extension.toml`. Raw discovery does not deduplicate skill
names, so `afs skills list` and `afs skills match` can show more than one file
with the same declared name. Runtime matching resolves those collisions as
described below.

## Frontmatter

```yaml
---
name: health-repair
triggers: ["doctor", "repair", "broken mount", "stale index"]
profiles: ["general"]
requires: ["cli"]
enforcement:
  - Run repair commands in dry-run mode before applying.
verification:
  - afs doctor reports ok for the affected checks.
---
```

Recognized fields:

| Field          | Meaning                                                             |
| -------------- | ------------------------------------------------------------------- |
| `name`         | Skill identity; defaults to the directory name when omitted.        |
| `triggers`     | Phrases matched against the task prompt. No triggers = never auto-matched. |
| `profiles`     | Restrict visibility to these profiles (`profile:` also accepted).   |
| `requires`     | Declared dependencies (informational, e.g. `mcp`, `knowledge/work`). |
| `enforcement`  | Hard rules the agent must follow while the skill is active.         |
| `verification` | How to check the work (aliases: `checks`, `quality_gates`).         |

The parser is a deliberate YAML subset: `key: value`, inline `[a, b]` lists,
and indented `- item` lists. Nested maps and multiline strings are not
supported and are silently ignored — keep frontmatter flat.

## Matching and Ranking

`skills match`, `skill.match`, and the session launch paths use the same scoring
rule: each trigger that appears in the prompt (as a whole word or a substring)
adds one point, and skills scoring zero are dropped. They differ after scoring:

- `afs skills match` ranks the raw discovery results by score. Equal scores keep
  discovery order, which is name-sorted; duplicate names remain visible. Its
  `--top-k` value is applied directly and is not the runtime 10-match cap.
- `skill.match` and session launch use `build_skill_matches`. It deduplicates
  names case-insensitively before scoring, keeping the first discovered entry
  (earlier root, then earlier path within that root). It breaks score ties by
  root priority, then name and path, and returns at most 10 matches. The MCP
  tool validates `top_k` from 1 through 10; session callers are clamped by the
  same 10-match builder cap.

Practical authoring consequences:

- Word-boundary matching does not stem: `verify` does not match
  "verification". List both forms as triggers when both phrasings are likely.
- Multi-word triggers match as substrings, so `"verification plan"` is a
  strong, specific signal.
- A one-trigger runtime match can lose a tie to a skill from an earlier root;
  the CLI preview orders the same score by name instead. Give important skills
  enough distinct triggers to win on score, not on a tiebreak.

## Body Injection

Session launch matches are delivered inline, as are `skill.match` results when
`include_bodies: true` is requested. The CLI `afs skills match` preview is
metadata-only. Runtime body delivery uses two budgets:

- The top **3** matches include their `SKILL.md` body (frontmatter stripped),
  each bounded to **2,000** characters.
- All injected bodies together are bounded to **6,000** characters, spent in
  rank order — a long first body shrinks what later matches may inject.

Truncation cuts inside a match are flagged (`body_truncated: true`); matches
whose body was dropped entirely carry `body_omitted: match_limit` or
`body_omitted: aggregate_limit`. Markdown code fences opened by a cut are
closed so injected prompts stay well-formed. Front-load the essential commands
and rules — anything below the fold only survives for short skills.
Lower-ranked matches are listed by name and path; agents load them on demand
with `skill.read`.

## Bundled Skill Library

Core AFS ships 30 skills. Feature guides map one-to-one onto AFS surfaces:

| Skill | Covers |
| ----- | ------ |
| `session-workflows` | bootstrap, packs, prepare-client, session events |
| `mission-tracking` | mission create/update/list lifecycle |
| `context-search` | `afs query`, `context.query`, index rebuild |
| `context-setup` | context init/ensure/mounts |
| `fs-operations` | `context.read/write/move/delete` file bridge |
| `health-repair` | doctor, context repair, health status |
| `event-log` | history events, analytics, replay |
| `memory-maintenance` | memory consolidation and history-memory agent |
| `hivemind-comms` | hivemind topics, subscribe/send |
| `approvals-and-gates` | work approvals flow and gate discipline |
| `structured-schemas` | schema-bound structured responses |
| `verification-plans` | verification plan authoring and execution |
| `agent-ops` | manifest, runs, jobs, supervisor triggers |
| `task-queue` | background job queue operations |
| `profile-management` | profiles, extensions, overlays |
| `mcp-server` | server registration, catalogs, tool styles |
| `skill-authoring` | writing new skills |
| `extension-authoring` | writing extensions |
| `afs-cli-map` | full CLI command map |

Engineering-practice skills apply across repos: `code-review`,
`implementation-planning`, `software-design`, `cpp-quality`, `python-quality`,
`typescript-quality`.

Team-collaboration skills cover the quality loop between engineers and
agents:

| Skill | Covers |
| ----- | ------ |
| `triage` | classifying, deduping, and routing failures and findings |
| `research` | evidence-first investigation with cited sources |
| `adversarial-verification` | refute-first verification of findings, fixes, and guards |
| `review-handoff` | team-consumable review packages: verdicts, checklists, validation |
| `issue-tracking` | tickets vs missions vs tasks; linking work to evidence |

## Authoring Checklist

1. Create `src/afs/bundled_skills/<name>/SKILL.md` for core skills, or
   `<extension>/skills/<name>/SKILL.md` for extension skills.
2. Add frontmatter with `name` and specific, multi-word `triggers`.
3. Front-load the command table or core rules; keep the body under 2,000
   characters so it always injects whole.
4. Add `enforcement` lines for anything the agent must never skip.
5. Check trigger scoring with `afs skills match "<a realistic task phrase>"`.
   Use `skill.match` or a session bootstrap when verifying runtime duplicate
   resolution, root-priority, match-cap, or body-injection behavior.

The `skill-authoring` bundled skill carries the same checklist in
agent-facing form.
