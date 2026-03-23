# Gemini Workflow Roadmap

## Goal

Make AFS a better coding scaffold for Gemini without baking Google-internal
behavior into core AFS.

The boundary is:

- core AFS owns generic workflow scaffolding, context shaping, MCP ergonomics,
  and verification rails
- Gemini-specific adapters own model knobs and Gemini API/runtime details
- `afs_google` owns Google-internal auth, corp systems, workspace conventions,
  and private integrations

## Gemini-Oriented Design Targets

Gemini is strong at:

- long-context reading when the context is ordered and clean
- fast extraction and classification
- tool use when schemas and instructions are small and explicit

Gemini needs more help with:

- drifting mid-task when the working contract is implicit
- noisy terminal output and over-wide tool surfaces
- keeping short edit loops disciplined without explicit verification rails

## Feature Set

### Core AFS

1. Prompt/compiler shaping for session packs
   Context first, explicit task at the end, short working contract, and stable
   ordering for repeat calls.

2. Cache-stable context packs
   Deterministic pack ordering and reusable stable prefixes so repeated Gemini
   work can reuse cached context effectively.

3. Execution workflow profiles
   Generic workflows like `scan_fast`, `edit_fast`, `review_deep`, and
   `root_cause_deep` instead of exposing raw model-specific knobs in core.

4. Tool-profile narrowing
   Small preferred AFS surface mixes like `context_readonly`,
   `context_repair`, `edit_and_verify`, and `handoff_only`.

5. Plan -> act -> verify rails
   Small explicit contracts for task execution so the model stays on the
   shortest path and reports missing verification clearly.

6. Output compression
   Summaries for terminal/test/search output so Gemini gets signal instead of
   log spam.

7. Long-context mode selection
   Clear choice between a focused file pack, retrieval pack, or broader repo
   slice instead of ad hoc context stuffing.

8. Retry/fallback behavior
   Smaller-context, tighter-schema, or different-workflow retries for benign
   recovery when a run gets noisy or diffuse.

### Gemini Adapter Layer

- map workflow profiles onto Gemini Flash / Pro / Deep Think usage guidance
- integrate Gemini context caching APIs
- preserve Gemini-specific turn metadata when adapters manage history manually
- expose Gemini-specific prompt templates only where the runtime needs them

### `afs_google`

- Google-internal APIs and auth
- corp-specific workspace bootstrapping
- internal conventions, roots, and provider wiring

## Initial Implementation

The first implementation pass starts on the existing `session pack` surface
instead of adding a parallel Gemini-only subsystem.

Implemented in this pass:

- explicit `task` support in `session pack`
- workflow profiles encoded into the pack
- tool-profile hints encoded into the pack
- rendered packs place the task at the end so the context comes first
- token budgeting reserves space for the workflow contract and task suffix
- built-in `afs://schemas/<name>` resources now expose compact JSON contracts
  for `plan`, `file-shortlist`, `review-findings`, `edit-intent`,
  `verification-summary`, and `handoff-summary`
- MCP prompt `afs.workflow.structured` now combines one of those built-in
  schemas with a normal `session.pack` payload so clients can request a
  schema-bound plan/review/verification response without hand-assembling the
  prompt contract
- `session pack` now supports explicit `pack_mode` selection for `focused`,
  `retrieval`, and `full_slice` context shaping
- MCP tool `operator.digest` now compresses `pytest`, `traceback`, `grep`,
  `diffstat`, and generic command output into compact summaries before that
  output goes back into model context
- `execution_profile` now carries a prompt-only loop policy plus workflow/model
  retry guidance so Gemini can narrow context, switch schemas, or escalate
  models without AFS taking over host-managed session state

This gives Gemini users a better prompt scaffold immediately while keeping the
underlying abstractions generic enough for Claude, Codex, and future adapters.

## Completed Follow-ups

1. Tool-profile enforcement: `AFS_TOOL_PROFILE` env var now filters both
   `tools/list` responses and `tools/call` enforcement via `agent_scope.py`.
   `AFS_ALLOWED_TOOLS` (explicit globs) takes precedence when both are set.
   `is_tool_allowed()` enables non-raising filtering at the MCP layer.
2. Stable prefix hash: `cache.stable_prefix_hash` in context packs excludes
   volatile sections (scratchpad, tasks, hivemind, handoff, health,
   recommendations) so Gemini context-cache adapters can match on the
   knowledge-heavy prefix even when session state drifts between calls.
   Sections also use deterministic `(priority, title)` sort ordering.
3. Structured rail placement: `afs.workflow.structured` stays prompt-only for
   now. Host CLIs keep their own turn loops; AFS carries schemas, tool
   narrowing, operator digests, and retry guidance without introducing
   execution state into core.

## Next Steps

1. Extend `operator.digest` beyond the initial `pytest` / `traceback` / `grep`
   / `diffstat` parsers with richer command-family digests where they prove
   useful in practice.
2. Build the Gemini adapter layer: thinking-level/budget mapping, cache API
   integration, thought-signature handling, Gemini-specific prompt templates.
