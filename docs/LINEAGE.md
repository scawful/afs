# Research Lineage

AFS is an independent implementation and extension of the agentic file system
abstraction proposed in:

> Xiwei Xu, Robert Mao, Quan Bai, Xuewu Gu, Yechao Li, Liming Zhu.
> **"Everything is Context: Agentic File System Abstraction for Context
> Engineering."** arXiv:2512.05470 [cs.SE], 5 December 2025.
> https://arxiv.org/abs/2512.05470

The arXiv preprint was posted December 5, 2025. Design work applying it began
December 10, 2025 with an early context-panel prototype around the paper's
mount taxonomy; the AFS repository baseline followed on December 30, 2025.
Earlier context tooling was consolidated under this design.

AFS is an independent project and is not affiliated with the paper's authors,
CSIRO Data61, or ArcBlock's AIGNE framework (the paper's reference implementation).

## Concept mapping

The paper's core claim is that context engineering benefits from a persistent,
governed, filesystem-native substrate. AFS adopts that direction, but the table
below is a design mapping rather than a claim of complete conformance. Some
rows are mature runtime paths; others are bounded foundations or compatibility
surfaces.

| Paper concept | AFS realisation |
|---|---|
| Uniform namespace with typed mounts | V1 `.context/` roots use legacy `MountType` roles. Opt-in v2 uses six categories—`history`, `memory`, `scratchpad`, `knowledge`, `tools`, `human`—plus internal `.afs/` state (`context_layout.py`, `models.py`). |
| Persistent Context Repository | Filesystem context plus rebuildable SQLite/FTS and vector indexes (`context_index.py`, `hybrid_search.py`). Project source remains in place. |
| History | Append-only AFS event records and history storage (`event_log.py`, `history.py`). Capture covers instrumented AFS paths; it is not a claim that every host/model interaction is recorded. |
| Memory | Durable entries, readable notes and handoffs, and an optional history-consolidation loop (`artifacts.py`, `handoff.py`, `memory_consolidation.py`). |
| Scratchpad | Writable working state plus immutable draft notes with explicit promotion/archive (`scratchpad.py`). Other legacy subsystems still use compatibility subpaths. |
| Human input | V2 reserves `human/`; current work-assistant and agent-gate flows store human approval/rationale state through compatibility storage (`work_assistant.py`, `agents/guardrails.py`). |
| Context Constructor | Token-budgeted packs plus scoped hybrid retrieval. Ranking uses deterministic reciprocal-rank fusion; semantic retrieval is explicit, not mandatory (`context_pack.py`, `hybrid_search.py`). |
| Context Loader/Updater | Session bootstrap, packs, and client wrappers deliver bounded context to Claude, Codex, Gemini, and compatible harnesses (`session_bootstrap.py`, `scripts/afs-*`). |
| Context Evaluator | Structured-response schemas, verification plans, and optimization decision records cover selected workflows; they are not a universal output evaluator (`schema.py`, `verification.py`). |
| Tool interoperability | The AFS MCP server exposes scoped `context.*`, `messages.*`, `note.*`, and `handoff.*` tools. It exposes AFS to MCP clients; it does not automatically mount arbitrary MCP servers as files (`mcp_server.py`). |
| Plugin architecture | Validated extension manifests add CLI, MCP, source, and agent surfaces (`extensions.py`). |
| Traceability | Instrumented event history, agent run records, immutable handoff revisions, and explicit lifecycle events (`agent_runs.py`, `handoff.py`). |

## Delta beyond the paper

Where AFS extends the abstraction rather than implementing it:

- **Harness neutrality.** The paper's reference implementation lives inside one
  framework (AIGNE, TypeScript). AFS treats the harness as pluggable: the same
  `.context/` state serves Claude, Codex, and Gemini through session packs and
  wrapper scripts, so context outlives any single vendor's agent runtime.
- **Central scoped namespace.** V2 separates stable project identity from a
  checkout path, defaults reads to current-project plus `common`, and requires
  explicit cross-project authorization. The six visible categories stay small;
  registries, messages, and indexes live under `.afs/`.
- **Cross-language file contract.** Non-Python integrations can consume the
  documented files and MCP schemas without importing AFS internals. This is an
  interoperability goal supported by current clients, not proof that every AFS
  subsystem is language-neutral.
- **Session handoffs.** Immutable Markdown revisions record accomplished work,
  blockers, and next steps. Revisions link through `supersedes`; acknowledgement
  and closure are separate append-only lifecycle state.
- **Inter-agent coordination.** Scoped messages extend the single-agent
  repository model. `hivemind` remains a deprecated compatibility name.
- **Human governance of supported external-write flows.** The work assistant
  represents supported external writes as approval records and requires human
  confirmation before its connector execution path. This is not a security
  sandbox and does not intercept arbitrary third-party tools.
- **Operational tooling.** `afs doctor`, `afs health`, session reaping, and
  janitor integration — the maintenance layer a persistent context store needs
  in practice but a paper does not cover.

## What was tried and shed

Between January and April 2026 the repository also carried a speculative
self-improvement stack (discriminator, MoE routing, active learning,
continuous learning). It was moved behind the extension boundary in March 2026
and largely frozen there. The surviving core is the paper-shaped part:
repository, constructor, loader, evaluator, plus the deltas above. Domain and
training workflows live in extensions (see Scope Boundary in
[index.md](index.md)).

## Specs

The file formats are the durable interface. Current contract documents:

- `docs/*` contract pages — shared `.context/` paths, mount policies, and JSON
  formats consumed by compatible clients.
- [MEMORY_SYSTEM.md](MEMORY_SYSTEM.md) — memory and context layout.
- [MCP_SERVER.md](MCP_SERVER.md) — tool surface exposed to MCP clients.
- [ARCHITECTURE.md](ARCHITECTURE.md) — system architecture.
