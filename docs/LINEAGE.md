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

The paper's core claim: context engineering needs a persistent, governed,
filesystem-native substrate — "everything is a file" applied to agent context.
AFS realises each of the paper's components as follows.

| Paper concept | AFS realisation |
|---|---|
| Uniform namespace with typed mounts | `.context/` roots with typed mounts: `scratchpad`, `memory`, `knowledge`, `history`, `tools`, `items`, `hivemind`, `global` (`context_fs.py`, `context_paths.py`) |
| Persistent Context Repository | `.context/` directories plus SQLite context index (`context_index.py`) |
| History — immutable, global source of truth | Event log and history mounts (`event_log.py`, `history.py`); lifecycle events emitted by client wrappers |
| Memory — indexed, mutable, agent-scoped views | Memory mounts plus consolidation of history into durable entries (`memory_consolidation.py`, `llm_memory.py`) |
| Scratchpad — transient but auditable workspace | `scratchpad/` mount; default writable working area per agent contract (`AGENTS.md`) |
| Human input — `/context/human/` annotations | Work-assistant approval records; external writes gated behind human confirmation (`gws_policy.py`, work assistant) |
| Context Constructor — select artefacts under a token budget, emit a manifest | Token-budgeted context packs with caching (`context_pack.py`); retrieval fuses keyword and semantic search via RRF (`context_index.py`, `embeddings.py`) |
| Context Loader/Updater — deliver context into the model window | Session bootstrap and client harness; wrappers inject `AFS_SESSION_*` into Claude, Codex, and Gemini CLIs (`session_bootstrap.py`, `scripts/afs-*`) |
| Context Evaluator — validate outputs, write back with lineage | Schema-validated structured responses (`schema.py`), memory consolidation write-back |
| Mounting external services (MCP servers as mounts) | AFS MCP server exposing `context.*`, `handoff.*`, and optional tool surfaces (`mcp_server.py`) |
| Plugin architecture for new backends | Extension system via `afs.toml` and `extension.toml` manifests (`extensions.py`); domain code lives in extension packages |
| Traceability — every interaction logged | Event history (`events query`), agent run records (`agent_runs.py`), handoff records (`handoff.py`) |

## Delta beyond the paper

Where AFS extends the abstraction rather than implementing it:

- **Harness neutrality.** The paper's reference implementation lives inside one
  framework (AIGNE, TypeScript). AFS treats the harness as pluggable: the same
  `.context/` state serves Claude, Codex, and Gemini through session packs and
  wrapper scripts, so context outlives any single vendor's agent runtime.
- **Cross-language consumers over a file contract.** Non-Python clients and
  runtime integrations can read the same context roots without depending on
  AFS internals. This is direct evidence for the paper's thesis: when the
  filesystem is the interface, integration needs a spec, not a mandatory SDK.
- **Session handoffs.** Structured records of accomplished work, blockers, and
  next steps (`handoff.create`) that the next session's bootstrap ingests —
  cross-session continuity as a first-class protocol rather than a memory
  side-effect.
- **Inter-agent coordination.** The hivemind message bus and task items extend
  the single-agent repository model to explicit multi-agent handoff.
- **Human governance of external writes.** The paper places humans as
  verifiers of uncertain memory. AFS additionally gates all agent writes to
  shared external systems (docs, sheets, tickets) behind approval records
  executed only with human confirmation.
- **Operational tooling.** `afs doctor`, health checks, session reaping, and
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
