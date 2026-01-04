# Feature Map (legacy -> current)

Scope: inventory of legacy features and their status in the current AFS/AFS Scawful repos. Use this as a porting checklist; verify specifics when moving code.

Sources (local workspace):
- Previous core: `lab/legacy/afs-legacy`
- Previous plugin: `trunk/scawful/research/afs_scawful`
- Current core: `lab/afs`
- Current plugin: `lab/afs-scawful`

Status legend: Ported, Partial, Planned, Not started.

## AFS Core
- Context roots + workspace registry: Ported (config + CLI + workspaces.toml). `src/afs/config.py`, `src/afs/cli.py`.
- Context validation/mapping/policy/graph: Ported. `src/afs/manager.py`, `src/afs/mapping.py`, `src/afs/policy.py`, `src/afs/validator.py`, `src/afs/graph.py`.
- Discovery + ensure-all: Ported. `src/afs/discovery.py`, `src/afs/cli.py`.
- Studio UI (ImGui): Ported (afs_studio standalone repo; legacy mirror in apps/studio). Active.
- CLI surface: Partial (core context + graph + plugins; orchestration/services missing).
- Plugin framework + adapters: Partial (discovery + load only; adapters not yet).
- Orchestration/pipelines/swarm: Partial (routing skeleton + CLI).
- Service management + daemons: Partial (definitions + render preview + adapters).
- TUI: Not started.
- API/backends/integrations/models/llm: Not started.
- Native/cc/editor tooling: Not started.

## AFS Scawful Plugin
- Training paths/resources config: Ported. `src/afs_scawful/config.py`.
- Dataset registry indexing: Ported. `src/afs_scawful/registry.py`, `src/afs_scawful/cli.py`.
- Resource indexing: Ported. `src/afs_scawful/resource_index.py`, `src/afs_scawful/cli.py`.
- Training monitor config: Ported (AFS Studio loads plugin config).
- Validators (ASM/C++/KG/ASAR): Partial (base + initial ports).
- Generators (ASM/Oracle/curated/etc.): Partial (doc-section generator + base).
- Ops/runbooks/host scripts: Not started (needs reorg as CLI tasks).

## Near-term ports (priority)
1) Service adapters (launchd/systemd) in dry-run mode.
2) Generator base + one small generator (plugin).
3) TUI starter screen (status + contexts).
