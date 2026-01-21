# AFS (Agentic File System)

Research-only. Not a product.

Scope: core AFS primitives + internal workflow tooling.

**Provenance:** avoid employer/internal sources; skip unclear origins.

## Documentation
- **CLI Reference:** `docs/CLI_REFERENCE.md` (Quickstart, Commands, Installation)
- **Architecture:** `docs/ARCHITECTURE.md`
- **Glossary:** `docs/GLOSSARY.md`
- **Training:** `docs/TRAINING_INFRASTRUCTURE.md`
- **LM Studio:** `LMSTUDIO_SETUP.md`
- **Registry:** `~/src/lab/afs-scawful/config/chat_registry.toml`

## Key Concepts
- **"Context as Files":** Agents read/write to `~/.context`.
- **Workspace Integration:** `afs workspace sync` maps `src` to AFS.
- **Tooling Strategy:** See `docs/TOOLING_STRATEGY.md`.