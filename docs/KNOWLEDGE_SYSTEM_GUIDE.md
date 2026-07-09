# AFS Knowledge System & Gemini Setup Guide

Guide for agents maintaining a generic AFS knowledge mount, rebuilding
embeddings, and setting up Gemini integration on a new machine.

Domain-specific routing tables, private corpus notes, assistant model docs, and
project-specific skills belong in companion extension repos, not in core AFS
docs.

## Knowledge Mounts

AFS does not require a single global knowledge tree. Declare the knowledge roots
that matter to the active profile:

```toml
[profiles.dev]
knowledge_mounts = ["~/work/knowledge", "~/.context/knowledge"]

[profiles.personal]
knowledge_mounts = ["~/personal/knowledge"]
```

A companion repo can also contribute knowledge through `extension.toml`:

```toml
name = "afs_example"
knowledge_mounts = ["knowledge"]
skill_roots = ["skills"]
```

## Adding a Knowledge Document

1. Pick the mount and directory that owns the topic.
2. Write the document in a scannable format with concrete paths, commands, and
   cross-references.
3. Update the local index or routing file for that mount, if it has one.
4. Rebuild embeddings after substantive changes.

Suggested metadata block:

```markdown
# Title — Short Description

**Path**: `<workspace-root>/project/`
**Scope**: work | personal | repo | domain

Brief description paragraph.
```

Keep core knowledge generic. If the document depends on one user's projects,
private corpora, model lineages, game-specific labels, or local machine paths,
store it in the companion repo that owns that domain.

## Updating or Removing Documents

- Read the current document first.
- Edit in place when the scope is unchanged.
- Remove stale routing references when deleting a document.
- Rebuild embeddings after significant additions, removals, or rewrites.

## Embedding Index Management

Keyword-only indexing works without an API key:

```bash
afs embeddings index \
  --knowledge-path ~/.context/knowledge \
  --provider none \
  --include "*.md"
```

Gemini vectors require `GEMINI_API_KEY` or `GOOGLE_API_KEY`:

```bash
afs embeddings index \
  --knowledge-path ~/.context/knowledge \
  --provider gemini \
  --include "*.md"
```

Verify with a targeted search:

```bash
afs embeddings search \
  --knowledge-path ~/.context/knowledge \
  --provider none \
  "project handoff"
```

## Provider Comparison

| Provider | Speed | Quality | Requires | Best for |
|----------|-------|---------|----------|----------|
| `none` | Instant | Keyword-only | Nothing | Offline checks |
| `gemini` | ~1s/doc | Strong semantic retrieval | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | Production semantic search |
| `ollama` | Local-model dependent | Good with the right model | Local Ollama server | Air-gapped/local workflows |
| `openai` | Cloud-model dependent | Strong semantic retrieval | `OPENAI_API_KEY` | Alternative cloud setup |

AFS handles asymmetric retrieval task types for Gemini automatically when running
`index` versus `search`.

## Gemini Setup

```bash
cd <afs-root>
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[gemini]"
export GEMINI_API_KEY="..."
afs embeddings index --knowledge-path ~/.context/knowledge --provider gemini --include "*.md"
```

## Core vs Companion Responsibilities

Core AFS owns:

- mount configuration
- indexing/search commands
- generic profile and extension wiring
- generic docs for maintaining knowledge mounts

Companion repos own:

- domain-specific routing tables
- project-specific `CLAUDE.md` / `AGENTS.md` references
- model/persona corpora
- domain skills and MCP servers
- private or machine-local knowledge inventories

See `docs/EXTENSION_MIGRATION.md` and `docs/PLUGINS.md` for companion repo
layout and discovery.
