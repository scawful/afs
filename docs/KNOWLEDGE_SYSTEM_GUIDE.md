# AFS Knowledge System & Gemini Setup Guide

Guide for agents maintaining the knowledge base, rebuilding embeddings, and
setting up Gemini integration on a new machine.

## Part 1: Knowledge System Maintenance

### Knowledge Base Location

All knowledge docs live at `~/.context/knowledge/` organized by domain:

```
~/.context/knowledge/
├── INDEX.md              ← master routing doc (start here)
├── hobby/                ← ROM hacking projects
├── alttp/                ← ALTTP game engine internals
├── snes/                 ← SNES hardware reference
├── models/               ← model training & serving
└── oracle-of-secrets/    ← structured data (JSON)
```

The knowledge base is mounted into AFS via the active profile's
`knowledge_mounts` in `~/.config/afs/config.toml`:

```toml
[profiles.dev]
knowledge_mounts = ["$AFS_ROOT-scawful/knowledge", "~/.context/knowledge"]
```

### Adding a New Knowledge Document

1. **Pick the right directory** based on domain:
   - `hobby/` — project-specific (Oracle, YAZE, tools)
   - `alttp/` — game engine (architecture, routines, data tables, sprites)
   - `snes/` — hardware (CPU, PPU, DMA)
   - `models/` — training pipeline, portfolio, serving, datasets

2. **Write the document** following the existing format:
   ```markdown
   # Title — Short Description

   **Path**: `<workspace-root>/project/`
   **Stage**: Alpha | **Language**: 65816 ASM | **Build**: Asar

   Brief description paragraph.

   ## Section with Tables

   | Column | Column | Column |
   |--------|--------|--------|
   | data   | data   | data   |
   ```

   Style rules:
   - Header block with Path, Stage, Language, Build metadata
   - Use tables heavily — scannable beats prose
   - Include concrete addresses, file paths, command examples
   - Cross-reference other knowledge docs by relative path
   - No emojis

3. **Update INDEX.md** — add the doc to both sections:
   - "By Task" table (if it maps to a common task)
   - "By Directory" file listing

4. **Update routing** — add "Also See" pointers in relevant places:
   - Project's `CLAUDE.md` (Reference Knowledge table)
   - Project's `.context/CONTEXT_INDEX.md` (if Oracle)
   - Relevant AFS skills (Knowledge References section)

5. **Rebuild embeddings** (see Part 2)

### Updating an Existing Document

1. Read the current doc first
2. Edit in place — don't create a new file
3. Rebuild embeddings after significant changes
4. No need to update INDEX.md unless the doc's scope changed

### Removing a Document

1. Delete the file
2. Remove from INDEX.md
3. Remove any routing references (CLAUDE.md, skills, CONTEXT_INDEX.md)
4. Rebuild embeddings

### Current Document Inventory

Run this to see what's indexed:

```bash
afs embeddings search --knowledge-path ~/.context/knowledge --provider none "" --min-score 0.0 --top-k 100
```

Or just list the files:

```bash
find ~/.context/knowledge -name "*.md" | sort
```

### Routing Architecture

Agents discover knowledge through three independent paths:

```
1. CLAUDE.md (auto-loaded by Claude Code)
   └── "Reference Knowledge" table → specific docs by task

2. CONTEXT_INDEX.md (Oracle project routing)
   └── "Global Knowledge Base" section → docs by topic
   └── Per-domain "Also See" columns → specific docs

3. AFS Skills (invoked by task type)
   └── "Knowledge References" section → relevant docs

4. Embeddings search (programmatic)
   └── afs embeddings search → semantic/keyword matching
   └── afs gemini context → context generation for sessions
```

If you add a new doc, wire it into at least path 1 (CLAUDE.md) and path 4
(rebuild embeddings). Paths 2 and 3 are nice-to-have for discoverability.

### Files That Route to Knowledge

| File | Location | What to update |
|------|----------|----------------|
| `INDEX.md` | `~/.context/knowledge/` | "By Task" + "By Directory" tables |
| `CLAUDE.md` | `<oracle-root>/` | "Reference Knowledge" table |
| `CLAUDE.md` | `<yaze-root>/` | "Reference Knowledge" table |
| `CONTEXT_INDEX.md` | `<oracle-root>/.context/` | "Global Knowledge Base" + domain tables |
| `mesen2-oos-debugging/SKILL.md` | `<afs-ext-root>/skills/` | "Knowledge References" |
| `alttp-disasm-labels/SKILL.md` | `<afs-ext-root>/skills/` | "Knowledge References" |
| `hyrule-navigator/SKILL.md` | `<afs-ext-root>/skills/` | "Knowledge References" |
| `zelda-model-manager/SKILL.md` | `<afs-ext-root>/skills/` | "Knowledge References" |
| `echo-persona/SKILL.md` | `<afs-ext-root>/skills/` | "Knowledge References" |
| `model-training-expert/SKILL.md` | `<afs-ext-root>/skills/` | "Knowledge References" |

---

## Part 2: Embedding Index Management

### Building the Index

```bash
# Keyword-only (no API key needed, works anywhere)
afs embeddings index \
  --knowledge-path ~/.context/knowledge \
  --provider none \
  --include "*.md"

# With Gemini vectors (requires GEMINI_API_KEY)
afs embeddings index \
  --knowledge-path ~/.context/knowledge \
  --provider gemini \
  --include "*.md"
```

The index is stored at:
- `~/.context/knowledge/embedding_index.json` — doc ID → filename mapping
- `~/.context/knowledge/embeddings/` — per-document JSON with vectors

### When to Rebuild

Rebuild after:
- Adding or removing knowledge documents
- Significantly rewriting a document's content
- Switching embedding providers (e.g., keyword → Gemini)

No rebuild needed for:
- Minor typo fixes
- Updating routing files (INDEX.md, CLAUDE.md, skills)

### Verifying the Index

```bash
# Check doc count
afs embeddings index --knowledge-path ~/.context/knowledge --provider none --include "*.md"
# Should report: total=N indexed=N skipped=0 errors=0

# Test search
afs embeddings search --knowledge-path ~/.context/knowledge --provider none "sprite RAM tables"
# Should return oracle-sprite-ram.md as top result
```

### Provider Comparison

| Provider | Speed | Quality | Requires | Best for |
|----------|-------|---------|----------|----------|
| `none` | Instant | Keyword-only | Nothing | Offline, quick setup |
| `gemini` | ~1s/doc | Best semantic | `GEMINI_API_KEY` | Production use |
| `ollama` | ~0.5s/doc | Good semantic | Local Ollama server | Air-gapped/local |
| `openai` | ~0.5s/doc | Good semantic | `OPENAI_API_KEY` | Alternative cloud |

### Asymmetric Retrieval (Gemini)

Gemini uses different task types for indexing vs searching:
- Indexing: `RETRIEVAL_DOCUMENT` (how the doc should be found)
- Searching: `RETRIEVAL_QUERY` (what the user is looking for)

AFS handles this automatically. The CLI switches task type based on
whether you're running `index` or `search`.

---

## Part 3: Gemini Integration Setup (New Machine)

### Prerequisites

- Python 3.10+
- AFS installed: `pip install -e .` from `<afs-root>/`
- Google API key with Gemini access

### Step 1: Install AFS with Gemini Support

```bash
cd <afs-root>
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[gemini]"
```

If you only need HTTP-based embeddings (no SDK):

```bash
pip install -e ".[embeddings]"
```

### Step 2: Set API Key

```bash
export GEMINI_API_KEY="your-api-key-here"

# Add to shell profile for persistence
echo 'export GEMINI_API_KEY="your-api-key-here"' >> ~/.zshrc
```

The key is resolved in this order:
1. `--gemini-api-key` CLI flag
2. `GEMINI_API_KEY` environment variable
3. `GOOGLE_API_KEY` environment variable

### Step 3: Register AFS MCP Server

```bash
afs gemini setup
```

This writes the AFS MCP entry into `~/.gemini/settings.json`. Verify:

```bash
cat ~/.gemini/settings.json
```

Should contain:

```json
{
  "mcpServers": {
    "afs": {
      "command": "/path/to/python3",
      "args": ["-m", "afs.mcp_server"]
    }
  }
}
```

### Step 4: Configure AFS

Create or update `~/.config/afs/config.toml`:

```toml
[general]
context_root = "/path/to/.context"
mcp_allowed_roots = ["/path/to/src"]

[[general.workspace_directories]]
path = "/path/to/src"
description = "Source code"

[profiles]
active_profile = "dev"

[profiles.dev]
knowledge_mounts = ["/path/to/.context/knowledge"]
```

Apply the profile:

```bash
afs context profile-apply
```

### Step 5: Sync Knowledge Base

Copy or symlink the knowledge directory to the new machine:

```bash
# If using the same filesystem layout
ln -s /shared/path/.context/knowledge ~/.context/knowledge

# Or rsync from another machine
rsync -av source-machine:~/.context/knowledge/ ~/.context/knowledge/
```

### Step 6: Build Embedding Index

```bash
afs embeddings index \
  --knowledge-path ~/.context/knowledge \
  --provider gemini \
  --include "*.md"
```

### Step 7: Verify Everything

```bash
afs gemini status
```

Expected output (all OK):

```
  [     ok] API key: GEMINI_API_KEY or GOOGLE_API_KEY
  [     ok] google-genai SDK: pip install google-genai
  [     ok] settings.json: /Users/you/.gemini/settings.json
  [     ok] MCP registered: /Users/you/.gemini/settings.json
  [     ok] Embeddings indexed: 37 docs at /Users/you/.context/knowledge
  [     ok] Embedding ping: 3072-dim vectors
```

Test search:

```bash
afs embeddings search \
  --knowledge-path ~/.context/knowledge \
  --provider gemini \
  "how to debug a sprite"
```

Test context generation:

```bash
afs gemini context "sprite development"
afs gemini context   # full index dump
```

### Step 8: Test MCP Integration

Start the MCP server manually to verify:

```bash
afs mcp serve
```

Or test via Gemini CLI if installed:

```bash
gemini   # should auto-discover AFS MCP tools
```

### Troubleshooting

| Problem | Fix |
|---------|-----|
| `afs: command not found` | `pip install -e .` or use `./scripts/afs` |
| `GEMINI_API_KEY not set` | `export GEMINI_API_KEY=...` |
| `google-genai not installed` | `pip install -e ".[gemini]"` |
| Embeddings index shows 0 files | Check `--include "*.md"` flag, verify knowledge path exists |
| MCP not registered | Run `afs gemini setup` |
| Search returns no results | Rebuild index, check `--min-score` threshold |
| `settings.json` in wrong location | Use `afs gemini setup --settings-path /correct/path` |
| SDK import error | `pip install google-genai>=1.0.0` |
| HTTP fallback 404 | Check model name — use `gemini-embedding-001` not `text-embedding-004` |

### Minimal Setup (Keyword-Only, No API Key)

If you just need the knowledge system without cloud embeddings:

```bash
pip install -e .
afs embeddings index --knowledge-path ~/.context/knowledge --provider none --include "*.md"
afs embeddings search --knowledge-path ~/.context/knowledge --provider none "your query"
```

This uses keyword matching against document text — no API key, no network,
works fully offline. Quality is lower than Gemini vectors but still useful.
