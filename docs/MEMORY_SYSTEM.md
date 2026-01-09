# AFS Unified Memory System

This document describes the memory and context management architecture for AFS (Agentic File System).

## 1. Architecture Overview

### Three-Tier Context System

AFS organizes agent memory into three tiers based on access patterns and data lifecycle:

```
~/.context/                           # Global context root
├── scratchpad/     [WRITABLE]       # Agent working memory
├── memory/         [READ-ONLY]      # Long-term constraints
└── knowledge/      [READ-ONLY]      # Reference materials
```

| Tier | Access | Purpose | Retention |
|------|--------|---------|-----------|
| **scratchpad** | Agent-writable | Working state, goals, metacognition | Session/ephemeral |
| **memory** | Read-only | Constraints, preferences, learned patterns | Persistent |
| **knowledge** | Read-only | Reference docs, code, domain data | Persistent |

### AFSManager Role in Context Management

The `AFSManager` class (`src/afs/manager.py`) is the primary interface for context operations:

```python
from afs.manager import AFSManager

manager = AFSManager()

# Initialize context for a project
context = manager.init(path=Path("~/myproject"))

# Ensure all required directories exist
context = manager.ensure(path=Path("~/myproject"))

# Mount external knowledge
manager.mount(
    source=Path("~/docs/reference"),
    mount_type=MountType.KNOWLEDGE,
    alias="reference"
)

# List current context state
context = manager.list_context()
```

**Key AFSManager Operations:**

| Method | Purpose |
|--------|---------|
| `init()` | Create new context with full scaffold |
| `ensure()` | Create missing directories, preserve existing |
| `mount()` | Symlink external resources into context |
| `unmount()` | Remove mounted resource |
| `list_context()` | Enumerate mounts and metadata |
| `clean()` | Remove context directory |
| `update_metadata()` | Modify context description/agents |

### Plugin System for Context Extensions

Plugins extend AFS capabilities through the `plugins` module (`src/afs/plugins.py`):

```python
from afs.plugins import discover_plugins, load_plugins

# Auto-discover plugins with configured prefixes
plugin_names = discover_plugins(config)

# Load discovered plugins
loaded = load_plugins(plugin_names, plugin_dirs=[Path("~/.afs/plugins")])
```

**Plugin Discovery:**
- Auto-discovers modules with prefix `afs_plugin` or `afs_scawful`
- Searches configured `plugin_dirs` plus system path
- `AFS_PLUGIN_DIRS` (colon-separated on macOS/Linux) prepends extra plugin folders
- `AFS_ENABLED_PLUGINS` (comma/space separated) appends explicit plugin names
- Defaults to `~/.config/afs/plugins` and `~/.afs/plugins` if present
- Plugins can register custom mount types, validators, or generators

**Plugin Hooks (new surfaces):**
- CLI modules can call `register_cli(subparsers)` or `register_parsers(subparsers)` from plugins.
- Generator backends can be registered via `afs.generators.register_backend`.
- Training format converters can be registered via `afs.training.converters.register_converter`.

---

## 2. Context Directory Structure

### Global Context (`~/.context/`)

The global context root contains agent state that spans all projects:

```
~/.context/
├── scratchpad/                  # Agent-writable working memory
│   ├── state.md                 # Current task state
│   ├── deferred.md              # Postponed items
│   ├── metacognition.json       # Self-assessment (optional)
│   ├── goals.json               # Active goals (optional)
│   ├── emotions.json            # Emotional state (optional)
│   └── epistemic.json           # Certainty levels (optional)
├── memory/                      # Read-only constraints
│   ├── knowledge_graph.json     # Entity/relationship graph
│   └── preferences.json         # User preferences
├── knowledge/                   # Reference materials
│   ├── alttp/                   # Domain-specific knowledge
│   ├── oracle-of-secrets/
│   └── knowledge_graph.json     # Domain graph (routines, addresses)
├── history/                     # Session history
├── hivemind/                    # Cross-agent decisions
├── global/                      # Shared global state
├── items/                       # Tracked items
├── embedding_service/           # Vector embedding config
│   ├── projects.json            # Project configurations
│   ├── daemon_status.json       # Service health
│   └── projects/                # Per-project embeddings
├── training/                    # Training artifacts
├── logs/                        # Service logs
└── metadata.json                # Context metadata
```

### Project Context (`<project>/.context/`)

Per-project context follows the same structure but scoped to a single project:

```
myproject/.context/
├── scratchpad/
│   └── state.md
├── memory/
├── knowledge/
├── metadata.json
└── ...
```

### Cognitive Scaffold Files

When `cognitive.enabled = true` in configuration, AFSManager creates:

| File | Content | Purpose |
|------|---------|---------|
| `state.md` | Markdown | Current task progress |
| `deferred.md` | Markdown | Postponed work items |
| `metacognition.json` | JSON object | Self-assessment metrics |
| `goals.json` | JSON array | Active goal list |
| `emotions.json` | JSON array | Emotional state tracking |
| `epistemic.json` | JSON object | Certainty/uncertainty |

**Example `metacognition.json`:**
```json
{
  "progress_status": "making_progress",
  "cognitive_load": 0.4,
  "strategy_effectiveness": 0.7,
  "frustration_level": 0.1,
  "should_ask_user": false
}
```

---

## 3. Data Lifecycle

### Creation: Agent Initialization

Context is created when:

1. **Explicit init**: `afs context init --path <project>`
2. **Ensure on access**: `manager.ensure()` creates missing structure
3. **Workspace discovery**: `afs context ensure-all` normalizes all contexts

```python
# AFSManager._ensure_context_dirs creates:
# - All configured directory types
# - .keep files for empty directories
# - metadata.json with creation timestamp

# AFSManager._ensure_cognitive_scaffold creates:
# - state.md / deferred.md templates
# - Empty JSON files for cognitive tracking
```

### Read: Querying Context

Agents query context through:

1. **Direct file read**: Read `scratchpad/state.md` for current state
2. **Manager API**: `manager.list_context()` for structured access
3. **Graph export**: `afs graph export` for relationship data
4. **Discovery**: `afs context discover` finds all contexts

```python
# Reading context programmatically
context = manager.list_context(context_path=Path(".context"))

# Access mounts by type
knowledge_mounts = context.get_mounts(MountType.KNOWLEDGE)
scratchpad_mounts = context.get_mounts(MountType.SCRATCHPAD)

# Check validity
if context.is_valid:
    print(f"Total mounts: {context.total_mounts}")
```

## 4. History Logging (CLI + Tools + Models)

AFS records CLI invocations, tool calls, filesystem reads/writes, and model
interactions into the context history log. This supports traceability,
provenance, and training exports.

Default log path:
```
~/.context/history/events_YYYYMMDD.jsonl
```

Large payloads are stored under:
```
~/.context/history/payloads/
```

Each entry is a JSON object with:
- `id` (event id)
- `timestamp` (UTC)
- `type` (`cli`, `tool`, `fs`, `model`)
- `source` (component name)
- `op` (operation)
- `metadata` (small attributes)
- `payload` (inline payload when small) or `payload_ref` (payload file path)

Disable logging by setting `AFS_HISTORY_DISABLED=1`.

### Config Knobs (config.toml)

```toml
[history]
enabled = true
include_payloads = true
max_inline_chars = 4000
payload_dir_name = "payloads"
redact_sensitive = true
```

## 5. Memory to Dataset Export

Export memory entries into `TrainingSample` JSONL via:

```
afs training memory-export --output /path/to/memory_export.jsonl
```

Notes:
- Default memory root: `<context_root>/memory`
- Strict mode only exports entries with explicit `instruction` and `output`
- `--allow-raw` uses `content` or `text` fields as output with a default instruction
- Exports redact common secrets by default; use `--no-redact` to disable.

### Supported Input Types
- `.jsonl`: one JSON entry per line
- `.json`: single entry or list of entries
- `.md` / `.txt`: raw content (only with `--allow-raw`)

### Background Agent (Controlled Launch)

To run memory export on a schedule, use the built-in agent:

```
afs agents run memory-export --dataset-output ~/src/training/datasets/memory_export.jsonl
```

Or start the background service (manual control):

```
afs services start memory-export
afs services stop memory-export
```

The service definition defaults to a 1-hour interval and is **not**
auto-started (`run_at_load = false`).

### Config Knobs (config.toml)

You can control the background agent defaults via:

```toml
[memory_export]
interval_seconds = 3600
dataset_output = "~/src/training/datasets/memory_export.jsonl"
report_output = "~/.context/scratchpad/afs_agents/memory_export.json"
allow_raw = false
allow_raw_tags = ["allow_raw"]
default_instruction = "Recall the following memory entry."
limit = 0
require_quality = true
min_quality_score = 0.5
score_profile = "generic"
enable_asar = false
auto_start = false

[[memory_export.routes]]
tags = ["scribe", "scribe_voice", "voice"]
output = "~/src/training/datasets/scribe_voice.jsonl"
domain = "scribe"
```

### History Export (Training Data)

Export history events into `TrainingSample` JSONL:
```
afs training history-export --output ~/src/training/datasets/history_export.jsonl
```
Notes:
- Exports redact common secrets by default; use `--no-redact` to disable.
- Scoring profiles include `generic`, `dialogue` (shorter responses), and `asm`.

### Codex History Import (AFS Logs)

Import Codex CLI sessions into the AFS history log:
```
afs training codex-history-import --history-root ~/.context/history
```

Notes:
- Default roots: `~/.codex` (plus `~/src/.codex` if present).
- Scans `~/src` for nested `.codex` folders by default (depth 4).
- Exports redact common secrets by default; use `--no-redact` to disable.

### Codex Export (Training Data)

Export Codex CLI logs into `TrainingSample` JSONL:
```
afs training codex-export --output ~/src/training/datasets/codex_export.jsonl
```

Notes:
- Default roots: `~/.codex` (plus `~/src/.codex` if present).
- Scans `~/src` for nested `.codex` folders by default (depth 4).
- Use `--include-system` to include system instructions in inputs.
- Exports redact common secrets by default; use `--no-redact` to disable.

### Antigravity Export (Training Data)

Export Antigravity trajectory summaries into `TrainingSample` JSONL:
```
afs training antigravity-export --output ~/src/training/datasets/antigravity_export.jsonl
```

Notes:
- Default DB path: `~/Library/Application Support/Antigravity/User/globalStorage/state.vscdb`
- Use `--db-path` to override the database location.
- Use `--include-paths-content` to inline `PathsToReview` contents (capped by `--max-path-chars`).
- Exports redact common secrets by default; use `--no-redact` to disable.

### Gemini Export (Training Data)

Export Gemini CLI logs into `TrainingSample` JSONL:
```
afs training gemini-export --output ~/src/training/datasets/gemini_export.jsonl
```

Notes:
- Default roots: `~/.gemini` plus `~/src/.gemini` (if present).
- Scans `~/src` for nested `.gemini` folders by default (depth 4).
- Includes `~/.gemini/disabled` session logs when present.
- Use `--root` to add more `.gemini` directories.
- Use `--scan-root` to control where nested `.gemini` scanning happens.
- Exports redact common secrets by default; use `--no-redact` to disable.

### Claude Export (Training Data)

Export Claude Code logs into `TrainingSample` JSONL:
```
afs training claude-export --output ~/src/training/datasets/claude_export.jsonl
```

Notes:
- Default roots: `~/.claude` plus `~/src/.claude` (if present).
- Scans `~/src` for nested `.claude` folders by default (depth 4).
- Exports redact common secrets by default; use `--no-redact` to disable.

## 6. Research Alignment (OCR Extracts)

OCR reference: `/Users/scawful/Documents/Research/2512.05470v1.pdf`
(`docs/RESEARCH_SOURCES.md`, R1). Extracted via OCR using `ocrmypdf`.

Relevant excerpt (OCR text, verify against the PDF):
> "Agents first write contextual information into a shared memory or store,
> select the most relevant elements for a given task, compress the selected
> context to fit model constraints, and isolate the final subset across agents
> for reasoning."

> "Architectural mechanisms are therefore needed to govern how persistent
> knowledge (long-term memory) transitions into bounded context (short-term
> window) in a traceable, verifiable, and human-aware manner..."

These statements motivate the history log (traceability) and memory export
pipeline (controlled transitions into training context).

### Write: Scratchpad Update Patterns

Only the `scratchpad/` directory is agent-writable. Common patterns:

**State updates:**
```markdown
# state.md
Task: Implement feature X
Progress: 75%
Focus: Unit tests
Blocking: None
```

**Goal tracking:**
```json
// goals.json
[
  {"id": "g1", "description": "Complete feature X", "status": "in_progress"},
  {"id": "g2", "description": "Write documentation", "status": "pending"}
]
```

**Metacognition updates:**
```python
# Agent self-assessment after task completion
metacog = {
    "progress_status": "completed",
    "cognitive_load": 0.2,
    "strategy_effectiveness": 0.9
}
scratchpad_path = context.path / "scratchpad" / "metacognition.json"
scratchpad_path.write_text(json.dumps(metacog, indent=2))
```

### Archive: Long-Term Storage

When scratchpad data should persist:

1. **Move to memory/**: Copy patterns/preferences for future sessions
2. **Export to knowledge/**: Reference materials for other agents
3. **History rotation**: `history/` preserves session logs

```
# Archive workflow
scratchpad/goals.json  ──[complete]──►  history/goals_2025-01-03.json
scratchpad/state.md    ──[archive]───►  memory/patterns/task_patterns.json
```

### Cleanup: Retention Policies

Default retention by directory:

| Directory | Retention | Cleanup Trigger |
|-----------|-----------|-----------------|
| `scratchpad/` | Session | New session start, manual clear |
| `memory/` | Permanent | Manual curation only |
| `knowledge/` | Permanent | Manual curation only |
| `history/` | 30 days | Age-based rotation |
| `logs/` | 7 days | Size/age rotation |

**Cleanup commands:**
```bash
# Remove stale context
afs context clean --path <project>

# Audit and validate
afs agents run context-audit -- --path ~/src
```

---

## 4. Mount Types

### Mount Type Definitions

AFS supports eight mount types defined in `MountType` enum:

```python
class MountType(str, Enum):
    MEMORY = "memory"         # Read-only constraints
    KNOWLEDGE = "knowledge"   # Read-only reference
    TOOLS = "tools"           # Executable scripts
    SCRATCHPAD = "scratchpad" # Agent-writable state
    HISTORY = "history"       # Session logs
    HIVEMIND = "hivemind"     # Cross-agent state
    GLOBAL = "global"         # Global shared state
    ITEMS = "items"           # Tracked items
```

### Access Policies

Each mount type has a default policy:

| Mount Type | Policy | Agent Access |
|------------|--------|--------------|
| `memory` | `read_only` | Read file contents |
| `knowledge` | `read_only` | Read file contents |
| `tools` | `executable` | Execute scripts |
| `scratchpad` | `writable` | Read/write files |
| `history` | `read_only` | Read session logs |
| `hivemind` | `writable` | Cross-agent writes |
| `global` | `writable` | Shared state writes |
| `items` | `writable` | Item tracking |

### Project-Level Context

Per-project context mounts external resources into the project:

```bash
# Mount reference documentation
afs context mount ~/docs/reference --type knowledge --alias reference

# Mount shared tools
afs context mount ~/src/tools/scripts --type tools --alias scripts

# List mounts
afs context list
```

**Mount point structure:**
```
myproject/.context/
├── knowledge/
│   └── reference -> ~/docs/reference  # Symlink
├── tools/
│   └── scripts -> ~/src/tools/scripts  # Symlink
└── scratchpad/
    └── state.md                        # Local file
```

### Global Context (`~/.context/`)

The global context serves as fallback and shared state:

- Located at `~/.context/` (configurable via `AFS_CONTEXT_ROOT`)
- Accessed when project context doesn't exist
- Stores cross-project knowledge graphs
- Manages embedding service configuration

### Temporary Context

For ephemeral operations:

```python
# Link project context to external root
context = manager.init(
    path=Path("myproject"),
    context_root=Path("/tmp/agent-context"),
    link_context=True
)
# Creates: myproject/.context -> /tmp/agent-context
```

---

## 5. Embedding Service

### Configuration: `projects.json`

The embedding service is configured at `~/.context/embedding_service/projects.json`:

```json
{
  "alttp": {
    "name": "alttp",
    "path": "/Users/scawful/src/third_party/usdasm",
    "project_type": "asm_disassembly",
    "description": "",
    "embedding_provider": null,
    "embedding_model": null,
    "enabled": true,
    "priority": 50,
    "include_patterns": ["*.asm", "*.md"],
    "exclude_patterns": ["*.bak", ".git"],
    "max_files": 10000,
    "cross_ref_projects": [],
    "knowledge_roots": []
  }
}
```

### Configuration Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Project identifier |
| `path` | string | Absolute path to source files |
| `project_type` | string | Type hint: `asm_disassembly`, `rom_hack`, `codebase`, `documentation` |
| `embedding_provider` | string/null | Provider name (currently disabled) |
| `embedding_model` | string/null | Model identifier (currently disabled) |
| `enabled` | bool | Include in embedding runs |
| `priority` | int | Processing order (lower = first) |
| `include_patterns` | array | Glob patterns for files to embed |
| `exclude_patterns` | array | Glob patterns to skip |
| `max_files` | int | Limit on files per project |
| `cross_ref_projects` | array | Projects to cross-reference |
| `knowledge_roots` | array | Additional knowledge paths |

### Provider Setup

**Note:** Embedding providers are currently disabled (`embedding_provider: null`).

To enable embeddings, configure a provider:

```json
{
  "embedding_provider": "openai",
  "embedding_model": "text-embedding-3-small"
}
```

Supported providers (when enabled):
- `openai` - OpenAI embeddings API
- `gemini` - Google Gemini embeddings
- `local` - Local embedding model (MLX/llama.cpp)

CLI note: `afs embeddings` currently supports `ollama` and `hf` providers for ad-hoc
indexing/eval without enabling the config-based provider system.

### Query Patterns

Embeddings are stored per-project:

```
~/.context/embedding_service/projects/<project>/embedding_index.json
~/.context/knowledge/<project>/embedding_index.json
```

**Query workflow:**
```python
# 1. Load embedding index
index_path = Path("~/.context/knowledge/alttp/embedding_index.json")
index = json.loads(index_path.read_text())

# 2. Compute query embedding (requires provider)
query_embedding = provider.embed("Sprite movement routines")

# 3. Search index (cosine similarity)
results = search_index(index, query_embedding, top_k=10)
```

---

## 6. Knowledge Graph

### Structure

Knowledge graphs store entity/relationship data:

```json
{
  "nodes": {
    "project:oracle-of-secrets": {
      "type": "project",
      "name": "oracle-of-secrets"
    },
    "oracle-of-secrets:routine:Sprite_Move": {
      "type": "routine",
      "name": "Sprite_Move",
      "project": "oracle-of-secrets",
      "address": "$22:8080",
      "category": "dungeon"
    }
  },
  "edges": [
    {
      "from": "project:oracle-of-secrets",
      "to": "oracle-of-secrets:routine:Sprite_Move",
      "kind": "contains"
    }
  ]
}
```

### Node Types

| Type | Description | Key Fields |
|------|-------------|------------|
| `project` | Project container | `name` |
| `routine` | ASM subroutine | `name`, `address`, `category`, `project` |
| `context` | Context root | `path`, `label` |
| `mount` | Mounted resource | `mount_type`, `source`, `is_symlink` |
| `mount_dir` | Mount directory | `mount_type`, `path` |

### Edge Types

| Kind | From | To | Description |
|------|------|----|-------------|
| `contains` | project/context | routine/mount | Containment |
| `calls` | routine | routine | Call relationship |
| `references` | routine | address | Memory reference |

### Query Patterns

**Find routines by category:**
```python
def find_routines_by_category(graph: dict, category: str) -> list:
    return [
        node for node in graph["nodes"].values()
        if node.get("type") == "routine"
        and node.get("category") == category
    ]

sprites = find_routines_by_category(knowledge_graph, "sprite")
```

**Find call relationships:**
```python
def find_callers(graph: dict, routine_id: str) -> list:
    return [
        edge["from"] for edge in graph.get("edges", [])
        if edge["to"] == routine_id and edge["kind"] == "calls"
    ]
```

### Graph Export

AFS can export context state as a graph:

```bash
# Export to default location
afs graph export

# Export with custom path
afs graph export --output ~/exports/context_graph.json
```

The export includes:
- All discovered contexts
- Mount points and metadata
- Summary statistics

---

## 7. Integration Points

### Book-of-Mudora MCP Server

The Book-of-Mudora MCP server provides domain knowledge access:

```json
// ~/.claude/mcp/book-of-mudora.json
{
  "command": ["node", "book-of-mudora/build/index.js"],
  "args": ["--knowledge-root", "~/.context/knowledge"]
}
```

**Capabilities:**
- Query ALTTP routine addresses
- Search knowledge graph
- Access embedded documentation

### Cortex AFSReader

Cortex integrates with AFS through the AFSReader interface:

```python
# Cortex integration pattern
from cortex.afs import AFSReader

reader = AFSReader(context_root=Path("~/.context"))
state = reader.read_agent_state()
knowledge = reader.query_knowledge("sprite routines")
```

### AFS Gateway

For external service access, AFS exposes a gateway:

```python
# Gateway service (when services.enabled = true)
from afs.services import ServiceManager

manager = ServiceManager(config)
manager.start("afs-gateway")

# Gateway endpoints:
# GET /context - List context state
# GET /knowledge/{path} - Query knowledge
# POST /scratchpad - Update scratchpad
```

### CLI Integration

Full CLI access to memory system:

```bash
# Context management
afs context init --path <project>
afs context ensure --path <project>
afs context list
afs context discover --path ~/src
afs context ensure-all --path ~/src

# Graph operations
afs graph export
afs graph export --output custom.json

# Agent operations
afs agents list
afs agents run context-audit -- --path ~/src
afs agents run context-inventory -- --output report.json

# Service management
afs services list
afs services status afs-gateway
afs services start afs-gateway

# Orchestration
afs orchestrator list
afs orchestrator plan "update documentation" --role researcher
```

---

## Appendix: Configuration Reference

### afs.toml

```toml
[general]
context_root = "~/.context"
agent_workspaces_dir = "~/.context/workspaces"
discovery_ignore = ["legacy", "archive", "archives"]

[[general.workspace_directories]]
path = "~/src"
description = "Main workspace"

[plugins]
auto_discover = true
auto_discover_prefixes = ["afs_plugin", "afs_scawful"]
plugin_dirs = ["~/.afs/plugins"]

[cognitive]
enabled = true
record_metacognition = true
record_goals = true
record_emotions = false
record_epistemic = false

[[directories]]
name = "scratchpad"
policy = "writable"
role = "scratchpad"

[[directories]]
name = "memory"
policy = "read_only"
role = "memory"

[[directories]]
name = "knowledge"
policy = "read_only"
role = "knowledge"

[orchestrator]
enabled = true
max_agents = 5
auto_routing = true

[[orchestrator.default_agents]]
name = "researcher"
role = "research"
backend = "local"
tags = ["knowledge", "search"]

[services]
enabled = false

[services.services.context-audit]
name = "context-audit"
enabled = true
auto_start = false
command = ["python", "-m", "afs.agents.context_audit"]
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AFS_CONTEXT_ROOT` | `~/.context` | Global context root |
| `AFS_CONFIG_PATH` | - | Override config file path |
| `AFS_PREFER_USER_CONFIG` | `true` | User config takes precedence |
| `AFS_PREFER_REPO_CONFIG` | `false` | Repo config takes precedence |
