# AFS Oracle Module

Integration module for Oracle of Secrets development with Triforce expert models and yaze-mcp emulator testing.

## Overview

The Oracle module provides:

1. **Triforce Orchestration** - Routes tasks to specialized expert models
2. **Agentic Testing** - Automated test-debug loops for patch development
3. **Embedding Generation** - Semantic search for Oracle codebase

## Components

### `orchestrator.py` - Triforce Expert Routing

Routes development tasks to specialized fine-tuned models:

| Expert | Role | Task Types |
|--------|------|------------|
| **Nayru** | Code Generation | General 65816 implementation |
| **Din** | Optimization | Performance tuning, cycle reduction |
| **Farore** | Debugging | Crash analysis, logic errors |
| **Veran** | Hardware | DMA, VRAM, register operations |
| **Onox** | Data | Tables, structures, constants |
| **Twinrova** | State/Sprites | State machines, sprite AI |
| **Agahnim** | Build/Validation | Namespace, syntax checks |

**Note:** Din is routed only for explicit optimization requests. Tool calling is not enabled for Din; treat outputs as suggestions and review before applying.

```python
from afs.oracle import TriforceOrchestrator, Expert

orchestrator = TriforceOrchestrator()

# Analyze a task
analysis = orchestrator.analyze_task(
    "Optimize this routine to use fewer cycles"
)
print(analysis.primary_expert)  # Expert.DIN
print(analysis.pipeline)  # [Expert.DIN, Expert.AGAHNIM]

# Invoke an expert
result = await orchestrator.invoke_expert(
    Expert.NAYRU,
    prompt="Implement a health restoration routine",
    context="Link health at $7E036C"
)
```

### `testing.py` - Agentic Test Loop

Automated patch development with test-debug iteration:

```python
from afs.oracle import AgenticTestLoop, OracleTodo, load_oracle_todos

# Load TODOs from project
todos = load_oracle_todos()

# Create test loop
loop = AgenticTestLoop(max_iterations=5, verbose=True)

# Run a TODO through the loop
result = await loop.run_todo(todos[0])

if result.success:
    print("Implementation successful!")
    print(result.final_code)
else:
    print(f"Failed after {len(result.iterations)} iterations")
```

#### OracleTodo Format

```python
todo = OracleTodo(
    id="npc-dialogue-flag",
    title="Add dialogue completion flag for NPCs",
    description="Set flag when NPC dialogue is completed...",
    category="feature",
    suggested_state="sanctuary",
    assertions=["$7E0ABC != 0x00", "!crashed"],
)
```

#### YazeMCPClient

Direct programmatic access to yaze-mcp tools:

```python
from afs.oracle import YazeMCPClient

client = YazeMCPClient()

# Test a patch
result = client.test_oracle_patch(
    patch_content="""
        org $008000
        LDA #$42
        STA $7E0010
        RTS
    """,
    test_state="sanctuary",
    assertions=["$7E0010 == 0x42"],
    frames=60
)

if result.success:
    print("Patch validated!")
```

### `embeddings.py` - Oracle Embedding Generator

Generates embeddings for semantic search over Oracle codebase:

```python
from afs.oracle import OracleEmbeddingGenerator

generator = OracleEmbeddingGenerator(
    oracle_path=Path("~/src/hobby/oracle-of-secrets"),
    output_dir=Path("~/.context/knowledge/oracle-of-secrets/embeddings")
)

# Generate all embeddings
stats = generator.generate_all()

print(stats.summary())
# Total chunks: 1234
# - ASM files: 200
# - ASM routines: 500
# - Symbols: 8438
# - Docs: 34
# - Comments: 62
```

#### Embedding Sources

| Source | Description | Priority |
|--------|-------------|----------|
| ASM Files | Full file content | High |
| ASM Routines | Individual routines | High |
| Symbols | RAM/ROM addresses | High |
| Documentation | Markdown docs | Medium |
| Comments | In-code comments | Medium |

## Tool Routing

Each expert has access to specific tools:

```python
EXPERT_TOOLS = {
    "nayru": ["validate_asm", "test_oracle_patch", "lookup_oracle_docs"],
    "din": ["validate_asm", "test_oracle_patch", "lookup_oracle_docs"],
    "farore": ["read_memory", "add_breakpoint", "step_emulator", "get_disassembly"],
    "veran": ["read_memory", "lookup_snes_register", "get_disassembly"],
    "onox": ["lookup_oracle_docs", "search_oracle_code"],
    "twinrova": ["read_memory", "lookup_alttp_ram"],
    "agahnim": ["validate_asm", "validate_namespace"],
}
```

## Test States

Pre-generated save states enable reproducible testing:

```
~/.context/knowledge/oracle-of-secrets/states/
├── baseline/
│   ├── title_screen.state
│   ├── file_select.state
│   └── new_game_start.state
├── dungeons/
│   ├── sanctuary.state
│   └── hyrule_castle.state
├── bosses/
│   └── armos_knights.state
└── state_library.json
```

## CLI Usage

```bash
# Run agentic test on a TODO
python -m afs.oracle.testing --todo npc-dialogue-flag -v

# List available TODOs
python -m afs.oracle.testing

# Generate embeddings
python -m afs.oracle.embeddings --output ~/.context/knowledge/oracle/embeddings/
```

## Testing

```bash
# Run all Oracle tests
pytest tests/oracle/ -v

# Run specific test module
pytest tests/oracle/test_testing.py -v
pytest tests/oracle/test_embeddings.py -v
pytest tests/oracle/test_orchestrator.py -v
```

## Configuration

### TODO File Location

```
~/.context/knowledge/oracle-of-secrets/todos.json
```

Format:
```json
{
  "todos": [
    {
      "id": "unique-id",
      "title": "Task title",
      "description": "Detailed description",
      "category": "feature|fix|sprite|item|menu",
      "priority": 1,
      "suggested_state": "sanctuary",
      "assertions": ["$7E0010 == 0x42", "!crashed"]
    }
  ]
}
```

### Symbols File

```
~/.context/knowledge/alttp/symbols.json
```

Contains 8,438+ symbols for semantic search and debugging.

## Integration with yaze-mcp

The Oracle module depends on yaze-mcp for emulator operations:

1. **Patch Testing** - `test_oracle_patch` validates assembly and runs tests
2. **State Loading** - `load_test_state` loads pre-generated save states
3. **Memory Operations** - Read/write for assertions and debugging
4. **Breakpoints** - Debugging support for Farore

See `~/src/tools/yaze-mcp/README.md` for yaze-mcp documentation.

## Architecture

```
afs/oracle/
├── __init__.py          # Public API exports
├── orchestrator.py      # Triforce expert routing
├── testing.py           # Agentic test loop, YazeMCPClient
├── embeddings.py        # Embedding generator
├── tools.py             # MCP tool definitions
└── README.md            # This file
```

## Dependencies

- `afs.oracle.tools` - Oracle-specific MCP tools
- `yaze-mcp` - Emulator testing infrastructure
- `asyncio` - Async expert invocation
- `grpcio` - yaze gRPC communication (via yaze-mcp)
