# Cortex-AFS Integration

This document describes the integration between Cortex (Swift menu bar application) and AFS (Agent File System, Python agent framework).

## 1. Overview

### What is Cortex?

Cortex is a macOS menu bar application written in Swift. It provides:

- Real-time visibility into agent activity and context state
- Synergy score monitoring
- Quick access to context management functions
- Visual indicators for agent health and activity

### What is AFS?

AFS (Agent File System) is a Python-based agent framework that provides:

- Structured context directories for AI agents
- Mount point management for knowledge, memory, and tools
- Policy-based access control (read-only, writable, executable)
- Plugin architecture for extending functionality
- Orchestration of multiple agent instances

### How They Integrate

Cortex and AFS communicate via a **file-based protocol** using the shared `~/.context/` directory. This approach offers:

- **Simplicity**: No network configuration required
- **Reliability**: File system operations are atomic and well-understood
- **Debuggability**: State can be inspected with standard Unix tools
- **Decoupling**: Either component can restart independently

```
+------------------+     ~/.context/     +------------------+
|     Cortex       |<------------------->|       AFS        |
|   (Swift UI)     |   (file-based)      | (Python agents)  |
+------------------+                     +------------------+
        |                                        |
        v                                        v
   Menu bar UI                           Agent orchestration
   Status display                        Context management
   Quick actions                         Policy enforcement
```

## 2. File Structure Contract

### Expected Directories in `~/.context/`

AFS agents expect the following directory structure to exist:

| Directory | Role | Policy | Description |
|-----------|------|--------|-------------|
| `scratchpad/` | `SCRATCHPAD` | Writable | Agent working memory, ephemeral state |
| `memory/` | `MEMORY` | Read-only | Long-term constraints, knowledge graphs |
| `knowledge/` | `KNOWLEDGE` | Read-only | Reference materials, embeddings |
| `history/` | `HISTORY` | Read-only | Conversation and action logs |
| `hivemind/` | `HIVEMIND` | Writable | Cross-session learning, decisions |
| `tools/` | `TOOLS` | Executable | Agent scripts and utilities |
| `global/` | `GLOBAL` | Writable | Shared state across projects |
| `items/` | `ITEMS` | Writable | Queued items, task lists |

### Directory Policies

Policies are defined in AFS configuration and enforced by agents:

```python
class PolicyType(str, Enum):
    READ_ONLY = "read_only"     # Agents can read but not modify
    WRITABLE = "writable"       # Agents can read and write
    EXECUTABLE = "executable"   # Agents can execute scripts
```

### Additional Directories

Beyond the core mount types, `~/.context/` may contain:

| Directory | Purpose |
|-----------|---------|
| `autonomy_daemon/` | Scheduled task state |
| `embedding_service/` | Vector embedding service state |
| `context_agent_daemon/` | Context manager state |
| `training/` | Model training artifacts |
| `logs/` | Service and agent logs |
| `metrics/` | Performance metrics |
| `models/` | Local model configurations |
| `swarms/` | Multi-agent coordination state |

## 3. Data Formats

### Agent State JSON Format

Agent state files follow this schema:

```json
{
  "agent_id": "string",
  "status": "idle|active|error",
  "last_updated": "2026-01-03T12:00:00Z",
  "current_task": "string|null",
  "progress": 0.0,
  "metadata": {
    "backend": "local|remote",
    "role": "general|coder|critic|researcher|planner"
  }
}
```

### Swarm Status Format

Located at `~/.context/swarm_status.json`:

```json
{
  "nodes": [
    "AgentNode(id='documenter', label='Documenter', status='complete')"
  ],
  "edges": [],
  "active_topic": "Current task description",
  "start_time": "2025-12-21T09:17:07.236155"
}
```

### Synergy Score Format

Synergy scores measure agent coordination effectiveness:

```json
{
  "score": 0.85,
  "timestamp": "2026-01-03T12:00:00Z",
  "components": {
    "task_alignment": 0.9,
    "resource_sharing": 0.8,
    "communication_efficiency": 0.85
  },
  "window_seconds": 300
}
```

### Knowledge Graph Format

Located at `~/.context/memory/knowledge_graph.json` or `~/.context/knowledge/knowledge_graph.json`:

```json
{
  "nodes": {
    "project:oracle-of-secrets": {
      "type": "project",
      "name": "oracle-of-secrets"
    },
    "oracle-of-secrets:routine:GetRandomInt": {
      "type": "routine",
      "name": "GetRandomInt",
      "project": "oracle-of-secrets",
      "address": "$0D:BA71",
      "category": "dungeon"
    }
  },
  "edges": [
    {
      "source": "node_id_1",
      "target": "node_id_2",
      "relation": "calls|references|contains"
    }
  ]
}
```

### Project Metadata Format

Project context metadata stored in `.context/metadata.json` within project directories:

```json
{
  "created_at": "2026-01-03T12:00:00Z",
  "description": "Project description",
  "agents": ["coder", "critic"],
  "directories": {
    "memory": "memory",
    "knowledge": "knowledge",
    "scratchpad": "scratchpad"
  }
}
```

### Scratchpad State Format

The scratchpad may contain various working files. Common patterns:

**state.md** - Current task state in markdown:
```markdown
Task: Description of current task
Progress: Percentage or status
Deliberation: Current reasoning
Reflection: Outcomes and next steps
```

**agent-root-grants.jsonl** - Root access grants (one JSON object per line):
```json
{"host": "halext-nj", "granted_at": "2026-01-03T12:00:00Z", "ttl_seconds": 1800, "commands": ["apt", "systemctl"]}
```

## 4. Reading Context from Cortex

### AFSReader Implementation

Cortex reads context state through the `AFSReader` Swift module. Key behaviors:

```swift
// Paths monitored by AFSReader
let contextRoot = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent(".context")

let monitoredPaths = [
    contextRoot.appendingPathComponent("scratchpad"),
    contextRoot.appendingPathComponent("swarm_status.json"),
    contextRoot.appendingPathComponent("memory/knowledge_graph.json")
]
```

### Polling Interval

Cortex polls for file changes every **5 seconds** by default. This interval balances:

- Responsiveness to state changes
- CPU/IO overhead
- Battery consumption on laptops

### File Change Detection

Cortex uses file modification timestamps for change detection:

```swift
// Pseudo-code for change detection
func checkForChanges() {
    for path in monitoredPaths {
        let currentMtime = getModificationTime(path)
        if currentMtime != lastKnownMtime[path] {
            lastKnownMtime[path] = currentMtime
            reloadContent(path)
            notifyObservers()
        }
    }
}
```

### Error Handling

When reading context files, Cortex handles:

- **Missing files**: Gracefully skip, use defaults
- **Malformed JSON**: Log error, retain last valid state
- **Permission errors**: Display warning in UI
- **Large files**: Stream or paginate reads

## 5. Future API Integration

### Planned REST Endpoints in AFS Gateway

The AFS Gateway service will expose HTTP endpoints for programmatic access:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/agents` | GET | List active agents |
| `/api/v1/agents/{id}` | GET | Get agent state |
| `/api/v1/agents/{id}/tasks` | POST | Submit task to agent |
| `/api/v1/context/scratchpad` | GET/PUT | Read/write scratchpad |
| `/api/v1/context/knowledge` | GET | Query knowledge graph |
| `/api/v1/synergy/score` | GET | Current synergy score |
| `/api/v1/health` | GET | Service health check |

### WebSocket for Real-Time Updates

A WebSocket endpoint will provide real-time state updates:

```
ws://localhost:8765/ws/v1/events

// Event types
{
  "type": "agent_state_changed",
  "agent_id": "documenter",
  "state": {...}
}

{
  "type": "synergy_score_updated",
  "score": 0.87
}

{
  "type": "task_completed",
  "agent_id": "coder",
  "task_id": "abc123"
}
```

### Health Check Protocol

The health check endpoint returns service status:

```json
GET /api/v1/health

{
  "status": "healthy|degraded|unhealthy",
  "timestamp": "2026-01-03T12:00:00Z",
  "components": {
    "orchestrator": "up",
    "embedding_service": "up",
    "context_daemon": "up"
  },
  "version": "0.1.0"
}
```

### Authentication (Planned)

API access will use bearer token authentication:

```
Authorization: Bearer <token>
```

Tokens can be generated via:
```bash
afs auth token generate --name cortex-client
```

## 6. Troubleshooting

### Common Issues

#### File Permission Errors

**Symptom**: Cortex shows stale data or "Permission denied" errors.

**Diagnosis**:
```bash
ls -la ~/.context/
# Check ownership and permissions
# Expected: drwxr-xr-x owned by your user
```

**Fix**:
```bash
chmod -R u+rw ~/.context/
chown -R $(whoami) ~/.context/
```

#### Malformed JSON

**Symptom**: Cortex shows parsing errors or missing data.

**Diagnosis**:
```bash
# Validate JSON syntax
python3 -m json.tool ~/.context/swarm_status.json

# Check for common issues
cat ~/.context/swarm_status.json | head -20
```

**Fix**:
```bash
# Backup and recreate if corrupted
mv ~/.context/swarm_status.json ~/.context/swarm_status.json.bak
echo '{"nodes": [], "edges": [], "active_topic": null}' > ~/.context/swarm_status.json
```

#### Missing Context Directory

**Symptom**: AFS agents fail to start or report missing directories.

**Diagnosis**:
```bash
ls -la ~/.context/
# Should show scratchpad/, memory/, knowledge/, etc.
```

**Fix**:
```bash
# Initialize context structure via AFS CLI
afs init ~/.context

# Or manually create required directories
mkdir -p ~/.context/{scratchpad,memory,knowledge,history,hivemind,tools,global,items}
```

#### Stale Agent State

**Symptom**: Cortex shows agent as "active" but no activity.

**Diagnosis**:
```bash
# Check daemon status
cat ~/.context/autonomy_daemon/daemon_status.json
cat ~/.context/context_agent_daemon/daemon_status.json

# Check for zombie processes
ps aux | grep -E "(afs|agent)"
```

**Fix**:
```bash
# Restart daemons
afs services restart

# Clear stale state
rm ~/.context/scratchpad/swarm/*.json
```

### Log Locations

| Log File | Purpose |
|----------|---------|
| `~/.context/logs/afs.log` | Main AFS service logs |
| `~/.context/logs/embedding.log` | Embedding service logs |
| `~/.context/logs/autonomy.log` | Autonomy daemon logs |
| `~/Library/Logs/Cortex/` | Cortex application logs (macOS) |

### Debug Commands

```bash
# AFS diagnostics
afs doctor                    # Health check
afs status                    # Current state summary
afs context show              # Display context tree

# File system debugging
find ~/.context -name "*.json" -mmin -5    # Recently modified JSON files
du -sh ~/.context/*                        # Directory sizes

# Process debugging
pgrep -fl afs                              # AFS processes
lsof +D ~/.context/                        # Open files in context

# JSON validation
for f in ~/.context/*.json; do
    python3 -m json.tool "$f" > /dev/null 2>&1 || echo "Invalid: $f"
done
```

### Performance Issues

If Cortex or AFS experiences slowdowns:

1. **Check knowledge graph size**:
   ```bash
   ls -lh ~/.context/memory/knowledge_graph.json
   # If > 10MB, consider pruning old nodes
   ```

2. **Check scratchpad accumulation**:
   ```bash
   find ~/.context/scratchpad -type f | wc -l
   # If > 100 files, clean old working files
   ```

3. **Check embedding service**:
   ```bash
   curl http://localhost:8766/health
   # Should respond quickly
   ```

### Getting Help

- AFS Issues: Check `~/.context/logs/afs.log` for errors
- Cortex Issues: Check macOS Console.app for Cortex entries
- File sync issues: Verify disk space with `df -h`
