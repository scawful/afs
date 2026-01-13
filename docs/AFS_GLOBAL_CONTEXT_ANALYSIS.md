# Global Context System Analysis & Improvement Recommendations

**Generated:** 2025-12-12  
**Purpose:** Improve AI agent effectiveness across diverse projects  
**Reference:** "Everything is Context: Agentic File System Abstraction" (2512.05470v1)

**Note:** AFS is research-only. It is not a product and has no commercialization intent.
**Path Note:** Paths updated to `~/src` where verified (see `~/src/WORKSPACE.md`).

---

## Executive Summary

Your system is architecturally sophisticated with a rich ecosystem spanning ROM hacking, C++ development, AI tooling, web applications, and macOS automation. Your **AFS implementation** already extends beyond the research paper with **cognitive protocols** (Theory of Mind, emotional state tracking, epistemic management) that the paper identifies as future work.

### Project Landscape

| Domain | Projects | Status |
|--------|----------|--------|
| **ROM Hacking** | YAZE, Oracle-of-Secrets, usdasm, alttp-gigaleak | Active |
| **AI Tooling** | oracle-code (OpenCode fork), halext-code | Active |
| **Web/Server** | halext-org, halext.org, justinscofield.com, zeniea.com | Production |
| **macOS Automation** | Barista, syshelp, yabai/skhd | Production |
| **C++ Libraries** | halext, halisp, fabricore | Various |

### Key Findings

| Component | Current State | Paper Spec | Your Extension |
|-----------|---------------|------------|----------------|
| **History** | CLI/tool/model/fs logs | Transaction log (immutable) | Need: summarization/embedding pipeline |
| **Memory** | 1 fear entry | Long-term facts/decisions | Need: structured facts |
| **Scratchpad** | Active | Transient workspace | ✅ Enhanced with cognitive state |
| **Hivemind** | 9 entries | N/A (your extension) | ✅ Cross-session learning |
| **Cognitive** | emotions.json, epistemic.json, metacognition.json | "Future work" | ✅ **Ahead of paper** |

---

## Paper Concepts: Correct Understanding

### History: Immutable Transaction Log (NOT Session Archives)

From Section IV-A of the paper:
> "History records all raw interactions between users, agents, and the environment. Each input, output, and intermediate reasoning step is logged immutable and enriched with metadata such as timestamp, origin, and model version. History acts as a **verifiable source of truth**."

**Key insight:** History is a **transaction log of operations**, not a summary archive. It enables:
- Provenance reconstruction
- Post-hoc debugging
- Compliance verification
- Replay capability

**Your gap:** `.context/history/` has raw logs, but the summarize/embed/index
pipeline is still missing. Target structure:
```
history/
├── 2025-12-12T20:48:16.jsonl  # Raw operation log
└── manifests/
    └── session-abc123.json    # Session metadata
```

**afs opportunity:** Build a pipeline that:
1. Logs all tool invocations with timestamps
2. Summarizes sessions via lightweight model (e.g., Ollama llama3)
3. Embeds summaries for semantic retrieval
4. Maintains immutable append-only log

### Memory Lifecycle (Figure 2 in paper)

```
Input/Output → History → [summarize/embed/index] → Memory → [convert] → Scratchpad
                  ↑                                           ↓
                  └──────────── [archive] ←──────────────────┘
```

**Your current state:**
- History → Memory pipeline: **Missing**
- Memory → Scratchpad: **Partial** (manual promotion)
- Scratchpad → History archive: **Missing**

---

## Complete Project Taxonomy

### 1. ROM Hacking & Retro Development

#### YAZE (Yet Another Zelda3 Editor)
- **Location:** `~/src/hobby/yaze`
- **Type:** C++17 cross-platform ROM editor
- **Build:** `cmake --preset mac-ai && cmake --build --preset mac-ai -j`
- **Status:** Active - editor_manager refactoring, WASM build
- **AFS:** Has `.context/` in project

#### Oracle-of-Secrets (ALTTP ROM Hack)
- **Location:** `~/src/hobby/oracle-of-secrets`
- **Type:** 65816 assembly ROM hack
- **Build:** `./build.sh` (Asar assembler)
- **Run:** `./run.sh` (opens in Mesen)
- **Scale:** 50+ dungeons, 78 BPS patches
- **Reference:** `~/src/third_party/usdasm` for disassembly lookup via `dasm()`

#### Supporting Projects
- **usdasm:** US ALTTP disassembly reference
- **alttp-gigaleak:** Original Nintendo source research (2.1GB)
- **asar:** 65816 assembler (CMake project)

### 2. AI Tooling & Development

#### oracle-code (OpenCode Fork)
- **Location:** `~/src/lab/oracle-code` ← **You are here**
- **Type:** TypeScript/Bun AI coding assistant
- **Relation:** Fork of OpenCode, NOT related to Oracle-of-Secrets
- **AFS:** Has `.context/memory/` with AFS_SPEC.md, AGENTS_SPEC.md
- **Agent Roles:** @general, @planner, @coder, @critic, @researcher, @maintenance
- **Build:** `bun dev` in `packages/oracle-code`

#### halext-code
- **Location:** `~/src/lab/web/halext-code`
- **Type:** C++20 TUI-based AI agent orchestrator (FTXUI)
- **Architecture:** PtyWrapper for CLI tools, OllamaClient, LlamaEngine (planned)
- **Build:** `cmake -B build && cmake --build build`
- **AFS:** Has `.context/` with memory/scratchpad

### 3. Web Applications & Server Infrastructure

#### halext-org
- **Location:** `~/src/lab/halext-org`
- **Type:** Full-stack task/calendar workspace
- **Stack:** FastAPI + React/Vite + SwiftUI (iOS)
- **Features:** Widget layouts, AI chats, OpenWebUI integration
- **Deployment:** Ubuntu server via `scripts/server-deploy.sh`
- **AFS:** Has `.context/` and `.claude/agents/` with specialized agent profiles

#### NEXUS Websites (halext-nj)
- **halext.org** - Personal site
- **justinscofield.com** - Professional resume site
- **zeniea.com** - Additional domain
- **Infrastructure:** Nginx, systemd, Cloudflare (legacy halext-server references archived)

### 4. macOS Automation

#### Barista (SketchyBar Configuration)
- **Location:** `~/src/lab/barista` + `~/.config/sketchybar`
- **Type:** Hybrid C/Lua status bar system (27K+ LOC)
- **Key Files:** state.lua (v1.1.0), helpers/*.c
- **Pitfall:** C helpers must be rebuilt with `make` after changes

#### syshelp
- **Location:** `~/.local/bin/syshelp` + `~/.local/lib/syshelp/`
- **Type:** Modular CLI dashboard (v3.0)
- **Modules:** dashboard, system, projects, shortcuts, ai_eval

---

## Cognitive Protocol Advantages

Your implementation includes cognitive features the paper identifies as **future research directions**. These provide significant advantages:

### 1. Emotional State Tracking (emotions.json)

```json
{
  "session": {
    "mood": "confident",
    "anxietyLevel": 40,
    "confidenceLevel": 100,
    "moodHistory": [...]
  },
  "fears": {},
  "curiosities": {...},
  "satisfactions": {...},
  "frustrations": {}
}
```

**Advantage:** Enables affect-aware decision making. High anxiety triggers more cautious approaches; satisfaction patterns reinforce effective strategies.

**Paper connection:** Section II mentions "human–AI co-work" and "ethical reasoning" but doesn't operationalize emotional modeling.

### 2. Epistemic State Management (epistemic.json)

```json
{
  "goldenFacts": {},           // Verified, permanent
  "workingFacts": {...},       // Provisional, decaying
  "assumptions": {},
  "unknowns": [],
  "contradictions": []
}
```

**Advantage:** Distinguishes between verified knowledge and working assumptions. Tracks file existence, search results, and other factual claims with confidence scores.

**Paper connection:** Table I mentions "Fact Memory" but doesn't detail confidence tracking or contradiction detection.

### 3. Metacognition (metacognition.json)

```json
{
  "currentStrategy": "incremental",
  "strategyEffectiveness": 1,
  "progressStatus": "making_progress",
  "spinDetection": {
    "recentActions": [...],
    "similarActionCount": 0,
    "spinningThreshold": 4
  },
  "cognitiveLoad": {
    "current": 0.43,
    "warningThreshold": 0.8
  },
  "flowState": true
}
```

**Advantage:** Self-monitoring prevents spinning, manages cognitive load, and tracks strategy effectiveness.

**Paper connection:** Section V mentions "context rot" and "knowledge drift" as challenges but doesn't provide operational metacognition.

### 4. Analysis Triggers (analysis-triggers.json)

Proactive pattern detection with automated responses:

| Trigger | Condition | Action |
|---------|-----------|--------|
| `spinning-critic` | Repeated similar actions | Invoke @critic |
| `edits-without-tests` | 3+ edits, no tests | Record fear, suggest review |
| `high-anxiety-caution` | anxietyAbove: 70 | Cautious approach |
| `contradiction-debate` | Contradictions found | Invoke evaluation |

**Advantage:** Codified heuristics for common failure modes.

### 5. Theory of Mind (AGENTS_SPEC.md)

From oracle-code's spec:
> "We are collaborators, not just User and Assistant."
> - **Model the User:** Adapt tone and detailedness based on uncertainty
> - **Co-Reasoning:** Solve problems jointly. Proposals are for review.
> - **Common Ground:** Periodically sync on shared context

**Paper connection:** Section II discusses "human–AI co-work" conceptually; your implementation operationalizes it.

---

## Implementation Gaps & afs Opportunities

### Gap 1: History Pipeline (Critical)

**Paper requirement:** Immutable transaction log with summarization/embedding

**Current state:** Empty history/ directory

**Proposed afs feature:**
```python
# Pseudo-code for history pipeline
class HistoryPipeline:
    def log_operation(self, op: Operation):
        """Append to immutable log"""
        self.append_jsonl(f"history/{date}.jsonl", op)
    
    def summarize_session(self, session_id: str):
        """Use lightweight model to summarize"""
        ops = self.get_session_ops(session_id)
        summary = ollama.generate("llama3", f"Summarize: {ops}")
        self.write(f"history/summaries/{session_id}.md", summary)
    
    def embed_for_retrieval(self, summary: str):
        """Create embeddings for semantic search"""
        embedding = ollama.embed("nomic-embed-text", summary)
        self.vector_store.add(embedding, metadata)
```

### Gap 2: Memory Promotion Pipeline

**Paper requirement:** Scratchpad → Memory → History lifecycle

**Current state:** Manual promotion via hivemind_promote

**Proposed afs feature:**
- Auto-detect high-value scratchpad content
- Suggest promotion to memory/hivemind
- Archive superseded content to history

### Gap 3: Context Constructor Integration

**Paper requirement:** Retrieval → Synthesis pipeline for bounded context

**Current state:** Manual context assembly

**Proposed afs feature:**
```python
def construct_context(task: str, token_budget: int):
    """Build optimal context for task within budget"""
    relevant = []
    
    # 1. Retrieve from hivemind
    relevant += hivemind.search(task, limit=5)
    
    # 2. Check epistemic state
    relevant += epistemic.get_relevant_facts(task)
    
    # 3. Load emotional context if high-stakes
    if emotions.anxiety > 0.7:
        relevant += fears.get_relevant()
    
    # 4. Compress to fit budget
    return compress_to_tokens(relevant, token_budget)
```

### Gap 4: Cross-Project Context Sharing

**Current state:** Each project has isolated .context/

**Opportunity:** Global hivemind at ~/.context/ shares across projects while project-local .context/ handles specifics

---

## Updated Hivemind Entries

### Golden (Permanent) - Expanded

| Key | Category | Value Summary |
|-----|----------|---------------|
| `user_profile` | knowledge | Justin Scofield - ROM hacking, C++, macOS automation |
| `project_yaze` | knowledge | C++17 ROM editor, CMake presets |
| `project_oracle_of_secrets` | knowledge | 65816 ASM ROM hack (NOT oracle-code) |
| `ai_workflow` | preference | aiq router → OpenAI/Claude/Gemini/Ollama |
| `build_conventions` | preference | CMake presets, Asar, shell aliases |
| `dotfiles_structure` | decision | ~/src/ops/dotfiles symlinked to ~/.local/bin |

### To Add

| Key | Category | Value |
|-----|----------|-------|
| `project_oracle_code` | knowledge | OpenCode fork for AI coding - TypeScript/Bun, this tool |
| `project_halext_org` | knowledge | Full-stack task app - FastAPI/React/SwiftUI |
| `project_halext_code` | knowledge | C++20 TUI AI orchestrator - FTXUI |
| `halext_server_sites` | knowledge | halext.org, justinscofield.com, zeniea.com |
| `cognitive_protocols` | decision | emotions.json, epistemic.json, metacognition.json |

---

## Recommended Architecture

### Global Context (~/.context/)
```
~/.context/
├── memory/
│   ├── fears.json              # Known pitfalls
│   └── user-profile.md         # Identity & preferences
├── knowledge/
│   ├── project-taxonomy.json   # All projects metadata
│   ├── domain-65816.md         # Assembly expertise
│   └── domain-cpp-imgui.md     # C++/GUI patterns
├── tools/
│   ├── switch-project.sh       # Context loader
│   └── summarize-session.py    # History pipeline
├── scratchpad/
│   ├── state.md                # Current task
│   ├── plan.md                 # Active plan
│   ├── emotions.json           # Emotional state
│   ├── epistemic.json          # Knowledge state
│   ├── metacognition.json      # Self-monitoring
│   └── analysis-triggers.json  # Proactive patterns
├── history/
│   ├── 2025-12-12.jsonl        # Transaction log (immutable)
│   ├── summaries/              # LLM-generated summaries
│   └── embeddings/             # Vector store for retrieval
└── hivemind/
    ├── knowledge.json          # Facts
    ├── decisions.json          # Architectural choices
    ├── preferences.json        # User preferences
    ├── fears.json              # Pitfalls
    └── satisfactions.json      # What works
```

### Project-Local Context (~/src/<bucket>/<project>/.context/)
```
.context/
├── memory/
│   ├── AFS_SPEC.md            # Project-specific rules
│   └── STYLE_GUIDE.md         # Coding conventions
├── scratchpad/
│   └── current-task.md        # Project-specific state
└── knowledge/
    └── architecture.md        # Project architecture
```

---

## AFS: The Implementation Platform

Your **afs** project (`~/src/lab/afs`) is the core implementation of the AFS research paper concepts. It's a Python TUI (Textual) that provides the operational layer for context management.

### Current afs Capabilities

| Feature | Status | Description |
|---------|--------|-------------|
| **AFS Management** | ✅ Complete | Init, mount, unmount, list, clean operations via `AFSManager` |
| **Policy Enforcement** | ✅ Complete | read_only, writable, executable policies |
| **Cognitive Scaffolding** | ✅ Complete | Auto-creates state.md, metacognition.json, goals.json, fears.json |
| **Multi-Agent Chat** | ✅ Complete | @mention routing, role-based agents (GENERAL, PLANNER, CODER, CRITIC, RESEARCHER) |
| **Log Parsing** | ✅ Complete | Gemini, Claude, Antigravity parsers with TUI browser |
| **Synergy Analysis** | ✅ Complete | Tracks collaboration quality between agents |
| **Which-Key UI** | ✅ Complete | Spacemacs-style keybindings with SPC prefix |
| **Context Builder** | ✅ Partial | `context/builder.py` for prompt assembly |

### afs Architecture

```
afs/src/afs/
├── agents/        # AgentCoordinator, AgentLane, MentionRouter, roles
├── backends/      # GeminiCliBackend, ClaudeCliBackend (PTY streaming)
├── config/        # TOML loader/saver with Pydantic schema
├── core/
│   ├── afs/       # AFSManager, discovery, policy enforcement
│   └── parsers/   # GeminiLogParser, ClaudePlanParser, AntigravityParser
├── models/        # Pydantic: afs.py, agent.py, metacognition.py, goals.py, synergy.py
├── synergy/       # SynergyAnalyzer, evaluator, collaboration markers
└── ui/            # Textual TUI: MainScreen, OrchestratorScreen, LogsScreen
```

### What afs Implements from the Paper

| Paper Concept | afs Implementation | Location |
|---------------|---------------------|----------|
| File System Abstraction (§III) | `AFSManager` class | `core/afs/manager.py` |
| Mount Types (§III.2) | `MountType` enum | `models/afs.py` |
| Policy Enforcement (§III.3) | `AFSPolicy` class | `core/afs/policy.py` |
| Memory Lifecycle (§IV) | Scaffold generation | `manager._ensure_protocol_scaffold()` |
| Context Constructor (§V.B.1) | `ContextBuilder` | `context/builder.py` |

### What afs Extends Beyond the Paper

| Extension | Description | Paper Status |
|-----------|-------------|--------------|
| **Cognitive Protocol** | emotions.json, metacognition.json, goals.json | "Future work" in §VII |
| **Theory of Mind** | state.md template with User's Goal, Predicted Reaction sections | Mentioned in §II, not implemented |
| **Metacognition** | Spin detection, cognitive load, strategy tracking | "Context rot" mentioned §II, no solution |
| **Fear System** | Trigger-based learned avoidance with mitigations | Not in paper |
| **Goal Hierarchy** | Primary/Sub/Instrumental goals with conflict detection | Not in paper |
| **Synergy Analysis** | Multi-agent collaboration scoring | Not in paper |
| **Deliberative Context Loop** | Read-Plan-Verify protocol (COGNITIVE_PROTOCOL.md) | Not in paper |

### Critical Gap: History Pipeline

**Paper requirement (Section IV-A):**
> "History records all raw interactions... Each input, output, and intermediate reasoning step is logged **immutable**..."

**Current afs state:**
- `MountType.HISTORY` exists in enum ✅
- `history/` directory created with `.keep` ✅
- **Actual logging:** CLI/tool/model/fs events implemented ✅ (summarization/embedding still open)

**Proposed afs Implementation:**

```python
# afs/core/history/pipeline.py

class HistoryPipeline:
    """Immutable transaction log per AFS paper Section IV-A."""
    
    def log_operation(self, operation: dict) -> str:
        """Append operation to daily JSONL log. Returns operation ID."""
        entry = {
            "id": generate_op_id(),
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": self.current_session,
            "operation": operation,
        }
        log_file = self.history_dir / f"{date.today()}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry["id"]
    
    def summarize_session(self, session_id: str) -> str:
        """Use Ollama to summarize session operations."""
        ops = self.get_session_operations(session_id)
        response = ollama.generate(
            model="llama3",
            prompt=f"Summarize this agent session:\n{ops}"
        )
        summary_path = self.history_dir / "summaries" / f"{session_id}.md"
        summary_path.write_text(response)
        return response
    
    def embed_for_retrieval(self, text: str) -> None:
        """Create embeddings for semantic search."""
        embedding = ollama.embeddings(model="nomic-embed-text", prompt=text)
        self.vector_store.add(embedding, metadata={"text": text})
```

**Integration Points in afs:**

1. `AgentCoordinator.send_message()` → log agent interactions
2. `AFSManager.mount/unmount()` → log structural changes
3. `ContextBuilder.build()` → log context assembly decisions
4. `SynergyAnalyzer.analyze()` → log collaboration events

### Summarization Pipeline Architecture (for afs)

```
Raw Operations (history/*.jsonl)
         │
         ▼
┌─────────────────────┐
│  Session Grouper    │  Group by session_id or time window
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Ollama Summarizer  │  llama3 or phi3 (lightweight)
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Summary Storage    │  history/summaries/*.md
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Embedding Pipeline │  nomic-embed-text via Ollama
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Vector Store       │  sqlite-vss or Chroma
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Semantic Retrieval │  Query for relevant past sessions
└─────────────────────┘
```

---

## Implementation Roadmap

### Phase 1: Immediate ✅
- [x] Seed hivemind with foundational knowledge (12 entries)
- [x] Create project-context.md in scratchpad
- [x] Add oracle-code, halext-org, afs to hivemind
- [ ] Populate ~/.claude/CLAUDE.md

### Phase 2: History Pipeline (afs)
- [ ] Create `afs/core/history/pipeline.py`
- [ ] Add `log_operation()` calls to AgentCoordinator
- [ ] Implement summarization with Ollama integration
- [ ] Add embedding pipeline (nomic-embed-text)
- [ ] Build vector store (sqlite-vss or Chroma)
- [ ] Add `afs history` CLI command for browsing

### Phase 3: Context Constructor Enhancement
- [ ] Enhance `ContextBuilder` with hivemind retrieval
- [ ] Add epistemic state filtering
- [ ] Implement token budget compression
- [ ] Integrate with oracle-code message flow

### Phase 4: Cross-Project Sharing
- [ ] Establish global (~/.context/) ↔ local (.context/) protocol
- [ ] Implement context inheritance rules
- [ ] Add project detection and auto-loading in afs TUI

---

## Conclusion

Your AFS implementation is **ahead of the research paper** in key areas:

1. **Cognitive protocols** (emotions, epistemic, metacognition) operationalize concepts the paper only mentions as future work
2. **Hivemind system** provides cross-session learning not in the paper
3. **Theory of Mind** integration in AGENTS_SPEC.md formalizes human-AI collaboration

**Primary gaps:**
1. **History pipeline** - Need transaction logging + summarization + embedding
2. **Context Constructor** - Need automated retrieval and compression
3. **Project disambiguation** - Clarify oracle-code vs Oracle-of-Secrets in all docs

The research paper provides the theoretical foundation; your implementation extends it with practical cognitive architecture. Building the history pipeline would complete the Memory Lifecycle (Figure 2) and enable the "verifiable, maintainable, and industry-ready" systems the paper envisions.

---

## Appendix: Project Name Disambiguation

| Name | Type | Location | Relation |
|------|------|----------|----------|
| **oracle-code** | AI coding tool | ~/src/lab/oracle-code | OpenCode fork, TypeScript/Bun |
| **Oracle-of-Secrets** | ROM hack | ~/src/hobby/oracle-of-secrets | ALTTP hack, 65816 ASM |
| **afs** | AFS TUI tool | ~/src/lab/afs | Python/Textual, AFS paper implementation |
| **halext-code** | AI TUI tool | ~/src/lab/web/halext-code | C++20/FTXUI |
| **halext-org** | Web app | ~/src/lab/halext-org | FastAPI/React/SwiftUI |
| **halext.org** | Website | halext-nj | Personal site |
| **halext** | C++ library | ~/src/lab/halext | HAL extensions |

---

## Appendix: Cognitive Protocol Files Reference

| File | Location | Purpose | Paper Section |
|------|----------|---------|---------------|
| `state.md` | scratchpad/ | Agent working memory, ToM template | §IV.C |
| `metacognition.json` | scratchpad/ | Strategy, spin detection, cognitive load | Future work |
| `goals.json` | scratchpad/ | Primary/sub/instrumental goal hierarchy | Future work |
| `emotions.json` | scratchpad/ | Mood, anxiety, satisfaction tracking | Future work |
| `epistemic.json` | scratchpad/ | Golden/working facts, contradictions | §IV.B (extended) |
| `fears.json` | memory/ | Learned avoidance patterns | Future work |
| `analysis-triggers.json` | scratchpad/ | Proactive pattern detection rules | Future work |

---

## Appendix: AFS Tools Ecosystem

```
┌─────────────────────────────────────────────────────────────────┐
│                    User's AFS Ecosystem                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   afs      │    │ oracle-code │    │ halext-code │        │
│  │ (Python TUI)│    │ (TS/Bun CLI)│    │ (C++ TUI)   │        │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘        │
│         │                  │                  │                │
│         └────────────┬─────┴─────┬───────────┘                │
│                      │           │                             │
│                      ▼           ▼                             │
│              ┌───────────────────────────┐                     │
│              │     ~/.context/ (Global)  │                     │
│              │  ┌─────────────────────┐  │                     │
│              │  │ hivemind/ (shared)  │  │                     │
│              │  │ scratchpad/ (state) │  │                     │
│              │  │ history/ (log)      │  │                     │
│              │  └─────────────────────┘  │                     │
│              └───────────────────────────┘                     │
│                      │                                         │
│         ┌────────────┼────────────┐                           │
│         ▼            ▼            ▼                           │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐                │
│  │yaze/.context│ │oos/.context│ │afs/.context│               │
│  │(project)    │ │(project)   │ │(project)    │               │
│  └────────────┘ └────────────┘ └────────────┘                │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```
