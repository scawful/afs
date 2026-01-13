# AFS Tooling Strategy: "Context as Files"

## Core Philosophy
In the Agentic File System (AFS), agents do not use traditional JSON-RPC tool calls (like ChatGPT plugins). Instead, they interact with the world by **reading and writing files** in standard locations.

This "Context as Files" approach leverages the file system as a universal interface, persistent memory, and shared bus between agents.

## Architecture

### 1. The Mount Point (`~/.context`)
All active context is mounted here.
- `metadata.json`: Current session goals, active agents.
- `scratchpad/`: The "Working Memory". Agents plan here.
- `memory/`: Long-term docs and decisions.
- `tools/`: Executable scripts the agent can run (via shell).

### 2. The Interaction Loop
Instead of "Calling a Function", an agent:
1.  **Observes:** Reads `scratchpad/state.md` or `scratchpad/plan.md`.
2.  **Reasons:** Thinks about the next step.
3.  **Acts:**
    *   **Writes** a new plan to `scratchpad/plan.md`.
    *   **Writes** code to `src/...`.
    *   **Executes** a tool via `run_shell_command("scripts/tool ...")`.

### 3. External Attention (InfiAgent Pattern)
To handle massive context (e.g., reading 100 PDFs) without bloating the prompt:
- **Do NOT** load the entire file with `read_file`.
- **Use `fs_query`** (or `ask_file`): This tool spawns a temporary, isolated sub-agent that reads the file and returns *only* the answer to your specific question.
- This keeps the main agent's context window small ($O(1)$) while accessing infinite data.

### 4. "Tools" are just Scripts
Agents don't need special API definitions. They just need to know which scripts exist in `lab/afs/tools/` or the project `scripts/` folder.

**Standard Toolset:**
- `ctx mount <path>`: Bring a file into focus.
- `ws find <query>`: Search the workspace.
- `rg <pattern>`: Grep code.
- `write_file <path>`: Create/Edit content.
- `fs_query <path> <query>`: Ask a question about a file (External Attention).

## Training Strategy

To train models (like `farore` or `scawful-echo`) to use this, we need examples where the "Correct Answer" isn't just text, but a **sequence of file operations**.

**Example Training Sequence:**
1.  **User:** "Fix the bug in the player physics."
2.  **Model (Thought):** "I need to find the physics code first."
3.  **Model (Action):** `run_shell_command("ws find physics")`
4.  **System:** (Returns file list)
5.  **Model (Action):** `read_file("src/physics.asm")`
6.  **Model (Action):** `write_file("src/physics.asm", ...)`

We will generate synthetic datasets (`training/scripts/generate_afs_corpus.py`) that reinforce this **Search -> Read -> Plan -> Write** loop.
