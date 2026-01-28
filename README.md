# Agentic File System (AFS)

**Everything is Context.**

AFS is an experimental orchestration layer for managing multi-agent swarms and context loops directly within the filesystem. It treats documentation, tools, and memory as mountable context nodes, allowing AI agents to "live" and operate within the repository.

## Core Concepts

- **Context Mounting:** Dynamically load relevant documentation, code, and memories into the agent's working context.
- **Agent Swarms:** Orchestrate multiple specialized agents (Planner, Coder, Critic) to solve complex tasks.
- **File-Based Memory:** Store long-term memories and project knowledge in a structured, human-readable file system.
- **Tool Integration:** Expose command-line tools and scripts as callable functions for agents.

## Architecture

AFS is built on a modular architecture:

- **Core:** The central engine that manages context, agents, and tool execution.
- **Services:** Pluggable modules for specific functionalities (e.g., LLM providers, vector databases).
- **Tools:** Scripts and executables that agents can invoke to perform actions.
- **Context:** The structured file system where agents operate and store information.

## Getting Started

1.  **Installation:**
    ```bash
    git clone https://github.com/scawful/afs.git
    cd afs
    pip install -r requirements.txt
    ```

2.  **Configuration:**
    Configure your LLM providers and other settings in `config/`.

3.  **Running Agents:**
    Use the CLI tools to interact with agents and manage context.

## Documentation

See the `docs/` directory for detailed guides and architectural overviews.

## License

MIT
