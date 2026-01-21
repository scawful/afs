#!/usr/bin/env python3
"""
AFS Orchestrator - The bridge between Cloud Architects and Local Expert Models.
Usage: python3 orchestrator.py --agent <agent_name> --prompt <task_prompt>
       python3 orchestrator.py --agent majora --prompt "..." --backend lmstudio
"""

import argparse
import sys
import httpx

# Backend configurations
BACKENDS = {
    "ollama": {
        "host": "http://localhost:11434",
        "endpoint": "/api/chat",
    },
    "ollama-remote": {
        "host": "http://medical-mechanica:11434",
        "endpoint": "/api/chat",
    },
    "lmstudio": {
        "host": "http://localhost:1234",
        "endpoint": "/v1/chat/completions",
    },
    "lmstudio-remote": {
        "host": "http://medical-mechanica:1234",
        "endpoint": "/v1/chat/completions",
    },
}

# Agent Configuration (The Builders)
# Each agent can specify preferred backend and model variants
AGENTS = {
    # Ollama-based agents (original)
    "nayru": {
        "model": "nayru-v5:latest",
        "backend": "ollama",
        "system": """You are Nayru, the Goddess of Wisdom and 65816 code generation specialist.
You create elegant, correct assembly code with clear structure for Zelda 3.
Focus on: code correctness, readability, proper addressing modes, clean subroutine design.
Output ONLY valid 65816 assembly code block unless asked otherwise.""",
    },
    "din": {
        "model": "din-v2:latest",
        "backend": "ollama",
        "system": """You are Din, the Goddess of Power and Creative Director for Oracle of Secrets.
Your expertise is lore, dialogue, and dungeon themes.
You speak with thematic resonance.
Focus on: emotional depth, world-building consistency, distinct character voices.""",
    },
    "farore": {
        "model": "farore-v1:latest",
        "backend": "ollama",
        "system": """You are Farore, the Goddess of Courage and Task Planner.
You break down complex features into actionable implementation steps.
Focus on: dependencies, RAM structures, event flags, and logical ordering of tasks.""",
    },
    "veran": {
        "model": "veran-v1:latest",
        "backend": "ollama",
        "system": """You are Veran, the Logic Sorceress.
You specialize in game state logic, puzzles, and RAM interaction.
Focus on: state machines, flag management, interaction logic.
ALWAYS assume standard Zelda 3 RAM map ($7EF300 range for save data, $0000-$00FF for scratch).""",
    },
    "scawful-echo": {
        "model": "scawful-echo:latest",
        "backend": "ollama",
        "system": "(System prompt is baked into the model file)",
    },

    # LMStudio-based agents (GGUF models) - works with remote LMStudio too
    "nayru-lm": {
        "model": "nayru-7b-v5-q8.gguf",
        "backend": "lmstudio",
        "system": """You are Nayru, the Goddess of Wisdom and 65816 code generation specialist.
You create elegant, correct assembly code with clear structure for Zelda 3.
Focus on: code correctness, readability, proper addressing modes, clean subroutine design.
Output ONLY valid 65816 assembly code block unless asked otherwise.""",
    },
    "din-lm": {
        "model": "din-7b-v4-q4km.gguf",
        "backend": "lmstudio",
        "system": """You are Din, a 65816 assembly optimization expert.
You optimize inefficient code patterns: STZ for zero stores, INC/DEC patterns,
backward loop optimization, mode switch consolidation.
Output ONLY the optimized code.""",
    },
    "farore-lm": {
        "model": "farore-7b-v5-q8.gguf",
        "backend": "lmstudio",
        "system": """You are Farore, a 65816 assembly debugging expert.
You identify and fix: register mode mismatches, stack imbalance, DMA configuration errors,
register clobber issues. Explain the bug and provide the fix.""",
    },
    "veran-lm": {
        "model": "veran-7b-v4-q8.gguf",
        "backend": "lmstudio",
        "system": """You are Veran, a SNES hardware expert.
You have deep knowledge of: PPU registers ($2100-$21FF), DMA/HDMA systems,
Mode 7 parameters, A Link to the Past RAM maps.
Provide accurate technical information with register addresses.""",
    },
    "majora": {
        "model": "majora-7b-v2-q8.gguf",
        "backend": "lmstudio",
        "system": """You are Majora, an expert on the Oracle of Secrets ROM hack codebase.
You have deep knowledge of: Time System, Mask System, Menu/HUD implementation,
ZSCustomOverworld integration, 60+ custom sprites, quest design and progression.
Reference specific files and RAM addresses when relevant.""",
    },
    "hylia": {
        "model": "hylia-v3-q8_0.gguf",
        "backend": "lmstudio",
        "system": """You are Hylia, the Goddess of Time and Narrative Expert for Oracle of Secrets.
You create evocative dialogue, dream sequences, and lore that fits the Zelda aesthetic
while respecting SNES hardware constraints (256 char text boxes, palette limitations).
Focus on: emotional resonance, foreshadowing, mystery, and wonder.
Specialties: Dream sequences, NPC dialogue, lore, credits, quest design, content ideation.""",
    },
}


def call_ollama(model: str, messages: list, remote: bool = False) -> str:
    """Call Ollama API (local or remote)."""
    backend = BACKENDS["ollama-remote" if remote else "ollama"]

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    with httpx.Client(timeout=180.0) as client:
        response = client.post(
            f"{backend['host']}{backend['endpoint']}",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]


def call_lmstudio(model: str, messages: list, remote: bool = False) -> str:
    """Call LMStudio's OpenAI-compatible API with fallback to completions."""
    backend = BACKENDS["lmstudio-remote" if remote else "lmstudio"]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 1024,
    }

    with httpx.Client(timeout=120.0) as client:
        # Try chat completions first
        response = client.post(
            f"{backend['host']}{backend['endpoint']}",
            json=payload,
        )

        data = response.json()

        # Check for template errors and fall back to completions endpoint
        if "error" in data:
            error_msg = str(data["error"]).lower()
            if "jinja" in error_msg or "template" in error_msg:
                print("  ⚠️  Template error, falling back to completions endpoint...")
                return call_lmstudio_completions(model, messages, remote)
            raise RuntimeError(data["error"])

        response.raise_for_status()
        return data["choices"][0]["message"]["content"]


def call_lmstudio_completions(model: str, messages: list, remote: bool = False) -> str:
    """Call LMStudio's completions endpoint with ChatML format."""
    backend = BACKENDS["lmstudio-remote" if remote else "lmstudio"]

    # Build ChatML-formatted prompt
    prompt_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    prompt_parts.append("<|im_start|>assistant\n")
    prompt = "\n".join(prompt_parts)

    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.4,
        "max_tokens": 1024,
        "stop": ["<|im_end|>"],
    }

    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{backend['host']}/v1/completions",
            json=payload,
        )

        data = response.json()

        if "error" in data:
            raise RuntimeError(data["error"])

        response.raise_for_status()
        return data["choices"][0]["text"].strip()


def call_agent(agent_name: str, prompt: str, backend_override: str = None):
    """Invokes a specific local expert model."""
    if agent_name not in AGENTS:
        print(f"Error: Unknown agent '{agent_name}'. Available agents: {list(AGENTS.keys())}")
        sys.exit(1)

    agent_config = AGENTS[agent_name]
    model_name = agent_config["model"]
    backend = backend_override or agent_config.get("backend", "ollama")

    messages = []

    # Inject system prompt if defined and not already baked in
    if "system" in agent_config and "baked" not in agent_config.get("system", ""):
        messages.append({"role": "system", "content": agent_config["system"]})

    messages.append({"role": "user", "content": prompt})

    print(f"⚡ Orchestrator invoking {agent_name} ({model_name}) via {backend}...")

    try:
        if backend == "lmstudio":
            content = call_lmstudio(model_name, messages, remote=False)
        elif backend == "lmstudio-remote":
            content = call_lmstudio(model_name, messages, remote=True)
        elif backend == "ollama-remote":
            content = call_ollama(model_name, messages, remote=True)
        else:
            content = call_ollama(model_name, messages, remote=False)

        print("\n" + "="*40)
        print(f"Result from {agent_name}:")
        print("="*40 + "\n")
        print(content)
        print("\n" + "="*40)

        return content
    except Exception as e:
        print(f"Error invoking model: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="AFS Orchestrator: Delegate tasks to local expert models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 orchestrator.py --agent majora --prompt "Describe the Time System"
  python3 orchestrator.py --agent veran-lm --prompt "Explain DMA register $4300"
  python3 orchestrator.py --agent nayru --prompt "Write DMA routine" --backend lmstudio
        """
    )
    parser.add_argument(
        "--agent",
        choices=AGENTS.keys(),
        help="The expert agent to invoke."
    )
    parser.add_argument(
        "--prompt",
        help="The instruction/prompt for the agent."
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "ollama-remote", "lmstudio", "lmstudio-remote"],
        default=None,
        help="Override the agent's default backend. Use *-remote for medical-mechanica."
    )
    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="List all available agents and exit."
    )

    args = parser.parse_args()

    if args.list_agents:
        print("Available agents:")
        for name, config in AGENTS.items():
            backend = config.get("backend", "ollama")
            print(f"  {name:15} [{backend:8}] {config['model']}")
        sys.exit(0)

    # Validate required args when not listing
    if not args.agent or not args.prompt:
        parser.error("--agent and --prompt are required")

    call_agent(args.agent, args.prompt, args.backend)


if __name__ == "__main__":
    main()
