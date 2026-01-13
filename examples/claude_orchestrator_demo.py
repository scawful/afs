#!/usr/bin/env python3
"""Demo script for Claude Agent SDK orchestrator with local expert models.

This demonstrates the integration pattern:
1. Claude acts as the orchestrator (reasoning, planning, synthesis)
2. Local Ollama models provide domain expertise (65816 ASM, SNES hardware)
3. Claude delegates to experts and synthesizes responses

Usage:
    # Single query
    python examples/claude_orchestrator_demo.py --prompt "Explain how Mode 7 works"

    # Interactive mode
    python examples/claude_orchestrator_demo.py --interactive

    # Test expert models directly (no Claude API needed)
    python examples/claude_orchestrator_demo.py --test-experts

Requirements:
    - claude-agent-sdk: pip install claude-agent-sdk
    - Ollama with trained models: nayru-v7, veran-v4, farore-v5, majora-v2
    - ANTHROPIC_API_KEY environment variable
"""

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


EXPERT_MODELS = {
    "nayru": {
        "tag": "nayru-v7:latest",
        "description": "65816 ASM generation and explanation",
        "test_prompt": "What does LDA $7E0010 do?",
    },
    "veran": {
        "tag": "veran-v4:latest",
        "description": "SNES hardware (PPU, APU, DMA, Mode 7)",
        "test_prompt": "What is register $2100?",
    },
    "farore": {
        "tag": "farore-v5:latest",
        "description": "Debugging and diagnosis",
        "test_prompt": "How do I debug a stack overflow?",
    },
    "majora": {
        "tag": "majora-v2:latest",
        "description": "Oracle of Secrets ROM hack (Time System, Mask System, custom sprites)",
        "test_prompt": "Explain the Time System implementation in Oracle of Secrets",
    },
}


def query_ollama(model_tag: str, prompt: str, timeout: int = 120) -> str:
    """Query a local Ollama model."""
    try:
        result = subprocess.run(
            ["ollama", "run", model_tag, prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return f"Error: {result.stderr}"
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "Error: Query timed out (try increasing timeout)"
    except FileNotFoundError:
        return "Error: Ollama not found - install from https://ollama.ai"
    except Exception as e:
        return f"Error: {e}"


def test_experts():
    """Test each expert model directly without Claude API."""
    print("Testing expert models directly (no Claude API needed)\n")
    print("=" * 60)

    for name, config in EXPERT_MODELS.items():
        print(f"\n[{name.upper()}] {config['description']}")
        print(f"Model: {config['tag']}")
        print(f"Test: {config['test_prompt']}")
        print("-" * 40)

        response = query_ollama(config["tag"], config["test_prompt"])
        # Truncate long responses
        if len(response) > 500:
            response = response[:500] + "\n... (truncated)"
        print(response)
        print()


async def run_orchestrator(prompt: str):
    """Run the Claude orchestrator with expert tools."""
    try:
        from claude_agent_sdk import query, ClaudeAgentOptions
        from claude_agent_sdk.tools import tool
    except ImportError:
        print("Error: claude-agent-sdk not installed")
        print("Run: pip install claude-agent-sdk")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set")
        return

    # Define tools for expert delegation
    @tool
    def ask_nayru(question: str) -> str:
        """Ask Nayru for 65816 ASM code generation or explanation."""
        print(f"  → Delegating to Nayru...")
        return query_ollama(EXPERT_MODELS["nayru"]["tag"], question)

    @tool
    def ask_veran(question: str) -> str:
        """Ask Veran about SNES hardware (PPU, APU, DMA, Mode 7)."""
        print(f"  → Delegating to Veran...")
        return query_ollama(EXPERT_MODELS["veran"]["tag"], question)

    @tool
    def ask_farore(question: str) -> str:
        """Ask Farore to debug or diagnose code issues."""
        print(f"  → Delegating to Farore...")
        return query_ollama(EXPERT_MODELS["farore"]["tag"], question)

    @tool
    def ask_majora(question: str) -> str:
        """Ask Majora about ALTTP/Zelda-specific topics."""
        print(f"  → Delegating to Majora...")
        return query_ollama(EXPERT_MODELS["majora"]["tag"], question)

    system_prompt = """You are an orchestrator for SNES/65816 assembly development.

You have access to specialized local expert models:
- Nayru: 65816 ASM code generation and explanation
- Veran: SNES hardware (PPU, APU, DMA, Mode 7)
- Farore: Debugging and diagnosis
- Majora: General ALTTP/Zelda knowledge

When answering questions:
1. Determine which expert(s) would best answer the question
2. Delegate to the appropriate expert(s)
3. Synthesize their responses into a coherent answer
4. Add your own reasoning and context

For complex questions, consult multiple experts and combine their knowledge."""

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        tools=[ask_nayru, ask_veran, ask_farore, ask_majora],
    )

    print(f"\nPrompt: {prompt}")
    print("-" * 40)

    try:
        async for message in query(prompt=prompt, options=options):
            if hasattr(message, 'content'):
                for block in message.content:
                    if hasattr(block, 'text'):
                        print(block.text)
                    elif hasattr(block, 'type') and block.type == 'tool_use':
                        print(f"\n[Tool: {block.name}]")
    except Exception as e:
        print(f"Error: {e}")


async def interactive_mode():
    """Run interactive session."""
    print("Claude Orchestrator - Interactive Mode")
    print("Expert models: nayru, veran, farore, majora")
    print("Type 'quit' to exit, 'test' to test experts directly\n")

    while True:
        try:
            prompt = input("You: ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("quit", "exit"):
                break
            if prompt.lower() == "test":
                test_experts()
                continue

            await run_orchestrator(prompt)
            print()

        except KeyboardInterrupt:
            break
        except EOFError:
            break

    print("Goodbye!")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", help="Single query to run")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--test-experts", action="store_true", help="Test expert models directly")
    args = parser.parse_args()

    if args.test_experts:
        test_experts()
    elif args.interactive:
        asyncio.run(interactive_mode())
    elif args.prompt:
        asyncio.run(run_orchestrator(args.prompt))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
