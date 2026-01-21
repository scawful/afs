#!/usr/bin/env python3
"""
Query File Tool (External Attention)

This script implements the "External Attention" pattern from the InfiAgent paper.
It spins up a temporary, isolated agent context to answer a specific question
about a file, without loading the entire file into the main agent's context.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from afs.agent.harness import run_agent

async def main():
    parser = argparse.ArgumentParser(description="Answer a question about a file.")
    parser.add_argument("path", help="Path to the file to query")
    parser.add_argument("query", help="The question to ask about the file")
    parser.add_argument("--model", default="gemini-3-flash-preview", help="Model to use")
    
    args = parser.parse_args()
    
    file_path = Path(args.path).expanduser()
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
        
    try:
        content = file_path.read_text()
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Construct a focused prompt
    prompt = f"""
I will provide the content of a file named '{file_path.name}'.
Please answer the following question based ONLY on this content.

Question: {args.query}

--- BEGIN FILE CONTENT ---
{content}
--- END FILE CONTENT ---

Answer:
"""
    
    # Run the ephemeral agent
    # We pass tools=[] because this "reader" agent shouldn't need to use tools,
    # it just needs to read and answer.
    try:
        result = await run_agent(
            model=args.model,
            prompt=prompt,
            tools=[], 
            verbose=False
        )
        
        if result.success:
            print(result.response)
        else:
            print(f"Error processing query: {result.error}", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"Error running agent: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
