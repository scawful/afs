"""Researcher Agent (Level 2).

Implements the 'Literature Review' agent from the InfiAgent paper.
Iterates over a set of documents, processing each one with an ephemeral
'External Attention' agent, and compiling the results into a persistent
state file (report).

This demonstrates:
1. File-Centric State: Progress is saved to a file after each step.
2. Bounded Context: The main loop does not hold the content of all papers in memory.
3. External Attention: Each paper is processed by a separate agent instance.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List

from afs.agent.harness import run_agent

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("researcher")

async def analyze_document(file_path: Path, query: str, model: str) -> str:
    """Analyze a single document using an ephemeral agent (External Attention)."""
    try:
        content = file_path.read_text()
    except Exception as e:
        return f"Error reading file: {e}"

    # Construct a focused prompt
    # We do not use tools here; just pure reading.
    prompt = f"""
I will provide the content of a file named '{file_path.name}'.
Please answer the following question based ONLY on this content.
Be concise.

Question: {query}

--- BEGIN FILE CONTENT ---
{content[:50000]} 
--- END FILE CONTENT ---
(Content truncated to 50k chars if longer)

Answer:
"""
    
    result = await run_agent(
        model=model,
        prompt=prompt,
        tools=[], # No tools needed for pure analysis
        verbose=False
    )
    
    if result.success:
        return result.response
    else:
        return f"Analysis failed: {result.error}"

async def run_research_task(
    input_dir: Path,
    output_file: Path,
    query: str,
    glob_pattern: str = "*.pdf",
    model: str = "gemini-3-flash-preview"
):
    """Run the research loop."""
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    # 1. Initialize State
    files = sorted(list(input_dir.glob(glob_pattern)))
    logger.info(f"Found {len(files)} files to process in {input_dir}")
    
    # Check if we are resuming
    processed_files = set()
    if output_file.exists():
        logger.info(f"Resuming from existing report: {output_file}")
        existing_content = output_file.read_text()
        # Simple heuristic to find processed files
        for f in files:
            if f"## Analysis of {f.name}" in existing_content:
                processed_files.add(f.name)
    else:
        output_file.write_text(f"# Research Report\n\nQuery: {query}\n\n", encoding="utf-8")

    # 2. Iterate (Infinite Horizon Loop)
    for i, file_path in enumerate(files):
        if file_path.name in processed_files:
            logger.info(f"Skipping already processed: {file_path.name}")
            continue
            
        logger.info(f"Processing [{i+1}/{len(files)}]: {file_path.name}")
        
        # 3. External Attention Step
        # This spawns a separate context. The main loop's context is effectively just 'i' and 'file_path'.
        # We manually bridge the result to the persistent state.
        analysis = await analyze_document(file_path, query, model)
        
        # 4. State Consolidation
        # Write immediately to disk. This is the "File-Centric State".
        # Even if the script crashes, progress is saved.
        report_entry = f"\n## Analysis of {file_path.name}\n\n{analysis}\n\n---"
        
        with output_file.open("a", encoding="utf-8") as f:
            f.write(report_entry)
            
        logger.info(f"Saved analysis for {file_path.name}")

    logger.info("Research task complete.")

async def main():
    parser = argparse.ArgumentParser(description="InfiAgent Researcher (Level 2)")
    parser.add_argument("input_dir", help="Directory containing documents to research")
    parser.add_argument("output_file", help="Path to the markdown report file")
    parser.add_argument("--query", default="Summarize the key findings.", help="Research query")
    parser.add_argument("--glob", default="*.txt", help="File pattern (e.g. *.md, *.txt)")
    parser.add_argument("--model", default="gemini-3-flash-preview", help="Model to use")
    
    args = parser.parse_args()
    
    await run_research_task(
        Path(args.input_dir),
        Path(args.output_file),
        args.query,
        args.glob,
        args.model
    )

if __name__ == "__main__":
    asyncio.run(main())
