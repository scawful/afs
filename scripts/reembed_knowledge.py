#!/usr/bin/env python3
"""Re-embed knowledge base documents with embeddinggemma."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def embed_text(text: str, model: str, host: str, client: httpx.Client) -> list[float]:
    """Embed text using Ollama."""
    response = client.post(
        f"{host}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def process_embedding_file(
    file_path: Path,
    model: str,
    host: str,
    client: httpx.Client,
    dry_run: bool = False,
) -> bool:
    """Re-embed a single document."""
    try:
        with open(file_path) as f:
            doc = json.load(f)

        text = doc.get("text_preview", "")
        if not text:
            logger.warning(f"No text_preview in {file_path.name}")
            return False

        # Get new embedding
        new_embedding = embed_text(text, model, host, client)

        if dry_run:
            logger.info(f"Would update {file_path.name} ({len(new_embedding)} dims)")
            return True

        # Update document
        doc["embedding"] = new_embedding
        doc["embedding_provider"] = "ollama"
        doc["embedding_model"] = model
        doc["updated"] = datetime.now().isoformat()

        with open(file_path, "w") as f:
            json.dump(doc, f)

        return True

    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Re-embed knowledge base documents")
    parser.add_argument(
        "--kb-path",
        type=Path,
        default=Path.home() / ".context" / "knowledge" / "alttp",
        help="Knowledge base path",
    )
    parser.add_argument(
        "--model",
        default="embeddinggemma:latest",
        help="Embedding model",
    )
    parser.add_argument(
        "--host",
        default="http://localhost:11435",
        help="Ollama host",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write changes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of files (0 = all)",
    )
    args = parser.parse_args()

    embeddings_dir = args.kb_path / "embeddings"
    if not embeddings_dir.exists():
        logger.error(f"Embeddings directory not found: {embeddings_dir}")
        return 1

    # Get list of embedding files
    files = list(embeddings_dir.glob("*.json"))
    if args.limit > 0:
        files = files[: args.limit]

    logger.info(f"Found {len(files)} embedding files in {args.kb_path.name}")
    logger.info(f"Using model: {args.model} at {args.host}")

    if args.dry_run:
        logger.info("DRY RUN - no files will be modified")

    success = 0
    failed = 0

    with httpx.Client() as client:
        # Process files with progress
        for i, file_path in enumerate(files):
            if process_embedding_file(
                file_path, args.model, args.host, client, args.dry_run
            ):
                success += 1
            else:
                failed += 1

            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i + 1}/{len(files)} ({success} ok, {failed} failed)")

    logger.info(f"Complete: {success} succeeded, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
