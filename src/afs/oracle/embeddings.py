"""Oracle of Secrets Embedding Generator.

Generates embeddings for:
1. Assembly (.asm) files and routines
2. Symbol table entries
3. Documentation
4. Comments and code patterns

Uses the AFS embedding infrastructure for vector storage.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

logger = logging.getLogger(__name__)


# Paths
ORACLE_PROJECT = Path.home() / "src" / "hobby" / "oracle-of-secrets"
KNOWLEDGE_BASE = Path.home() / ".context" / "knowledge" / "oracle-of-secrets"
EMBEDDINGS_DIR = KNOWLEDGE_BASE / "embeddings"

# ALTTP symbols file
SYMBOLS_FILE = Path.home() / ".context" / "knowledge" / "alttp" / "symbols.json"


@dataclass
class EmbeddingChunk:
    """A chunk of content to embed."""
    id: str
    source_type: str  # asm, symbol, doc, comment
    source_path: str
    content: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "content": self.content[:500],  # Truncate for index
            "metadata": self.metadata,
        }


@dataclass
class OracleEmbeddingStats:
    """Statistics from embedding generation."""
    total_chunks: int = 0
    asm_files: int = 0
    asm_routines: int = 0
    symbols: int = 0
    docs: int = 0
    comments: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Total: {self.total_chunks} chunks\n"
            f"  ASM files: {self.asm_files}\n"
            f"  ASM routines: {self.asm_routines}\n"
            f"  Symbols: {self.symbols}\n"
            f"  Docs: {self.docs}\n"
            f"  Comments: {self.comments}\n"
            f"  Errors: {len(self.errors)}"
        )


class OracleEmbeddingGenerator:
    """Generates embeddings for Oracle of Secrets content."""

    # Patterns for extracting code structure
    ROUTINE_PATTERN = re.compile(
        r'^([A-Za-z_][A-Za-z0-9_]*):\s*$',
        re.MULTILINE
    )

    NAMESPACE_PATTERN = re.compile(
        r'namespace\s+(\w+)',
        re.IGNORECASE
    )

    MACRO_PATTERN = re.compile(
        r'^macro\s+(\w+)',
        re.MULTILINE | re.IGNORECASE
    )

    def __init__(
        self,
        oracle_path: Path | None = None,
        output_dir: Path | None = None,
        embed_fn: Callable[[str], list[float]] | None = None,
    ):
        """Initialize the generator.

        Args:
            oracle_path: Path to Oracle project
            output_dir: Output directory for embeddings
            embed_fn: Function to generate embeddings
        """
        self.oracle_path = oracle_path or ORACLE_PROJECT
        self.output_dir = output_dir or EMBEDDINGS_DIR
        self.embed_fn = embed_fn
        self.stats = OracleEmbeddingStats()

    def extract_asm_chunks(self, file_path: Path) -> Iterator[EmbeddingChunk]:
        """Extract embeddable chunks from an ASM file.

        Args:
            file_path: Path to .asm file

        Yields:
            EmbeddingChunk for each routine/section
        """
        try:
            content = file_path.read_text(errors="replace")
        except Exception as e:
            self.stats.errors.append(f"{file_path}: {e}")
            return

        rel_path = str(file_path.relative_to(self.oracle_path))
        self.stats.asm_files += 1

        # Find namespace
        namespace_match = self.NAMESPACE_PATTERN.search(content)
        namespace = namespace_match.group(1) if namespace_match else ""

        # Extract file-level chunk
        file_id = f"asm:{rel_path}"
        yield EmbeddingChunk(
            id=file_id,
            source_type="asm_file",
            source_path=rel_path,
            content=content[:3000],  # First 3000 chars for context
            metadata={
                "namespace": namespace,
                "line_count": content.count("\n"),
            },
        )

        # Extract routines
        lines = content.split("\n")
        routine_starts = []

        for i, line in enumerate(lines):
            match = self.ROUTINE_PATTERN.match(line)
            if match:
                routine_name = match.group(1)
                routine_starts.append((i, routine_name))

        # Create chunks for each routine
        for idx, (start_line, name) in enumerate(routine_starts):
            # Find end (next routine or EOF)
            if idx + 1 < len(routine_starts):
                end_line = routine_starts[idx + 1][0]
            else:
                end_line = len(lines)

            routine_lines = lines[start_line:end_line]
            routine_content = "\n".join(routine_lines)

            # Skip very short routines
            if len(routine_content) < 20:
                continue

            self.stats.asm_routines += 1

            routine_id = f"asm:{rel_path}:{name}"
            yield EmbeddingChunk(
                id=routine_id,
                source_type="asm_routine",
                source_path=f"{rel_path}:{start_line + 1}",
                content=routine_content[:2000],
                metadata={
                    "routine_name": name,
                    "namespace": namespace,
                    "start_line": start_line + 1,
                    "end_line": end_line,
                },
            )

        # Extract macros
        for match in self.MACRO_PATTERN.finditer(content):
            macro_name = match.group(1)
            start_pos = match.start()
            # Find macro content (up to endmacro)
            end_pos = content.find("endmacro", start_pos)
            if end_pos > start_pos:
                macro_content = content[start_pos:end_pos + 8]
                macro_id = f"macro:{rel_path}:{macro_name}"
                yield EmbeddingChunk(
                    id=macro_id,
                    source_type="macro",
                    source_path=rel_path,
                    content=macro_content[:1500],
                    metadata={
                        "macro_name": macro_name,
                    },
                )

    def extract_symbol_chunks(self, symbols_path: Path | None = None) -> Iterator[EmbeddingChunk]:
        """Extract embeddable chunks from symbol table.

        Args:
            symbols_path: Path to symbols.json

        Yields:
            EmbeddingChunk for each symbol
        """
        path = symbols_path or SYMBOLS_FILE
        if not path.exists():
            logger.warning(f"Symbols file not found: {path}")
            return

        try:
            with open(path) as f:
                symbols = json.load(f)
        except Exception as e:
            self.stats.errors.append(f"{path}: {e}")
            return

        for symbol in symbols:
            name = symbol.get("name", "")
            address = symbol.get("address", "")
            comment = symbol.get("comment", "")
            category = symbol.get("category", "unknown")

            if not name:
                continue

            self.stats.symbols += 1

            # Create rich content for embedding
            content = f"Symbol: {name}\nAddress: {address}"
            if comment:
                content += f"\nDescription: {comment}"
            if category:
                content += f"\nCategory: {category}"

            symbol_id = f"symbol:{name}"
            yield EmbeddingChunk(
                id=symbol_id,
                source_type="symbol",
                source_path="symbols.json",
                content=content,
                metadata={
                    "name": name,
                    "address": address,
                    "category": category,
                },
            )

    def extract_doc_chunks(self) -> Iterator[EmbeddingChunk]:
        """Extract embeddable chunks from documentation.

        Yields:
            EmbeddingChunk for each doc section
        """
        docs_path = self.oracle_path / "Docs"
        if not docs_path.exists():
            return

        for doc_file in docs_path.rglob("*.md"):
            try:
                content = doc_file.read_text(errors="replace")
            except Exception as e:
                self.stats.errors.append(f"{doc_file}: {e}")
                continue

            rel_path = str(doc_file.relative_to(self.oracle_path))
            self.stats.docs += 1

            # Split by headers for better chunks
            sections = re.split(r'^(#{1,3}\s+.+)$', content, flags=re.MULTILINE)

            current_header = doc_file.stem
            for i, section in enumerate(sections):
                if section.startswith("#"):
                    current_header = section.strip("# \n")
                    continue

                section = section.strip()
                if len(section) < 50:
                    continue

                section_id = f"doc:{rel_path}:{current_header}"
                yield EmbeddingChunk(
                    id=section_id,
                    source_type="doc",
                    source_path=rel_path,
                    content=f"# {current_header}\n\n{section[:2000]}",
                    metadata={
                        "header": current_header,
                        "file": doc_file.name,
                    },
                )

    def extract_comment_chunks(self) -> Iterator[EmbeddingChunk]:
        """Extract embeddable chunks from code comments.

        Yields:
            EmbeddingChunk for significant comment blocks
        """
        comment_pattern = re.compile(r'^;[;=\-]+\n((?:;.*\n)+)', re.MULTILINE)

        for asm_file in self.oracle_path.rglob("*.asm"):
            try:
                content = asm_file.read_text(errors="replace")
            except Exception:
                continue

            rel_path = str(asm_file.relative_to(self.oracle_path))

            for match in comment_pattern.finditer(content):
                comment_block = match.group(0)
                # Clean up comment markers
                clean_comment = "\n".join(
                    line.lstrip("; ") for line in comment_block.split("\n")
                    if line.startswith(";")
                )

                if len(clean_comment) < 100:
                    continue

                self.stats.comments += 1

                # Generate ID from content hash
                comment_hash = hash(clean_comment) & 0xFFFFFFFF
                comment_id = f"comment:{rel_path}:{comment_hash:08x}"

                yield EmbeddingChunk(
                    id=comment_id,
                    source_type="comment",
                    source_path=rel_path,
                    content=clean_comment[:1500],
                    metadata={
                        "line": content[:match.start()].count("\n") + 1,
                    },
                )

    def generate_all(self, include_symbols: bool = True) -> OracleEmbeddingStats:
        """Generate all embeddings for Oracle project.

        Args:
            include_symbols: Include ALTTP symbols

        Returns:
            Statistics from generation
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        embeddings_dir = self.output_dir / "vectors"
        embeddings_dir.mkdir(exist_ok=True)

        index = []
        chunks_processed = 0

        # Process ASM files
        logger.info("Processing ASM files...")
        for asm_file in self.oracle_path.rglob("*.asm"):
            for chunk in self.extract_asm_chunks(asm_file):
                index.append(chunk.to_dict())
                chunks_processed += 1
                self._save_embedding(chunk, embeddings_dir)

        # Process symbols
        if include_symbols:
            logger.info("Processing symbols...")
            for chunk in self.extract_symbol_chunks():
                index.append(chunk.to_dict())
                chunks_processed += 1
                self._save_embedding(chunk, embeddings_dir)

        # Process docs
        logger.info("Processing documentation...")
        for chunk in self.extract_doc_chunks():
            index.append(chunk.to_dict())
            chunks_processed += 1
            self._save_embedding(chunk, embeddings_dir)

        # Process comments
        logger.info("Processing comments...")
        for chunk in self.extract_comment_chunks():
            index.append(chunk.to_dict())
            chunks_processed += 1
            self._save_embedding(chunk, embeddings_dir)

        self.stats.total_chunks = chunks_processed

        # Save index
        index_path = self.output_dir / "oracle_embedding_index.json"
        with open(index_path, "w") as f:
            json.dump({
                "version": "1.0",
                "project": "oracle-of-secrets",
                "chunks": index,
                "stats": {
                    "total": self.stats.total_chunks,
                    "asm_files": self.stats.asm_files,
                    "asm_routines": self.stats.asm_routines,
                    "symbols": self.stats.symbols,
                    "docs": self.stats.docs,
                    "comments": self.stats.comments,
                },
            }, f, indent=2)

        logger.info(f"Saved index to {index_path}")
        return self.stats

    def _save_embedding(self, chunk: EmbeddingChunk, output_dir: Path) -> None:
        """Save embedding vector for a chunk.

        Args:
            chunk: The chunk to embed
            output_dir: Directory to save vector
        """
        if self.embed_fn is None:
            return

        try:
            vector = self.embed_fn(chunk.content)
            if vector:
                # Save as JSON (could also use numpy/faiss format)
                safe_id = chunk.id.replace("/", "_").replace(":", "_")
                vector_path = output_dir / f"{safe_id}.json"
                with open(vector_path, "w") as f:
                    json.dump({
                        "id": chunk.id,
                        "vector": vector,
                    }, f)
        except Exception as e:
            self.stats.errors.append(f"Embedding {chunk.id}: {e}")


def create_embedding_fn(backend: str = "ollama") -> Callable[[str], list[float]] | None:
    """Create embedding function based on backend.

    Args:
        backend: "ollama", "hf", or "none"

    Returns:
        Embedding function or None
    """
    if backend == "none":
        return None

    if backend == "ollama":
        from ..embeddings import create_ollama_embed_fn
        return create_ollama_embed_fn()

    if backend == "hf":
        from ..embeddings import create_hf_embed_fn
        return create_hf_embed_fn("sentence-transformers/all-MiniLM-L6-v2")

    return None


async def main():
    """Generate Oracle embeddings."""
    import argparse

    parser = argparse.ArgumentParser(description="Oracle Embedding Generator")
    parser.add_argument("--output", "-o", type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--embed", choices=["ollama", "hf", "none"], default="none")
    parser.add_argument("--no-symbols", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    embed_fn = create_embedding_fn(args.embed)

    generator = OracleEmbeddingGenerator(
        output_dir=args.output,
        embed_fn=embed_fn,
    )

    stats = generator.generate_all(include_symbols=not args.no_symbols)

    print("\nEmbedding Generation Complete")
    print("=" * 40)
    print(stats.summary())

    if stats.errors:
        print(f"\nErrors ({len(stats.errors)}):")
        for err in stats.errors[:10]:
            print(f"  - {err}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
