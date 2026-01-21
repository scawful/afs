"""Corpus builder for pre-training data collection.

Collects and processes large corpora from:
- ALTTP disassembly (65816 assembly)
- Personal conversations and notes (Avatar models)
"""

import logging
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CorpusConfig:
    """Configuration for corpus building."""
    source_dirs: list[Path] = field(default_factory=list)
    file_patterns: list[str] = field(default_factory=lambda: ["*.asm", "*.s", "*.inc"])
    min_line_length: int = 2
    max_line_length: int = 500
    skip_patterns: list[str] = field(default_factory=list)
    chunk_size: int = 512  # tokens per chunk


@dataclass
class CorpusChunk:
    """A chunk of corpus text for training."""
    text: str
    source_file: str
    line_start: int
    line_end: int
    metadata: dict = field(default_factory=dict)


class CorpusBuilder:
    """Builds training corpora from source files."""

    def __init__(self, config: CorpusConfig | None = None):
        self.config = config or CorpusConfig()
        self._skip_regex = [re.compile(p) for p in self.config.skip_patterns]

    def collect_files(self) -> Iterator[Path]:
        """Collect all matching files from source directories."""
        for source_dir in self.config.source_dirs:
            if not source_dir.exists():
                logger.warning(f"Source directory not found: {source_dir}")
                continue

            for pattern in self.config.file_patterns:
                for file_path in source_dir.rglob(pattern):
                    yield file_path

    def process_file(self, file_path: Path) -> Iterator[str]:
        """Process a single file and yield cleaned lines."""
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    cleaned = self._clean_line(line)
                    if cleaned:
                        yield cleaned
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")

    def _clean_line(self, line: str) -> str | None:
        """Clean and validate a single line."""
        line = line.rstrip()

        # Check length
        if len(line) < self.config.min_line_length:
            return None
        if len(line) > self.config.max_line_length:
            line = line[:self.config.max_line_length]

        # Check skip patterns
        for pattern in self._skip_regex:
            if pattern.search(line):
                return None

        return line

    def build_chunks(
        self,
        tokenizer: Callable[[str], list] | None = None,
    ) -> Iterator[CorpusChunk]:
        """Build training chunks from corpus."""
        # Simple word tokenizer if none provided
        if tokenizer is None:
            tokenizer = lambda x: x.split()

        for file_path in self.collect_files():
            lines = list(self.process_file(file_path))
            if not lines:
                continue

            # Build chunks
            current_chunk = []
            current_tokens = 0
            line_start = 0

            for i, line in enumerate(lines):
                tokens = tokenizer(line)
                token_count = len(tokens)

                if current_tokens + token_count > self.config.chunk_size:
                    # Yield current chunk
                    if current_chunk:
                        yield CorpusChunk(
                            text="\n".join(current_chunk),
                            source_file=str(file_path),
                            line_start=line_start,
                            line_end=i,
                            metadata={"token_count": current_tokens},
                        )
                    current_chunk = [line]
                    current_tokens = token_count
                    line_start = i
                else:
                    current_chunk.append(line)
                    current_tokens += token_count

            # Yield remaining
            if current_chunk:
                yield CorpusChunk(
                    text="\n".join(current_chunk),
                    source_file=str(file_path),
                    line_start=line_start,
                    line_end=len(lines),
                    metadata={"token_count": current_tokens},
                )

    def get_statistics(self) -> dict:
        """Get corpus statistics."""
        total_files = 0
        total_lines = 0
        total_chars = 0

        for file_path in self.collect_files():
            total_files += 1
            for line in self.process_file(file_path):
                total_lines += 1
                total_chars += len(line)

        return {
            "total_files": total_files,
            "total_lines": total_lines,
            "total_chars": total_chars,
            "avg_line_length": total_chars / total_lines if total_lines > 0 else 0,
        }


class DisassemblyParser(CorpusBuilder):
    """Specialized corpus builder for 65816 disassembly."""

    def __init__(self, config: CorpusConfig | None = None):
        if config is None:
            config = CorpusConfig(
                file_patterns=["*.asm", "*.s", "*.inc", "*.65816"],
                skip_patterns=[
                    r"^\s*;.*$",  # Skip comment-only lines (optional)
                    r"^\.db\s+\$[0-9A-Fa-f]{2}(?:,\s*\$[0-9A-Fa-f]{2}){10,}",  # Skip long data tables
                ],
            )
        super().__init__(config)

    def _clean_line(self, line: str) -> str | None:
        """Clean assembly line with special handling."""
        line = super()._clean_line(line)
        if not line:
            return None

        # Normalize whitespace around instructions
        line = re.sub(r"\s+", " ", line)

        # Keep labels and instructions
        return line


class ConversationParser(CorpusBuilder):
    """Specialized corpus builder for conversation/notes data."""

    def __init__(self, config: CorpusConfig | None = None):
        if config is None:
            config = CorpusConfig(
                file_patterns=["*.md", "*.txt", "*.json"],
                min_line_length=5,
                skip_patterns=[
                    r"^#",  # Skip markdown headers (keep content)
                    r"^\s*```",  # Skip code fence markers
                    r"^---$",  # Skip horizontal rules
                ],
            )
        super().__init__(config)

    def _clean_line(self, line: str) -> str | None:
        """Clean conversation line."""
        line = super()._clean_line(line)
        if not line:
            return None

        # Remove markdown formatting but keep content
        line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)  # Bold
        line = re.sub(r"\*([^*]+)\*", r"\1", line)  # Italic
        line = re.sub(r"`([^`]+)`", r"\1", line)  # Inline code

        return line
