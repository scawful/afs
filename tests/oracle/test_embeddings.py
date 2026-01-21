"""Tests for Oracle embedding generator."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from afs.oracle.embeddings import (
    EmbeddingChunk,
    OracleEmbeddingGenerator,
    OracleEmbeddingStats,
)


class TestEmbeddingChunk(unittest.TestCase):
    """Tests for EmbeddingChunk dataclass."""

    def test_chunk_creation(self):
        """Test creating an embedding chunk."""
        chunk = EmbeddingChunk(
            id="asm:test.asm:TestRoutine",
            source_type="asm_routine",
            source_path="test.asm:10",
            content="LDA #$42\nRTS",
            metadata={"routine_name": "TestRoutine"},
        )

        self.assertEqual(chunk.id, "asm:test.asm:TestRoutine")
        self.assertEqual(chunk.source_type, "asm_routine")
        self.assertIn("routine_name", chunk.metadata)

    def test_to_dict(self):
        """Test converting chunk to dictionary."""
        chunk = EmbeddingChunk(
            id="test",
            source_type="asm",
            source_path="test.asm",
            content="A" * 1000,  # Long content
        )

        d = chunk.to_dict()

        self.assertEqual(d["id"], "test")
        self.assertLessEqual(len(d["content"]), 500)  # Should be truncated


class TestOracleEmbeddingStats(unittest.TestCase):
    """Tests for OracleEmbeddingStats dataclass."""

    def test_stats_default_values(self):
        """Test default stats values."""
        stats = OracleEmbeddingStats()

        self.assertEqual(stats.total_chunks, 0)
        self.assertEqual(stats.asm_files, 0)
        self.assertEqual(stats.symbols, 0)
        self.assertEqual(stats.errors, [])

    def test_stats_summary(self):
        """Test stats summary string."""
        stats = OracleEmbeddingStats(
            total_chunks=100,
            asm_files=10,
            asm_routines=50,
            symbols=30,
            docs=5,
            comments=5,
        )

        summary = stats.summary()

        self.assertIn("100", summary)
        self.assertIn("ASM files: 10", summary)
        self.assertIn("Symbols: 30", summary)


class TestOracleEmbeddingGenerator(unittest.TestCase):
    """Tests for OracleEmbeddingGenerator."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def test_extract_asm_chunks_file_content(self):
        """Test extracting chunks from ASM content."""
        # Create a test ASM file
        asm_content = """
; Test file
namespace Oracle

TestRoutine:
    LDA #$42
    STA $7E0010
    RTS

AnotherRoutine:
    LDA #$00
    RTS
"""
        asm_file = self.temp_path / "test.asm"
        asm_file.write_text(asm_content)

        generator = OracleEmbeddingGenerator(
            oracle_path=self.temp_path,
            output_dir=self.temp_path / "output",
        )

        chunks = list(generator.extract_asm_chunks(asm_file))

        # Should have file chunk + routine chunks
        self.assertGreater(len(chunks), 0)

        # Check for file chunk
        file_chunks = [c for c in chunks if c.source_type == "asm_file"]
        self.assertEqual(len(file_chunks), 1)

        # Check for routine chunks
        routine_chunks = [c for c in chunks if c.source_type == "asm_routine"]
        self.assertGreaterEqual(len(routine_chunks), 1)

    def test_routine_pattern_matching(self):
        """Test that routine pattern matches correctly."""
        pattern = OracleEmbeddingGenerator.ROUTINE_PATTERN

        # Should match
        self.assertTrue(pattern.match("TestRoutine:"))
        self.assertTrue(pattern.match("_PrivateRoutine:"))
        self.assertTrue(pattern.match("Module_Init123:"))

        # Should not match
        self.assertFalse(pattern.match("    LDA #$42"))
        self.assertFalse(pattern.match("; Comment:"))
        self.assertFalse(pattern.match("123Invalid:"))

    def test_namespace_pattern_matching(self):
        """Test namespace pattern matching."""
        pattern = OracleEmbeddingGenerator.NAMESPACE_PATTERN

        match = pattern.search("namespace Oracle")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "Oracle")

    def test_extract_symbol_chunks_with_valid_file(self):
        """Test extracting chunks from symbol file."""
        symbols = [
            {"name": "LinkHealth", "address": "$7E036C", "comment": "Link's health", "category": "player"},
            {"name": "RoomID", "address": "$7E00A0", "comment": "Current room", "category": "state"},
        ]

        symbols_file = self.temp_path / "symbols.json"
        with open(symbols_file, "w") as f:
            json.dump(symbols, f)

        generator = OracleEmbeddingGenerator(
            oracle_path=self.temp_path,
            output_dir=self.temp_path / "output",
        )

        chunks = list(generator.extract_symbol_chunks(symbols_file))

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].source_type, "symbol")
        self.assertIn("LinkHealth", chunks[0].content)

    def test_extract_doc_chunks(self):
        """Test extracting chunks from markdown docs."""
        docs_dir = self.temp_path / "Docs"
        docs_dir.mkdir()

        doc_content = """# Test Documentation

## Overview

This is a test document with some content.
It has multiple paragraphs.

## Features

- Feature 1
- Feature 2
"""
        doc_file = docs_dir / "Test.md"
        doc_file.write_text(doc_content)

        generator = OracleEmbeddingGenerator(
            oracle_path=self.temp_path,
            output_dir=self.temp_path / "output",
        )

        chunks = list(generator.extract_doc_chunks())

        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0].source_type, "doc")

    def test_generate_all_creates_index(self):
        """Test that generate_all creates an index file."""
        # Create minimal test structure
        asm_file = self.temp_path / "test.asm"
        asm_file.write_text("; Test\norg $008000\nNOP")

        output_dir = self.temp_path / "output"

        generator = OracleEmbeddingGenerator(
            oracle_path=self.temp_path,
            output_dir=output_dir,
            embed_fn=None,  # No actual embedding
        )

        stats = generator.generate_all(include_symbols=False)

        # Check index was created
        index_path = output_dir / "oracle_embedding_index.json"
        self.assertTrue(index_path.exists())

        # Check stats
        self.assertGreater(stats.asm_files, 0)


class TestEmbeddingFunctionCreation(unittest.TestCase):
    """Tests for embedding function creation."""

    def test_create_none_embedding_fn(self):
        """Test creating null embedding function."""
        from afs.oracle.embeddings import create_embedding_fn

        fn = create_embedding_fn("none")
        self.assertIsNone(fn)


if __name__ == "__main__":
    unittest.main()
