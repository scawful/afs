from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def temp_dir() -> Path:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_jsonl_file(temp_dir: Path) -> Path:
    """Create a small JSONL training set."""
    path = temp_dir / "samples.jsonl"
    samples = [
        {
            "instruction": "Question one",
            "output": "Answer one",
            "thinking": "Reasoning one",
            "domain": "asm",
            "source": "test",
            "quality_score": 0.9,
        },
        {
            "instruction": "Question two",
            "output": "Answer two",
            "thinking": "Reasoning two",
            "domain": "asm",
            "source": "test",
            "quality_score": 0.4,
        },
        {
            "instruction": "Question three",
            "output": "Answer three",
            "thinking": "Reasoning three",
            "domain": "asm",
            "source": "test",
            "quality_score": 0.2,
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample) + "\n")
    return path
