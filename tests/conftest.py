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


@pytest.fixture(autouse=True)
def _isolated_approval_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the global agent approval gate out of ``~/.config/afs/agents``.

    Tests that need a specific store still set ``AFS_AGENT_APPROVALS_PATH``
    themselves; their ``monkeypatch.setenv`` runs after this and wins.
    """
    monkeypatch.delenv("AFS_AGENT_APPROVALS_DB", raising=False)
    monkeypatch.setenv(
        "AFS_AGENT_APPROVALS_PATH", str(tmp_path / "isolated-approvals.json")
    )


@pytest.fixture(autouse=True)
def _clear_native_skills_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep host ``AFS_NATIVE_SKILLS=1`` from collapsing matched-skill fixtures.

    Cursor/agent shells often export native skills for production prompts.
    Unit tests that assert Relevant Skills / Skill Enforcement payloads need
    the matched-skill renderer unless they opt in with ``native_skills=True``.
    """
    monkeypatch.delenv("AFS_NATIVE_SKILLS", raising=False)


@pytest.fixture(autouse=True)
def _enable_optional_feature_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep training/embeddings CLI tests usable under afs-lite fail-closed gates."""
    monkeypatch.setenv("AFS_ENABLE_TRAINING", "1")
    monkeypatch.setenv("AFS_ENABLE_EMBEDDINGS", "1")


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
