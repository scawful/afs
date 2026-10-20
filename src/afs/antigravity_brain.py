"""Read Antigravity "brain" conversation artifacts and import them as AFS drafts.

Antigravity keeps per-conversation planning artifacts (``task.md``,
``implementation_plan.md``, ``walkthrough.md``, sometimes ``handoff.md``) under
``~/.gemini/antigravity/brain/<conversation-uuid>/``. Other harnesses never see
them unless they are copied into shared context. The directory records no
workspace, so the project is inferred from the git roots the artifacts mention.
"""

from __future__ import annotations

import re
import time
import uuid
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_BRAIN_ROOT = Path.home() / ".gemini" / "antigravity" / "brain"
ARTIFACT_ORDER = ("handoff.md", "walkthrough.md", "task.md", "implementation_plan.md")
MAX_IMPORT_CHARS = 20000
_ABSOLUTE_PATH = re.compile(r"(?<![\w.])(/[\w.@+-]+(?:/[\w.@+-]+)+)")
_HEADING = re.compile(r"^#+\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class BrainConversation:
    conversation_id: str
    path: Path
    updated_at: float
    files: list[str] = field(default_factory=list)
    title: str = ""
    project: str = ""

    def artifact_id(self) -> str | None:
        """Stable draft ID so re-importing the same conversation is detected."""
        try:
            return uuid.UUID(self.conversation_id).hex
        except ValueError:
            return None

    def entry_file(self) -> Path | None:
        for name in ARTIFACT_ORDER:
            if name in self.files:
                return self.path / name
        return self.path / self.files[0] if self.files else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "path": str(self.path),
            "updated_at": datetime.fromtimestamp(self.updated_at, timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "files": list(self.files),
            "title": self.title,
            "project": self.project,
        }


def _artifact_files(directory: Path) -> list[str]:
    names = [
        entry.name
        for entry in directory.iterdir()
        if entry.is_file() and entry.suffix == ".md" and not entry.name.startswith(".")
    ]
    order = {name: index for index, name in enumerate(ARTIFACT_ORDER)}
    return sorted(names, key=lambda name: (order.get(name, len(order)), name))


def _worktree_main_root(marker: Path) -> Path | None:
    """Map a linked worktree's ``.git`` file to its main checkout."""
    text = _read(marker).strip()
    if not text.startswith("gitdir:"):
        return None
    gitdir = Path(text.removeprefix("gitdir:").strip())
    if gitdir.parent.name != "worktrees" or gitdir.parent.parent.name != ".git":
        return None
    return gitdir.parent.parent.parent


def _git_root(path: Path, cache: dict[Path, Path | None]) -> Path | None:
    visited: list[Path] = []
    current = path
    while True:
        if current in cache:
            root = cache[current]
            break
        visited.append(current)
        marker = current / ".git"
        if marker.is_file():
            root = _worktree_main_root(marker) or current
            break
        if marker.exists():
            root = current
            break
        if current.parent == current or current == Path.home():
            root = None
            break
        current = current.parent
    for item in visited:
        cache[item] = root
    return root


def infer_project(texts: Iterable[str], cache: dict[Path, Path | None] | None = None) -> str:
    cache = {} if cache is None else cache
    roots: Counter[str] = Counter()
    for text in texts:
        for raw in _ABSOLUTE_PATH.findall(text):
            root = _git_root(Path(raw.rstrip(".:")), cache)
            if root is not None:
                roots[str(root)] += 1
    return roots.most_common(1)[0][0] if roots else ""


def _title(text: str, fallback: str) -> str:
    match = _HEADING.search(text)
    return match.group(1)[:120] if match else fallback


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def load_conversation(
    directory: Path, cache: dict[Path, Path | None] | None = None
) -> BrainConversation | None:
    if not directory.is_dir():
        return None
    files = _artifact_files(directory)
    if not files:
        return None
    texts = {name: _read(directory / name) for name in files}
    updated = max((directory / name).stat().st_mtime for name in files)
    conversation = BrainConversation(
        conversation_id=directory.name, path=directory, updated_at=updated, files=files
    )
    entry = conversation.entry_file()
    conversation.title = _title(texts.get(entry.name, "") if entry else "", directory.name)
    conversation.project = infer_project(texts.values(), cache)
    return conversation


def list_conversations(
    root: Path = DEFAULT_BRAIN_ROOT,
    *,
    since_days: float | None = None,
    project: Path | None = None,
    limit: int = 50,
) -> list[BrainConversation]:
    if not root.is_dir() or limit <= 0:
        return []
    cutoff = time.time() - since_days * 86400 if since_days else None
    directories = sorted(
        (entry for entry in root.iterdir() if entry.is_dir() and not entry.name.startswith(".")),
        key=lambda entry: entry.stat().st_mtime,
        reverse=True,
    )
    project_root = str(project.expanduser().resolve()) if project else ""
    cache: dict[Path, Path | None] = {}
    results: list[BrainConversation] = []
    for directory in directories:
        conversation = load_conversation(directory, cache)
        if conversation is None:
            continue
        if cutoff is not None and conversation.updated_at < cutoff:
            continue
        if project_root and conversation.project != project_root:
            continue
        results.append(conversation)
        if len(results) >= limit:
            break
    results.sort(key=lambda item: item.updated_at, reverse=True)
    return results


def import_body(conversation: BrainConversation) -> str:
    entry = conversation.entry_file()
    stamp = conversation.to_dict()["updated_at"]
    lines = [
        f"Imported from Antigravity conversation `{conversation.conversation_id}`.",
        f"Source: {conversation.path}",
        f"Updated: {stamp}",
    ]
    if conversation.project:
        lines.append(f"Project: {conversation.project}")
    others = [name for name in conversation.files if entry is None or name != entry.name]
    if others:
        lines.append(
            "Other artifacts: " + ", ".join(str(conversation.path / name) for name in others)
        )
    if entry is not None:
        text = _read(entry)
        lines.extend(["", f"## {entry.name}", "", text[:MAX_IMPORT_CHARS].rstrip()])
        if len(text) > MAX_IMPORT_CHARS:
            lines.append(f"\n[truncated; read {entry} for the rest]")
    return "\n".join(lines) + "\n"
