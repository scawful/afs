from __future__ import annotations

from pathlib import Path

import pytest

from afs import sensitivity
from afs.context_index import ContextSQLiteIndex
from afs.embeddings import _iter_source_files
from afs.hybrid_search import SourcePolicy, _path_allowed
from afs.manager import AFSManager
from afs.models import MountType
from afs.schema import AFSConfig, GeneralConfig
from afs.sensitivity import is_default_index_excluded


def _seed_tree(root: Path) -> None:
    (root / "embeddings").mkdir(parents=True)
    (root / "embeddings" / "vec.json").write_text('{"embedding": []}', encoding="utf-8")
    (root / ".afs").mkdir()
    (root / ".afs" / "state.md").write_text("runtime", encoding="utf-8")
    (root / "embedding_index.json").write_text("{}", encoding="utf-8")
    (root / "notes.md.bak").write_text("old", encoding="utf-8")
    (root / "notes.bak2").write_text("older", encoding="utf-8")
    (root / "notes.md").write_text("keep me", encoding="utf-8")


@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        ("embeddings/vec.json", True),
        ("sub/embeddings/vec.json", True),
        (".afs/state.md", True),
        ("embedding_index.json", True),
        ("notes.md.bak", True),
        ("notes.bak2", True),
        ("notes.md", False),
        ("embeddings_guide.md", False),
    ],
)
def test_is_default_index_excluded_patterns(tmp_path: Path, relative: str, expected: bool) -> None:
    assert is_default_index_excluded(tmp_path / relative, relative_path=relative) is expected


def test_context_root_under_embeddings_dir_is_not_excluded(tmp_path: Path) -> None:
    root = tmp_path / "embeddings" / "project"
    assert not is_default_index_excluded(root / "notes.md", relative_path="notes.md")


def test_music_root_is_excluded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    music = tmp_path / "Music"
    (music / "album").mkdir(parents=True)
    monkeypatch.setattr(sensitivity, "DEFAULT_INDEX_EXCLUDED_ROOTS", (str(music),))

    assert is_default_index_excluded(music)
    assert is_default_index_excluded(music / "album" / "track.md")
    assert not is_default_index_excluded(tmp_path / "Musicals" / "notes.md")


def test_context_index_skips_default_exclusions(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    scratchpad = context_root / "scratchpad"
    _seed_tree(scratchpad)
    manager = AFSManager(config=AFSConfig(general=GeneralConfig(context_root=context_root)))
    index = ContextSQLiteIndex(manager, context_root, db_path=tmp_path / "index.sqlite3")

    index.rebuild(mount_types=[MountType.SCRATCHPAD])
    paths = {row["relative_path"] for row in index.query(mount_types=[MountType.SCRATCHPAD])}

    assert paths == {"notes.md"}


def test_hybrid_path_allowed_rejects_default_exclusions(tmp_path: Path) -> None:
    _seed_tree(tmp_path)
    policy = SourcePolicy(allow_hidden=True)

    def allowed(relative: str) -> bool:
        return _path_allowed(tmp_path / relative, tmp_path, relative, policy, [])

    assert allowed("notes.md")
    for relative in (
        "embeddings/vec.json",
        ".afs/state.md",
        "embedding_index.json",
        "notes.md.bak",
    ):
        assert not allowed(relative), relative


def test_embeddings_walker_skips_default_exclusions(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _seed_tree(source)

    found = {
        path.relative_to(source).as_posix()
        for _, path in _iter_source_files(
            [source],
            include_patterns=[],
            exclude_patterns=[],
            max_files=None,
            include_hidden=True,
            output_dir=tmp_path / "out",
        )
    }

    assert found == {"notes.md"}
