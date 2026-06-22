from __future__ import annotations

import json
from pathlib import Path

from afs.extensions import load_extension_manifest
from afs.sources import (
    ContextSourceRecord,
    discover_source_provider_specs,
    materialize_source_records,
)


def test_context_source_record_renders_indexable_markdown() -> None:
    record = ContextSourceRecord(
        id="ABC 123",
        kind="ticket",
        title="Fix parser",
        body="Parser fails on escaped commas.",
        provider="example",
        uri="https://example.invalid/ABC-123",
        metadata={"priority": "high"},
    )

    rendered = record.render_markdown()

    assert record.id == "ABC-123"
    assert rendered.startswith("---\n")
    assert "# Fix parser" in rendered
    assert "Parser fails" in rendered
    frontmatter = json.loads(rendered.splitlines()[1])
    assert frontmatter["kind"] == "ticket"
    assert frontmatter["metadata"]["priority"] == "high"


def test_materialize_source_records_writes_under_context_items(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    record = ContextSourceRecord(id="review-1", kind="review", title="Review one", body="Looks good")

    dry = materialize_source_records(
        context_path=context,
        provider_name="example provider",
        records=[record],
        dry_run=True,
    )
    assert dry.written_paths == ()
    assert dry.target_dir == context / "items" / "sources" / "example-provider"

    result = materialize_source_records(
        context_path=context,
        provider_name="example provider",
        records=[record],
        dry_run=False,
    )

    assert len(result.written_paths) == 1
    written = result.written_paths[0]
    assert written.name == "review-review-1.md"
    assert "# Review one" in written.read_text(encoding="utf-8")


def test_extension_manifest_declares_context_sources(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "afs_example"
    repo.mkdir()
    (repo / "extension.toml").write_text(
        'name = "afs_example"\n'
        'description = "Example extension"\n'
        '[[context_sources]]\n'
        'name = "example_tasks"\n'
        'module = "afs_example.sources"\n'
        'factory = "register_context_source_provider"\n'
        'kinds = ["task", "review"]\n',
        encoding="utf-8",
    )

    manifest = load_extension_manifest(repo / "extension.toml")
    assert manifest.context_sources[0]["name"] == "example_tasks"

    config = {
        "extensions": {
            "enabled_extensions": ["afs_example"],
            "extension_dirs": [str(tmp_path)],
            "auto_discover": False,
        }
    }
    specs = discover_source_provider_specs(config=config)

    assert [spec.name for spec in specs] == ["example_tasks"]
    assert specs[0].kinds == ("task", "review")
