from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from afs.cli.sources import _sources_sync
from afs.context_layout import scaffold_v2
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


def test_materialize_source_records_rejects_v2_before_consuming_records(
    tmp_path: Path,
) -> None:
    context = tmp_path / ".context"
    scaffold_v2(context)

    def forbidden_records():
        raise AssertionError("v2 records must not be consumed")
        yield  # pragma: no cover

    with pytest.raises(ValueError, match="scoped ingestion is not implemented"):
        materialize_source_records(
            context_path=context,
            provider_name="example",
            records=forbidden_records(),
            dry_run=True,
        )

    assert not (context / "items").exists()


@pytest.mark.parametrize("apply", [False, True])
def test_sources_sync_rejects_v2_before_loading_or_invoking_provider(
    apply: bool,
    capsys,
    monkeypatch,
    tmp_path: Path,
) -> None:
    context = tmp_path / ".context"
    scaffold_v2(context)
    monkeypatch.setattr(
        "afs.cli.sources._load_manager_and_context",
        lambda _args: (object(), object(), context),
    )

    def forbidden_provider_load(*_args, **_kwargs):
        raise AssertionError("v2 provider must not be loaded or invoked")

    monkeypatch.setattr(
        "afs.cli.sources.load_source_providers",
        forbidden_provider_load,
    )
    args = argparse.Namespace(
        provider="example",
        query="",
        limit=10,
        apply=apply,
        json=False,
    )

    assert _sources_sync(args) == 2
    assert "scoped ingestion is not implemented" in capsys.readouterr().out
    assert not (context / "items").exists()


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
