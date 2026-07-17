from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from afs.cli.sources import _sources_status, _sources_sync
from afs.context_layout import scaffold_v2
from afs.extensions import load_extension_manifest
from afs.sources import (
    ContextSourceRecord,
    ResearchProviderError,
    ResearchRequest,
    ResearchSourceProvider,
    SourceProviderSpec,
    discover_source_provider_specs,
    materialize_source_records,
    normalize_research_records,
)


def test_research_request_requires_explicit_bounded_network_scope() -> None:
    local = ResearchRequest(query="  inspect parser history  ")
    assert local.query == "inspect parser history"
    assert local.network_allowed is False

    network = ResearchRequest(
        query="official parser guidance",
        network_allowed=True,
        allowed_domains=("Docs.Python.org.", "docs.python.org"),
        max_results=5,
        timeout_seconds=12,
        max_bytes=4096,
    )
    assert network.allowed_domains == ("docs.python.org",)
    assert ResearchRequest.from_dict(network.to_dict()) == network

    with pytest.raises(ValueError, match="explicitly allowed domain"):
        ResearchRequest(query="browse broadly", network_allowed=True)
    with pytest.raises(ValueError, match="must be a boolean"):
        ResearchRequest(query="ambiguous consent", network_allowed="false")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="plain host names"):
        ResearchRequest(
            query="bad scope",
            network_allowed=True,
            allowed_domains=("https://example.com/path",),
        )
    with pytest.raises(ValueError, match="plain host names"):
        ResearchRequest(
            query="port is not a domain",
            network_allowed=True,
            allowed_domains=("example.com:443",),
        )
    for private_style in (
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "example.local",
        "service.internal",
        "999.999.999.999",
    ):
        with pytest.raises(ValueError, match="public plain host names"):
            ResearchRequest(
                query="private destination",
                network_allowed=True,
                allowed_domains=(private_style,),
            )
    with pytest.raises(ValueError, match="max_results"):
        ResearchRequest(query="too much", max_results=51)
    with pytest.raises(ValueError, match="no more than 2000"):
        ResearchRequest(query="x" * 2001)


def test_research_provider_protocol_is_runtime_checkable() -> None:
    class ExampleProvider:
        name = "example"

        def research(self, request: ResearchRequest):
            return [
                ContextSourceRecord(
                    id="result-1",
                    kind="doc",
                    title=request.query,
                )
            ]

    assert isinstance(ExampleProvider(), ResearchSourceProvider)


def test_research_records_are_bounded_to_allowed_https_domains() -> None:
    request = ResearchRequest(
        query="official docs",
        network_allowed=True,
        allowed_domains=("example.com",),
        max_results=1,
        max_bytes=2048,
    )
    records = normalize_research_records(
        request,
        [
            {
                "id": "doc-1",
                "kind": "doc",
                "title": "Official docs",
                "uri": "https://docs.example.com/guide",
                "body": "bounded evidence",
                "provider": "spoofed-provider",
            }
        ],
        provider_name="example",
    )
    assert records[0].provider == "example"

    with pytest.raises(ResearchProviderError, match="outside the allowed"):
        normalize_research_records(
            request,
            [
                {
                    "id": "bad",
                    "title": "Redirected",
                    "uri": "https://example.invalid/private",
                }
            ],
            provider_name="example",
        )
    with pytest.raises(ResearchProviderError, match="outside the allowed"):
        normalize_research_records(
            request,
            [
                {
                    "id": "userinfo",
                    "title": "Empty userinfo is still userinfo",
                    "uri": "https://@example.com/private",
                }
            ],
            provider_name="example",
        )
    with pytest.raises(ResearchProviderError, match="returned 2"):
        normalize_research_records(
            request,
            [records[0], records[0]],
            provider_name="example",
        )
    for unsafe_record in (
        {
            "id": "newline",
            "title": "Trusted heading\nForged terminal line",
            "uri": "https://example.com/newline",
        },
        {
            "id": "escape",
            "title": "Escape",
            "uri": "https://example.com/\x1b[2J",
        },
    ):
        with pytest.raises(ResearchProviderError, match="control characters"):
            normalize_research_records(
                request,
                [unsafe_record],
                provider_name="example",
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
        "afs.cli.sources.load_source_provider_by_name",
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


class _ResearchOnlyProvider:
    name = "research-only"

    def research(self, request: ResearchRequest):
        return []


def test_sources_status_reports_research_only_provider_without_calling_sync(
    capsys,
    monkeypatch,
    tmp_path: Path,
) -> None:
    spec = SourceProviderSpec(name="research-only", module="example.research")
    monkeypatch.setattr(
        "afs.cli.sources._load_manager_and_context",
        lambda _args: (object(), object(), tmp_path / ".context"),
    )
    monkeypatch.setattr(
        "afs.cli.sources.discover_source_provider_specs", lambda **_kwargs: [spec]
    )
    monkeypatch.setattr(
        "afs.cli.sources.load_source_provider_by_name",
        lambda *_args, **_kwargs: _ResearchOnlyProvider(),
    )

    assert _sources_status(argparse.Namespace(json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["providers"] == [
        {
            "name": "research-only",
            "kinds": [],
            "capabilities": {"sync": False, "research": True},
            "status": {"ok": True, "detail": "research-only provider"},
        }
    ]


def test_sources_sync_rejects_research_only_provider_cleanly(
    capsys,
    monkeypatch,
    tmp_path: Path,
) -> None:
    context = tmp_path / ".context"
    context.mkdir()
    monkeypatch.setattr(
        "afs.cli.sources._load_manager_and_context",
        lambda _args: (object(), object(), context),
    )
    monkeypatch.setattr(
        "afs.cli.sources.load_source_provider_by_name",
        lambda *_args, **_kwargs: _ResearchOnlyProvider(),
    )
    args = argparse.Namespace(
        provider="research-only",
        query="",
        limit=10,
        apply=False,
        json=False,
    )

    assert _sources_sync(args) == 2
    assert "does not support sync" in capsys.readouterr().out
