"""Discovery and loading for extension-owned context source providers."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..context_layout import LAYOUT_VERSION, detect_layout_version
from ..plugins import load_enabled_extensions
from .models import (
    ContextSourceProvider,
    ContextSourceRecord,
    ResearchSourceProvider,
    SourceProviderSpec,
    SourceSyncResult,
    safe_source_id,
)

SourceProvider = ContextSourceProvider | ResearchSourceProvider

V2_SOURCE_SYNC_UNAVAILABLE = (
    "context source sync is unavailable for layout v2 because scoped ingestion "
    "is not implemented; use sources list/status for read-only inspection or a "
    "v1 context for .context/items/sources materialization. Future v2 ingestion "
    "will target knowledge/projects/<project-id> or explicit knowledge/common."
)


def assert_source_materialization_supported(context_path: Path) -> None:
    """Fail before provider records can enter an unscoped v2 compatibility path."""

    if detect_layout_version(context_path) == LAYOUT_VERSION:
        raise ValueError(V2_SOURCE_SYNC_UNAVAILABLE)


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, str) and str(item).strip()]


def _iter_manifest_source_specs(manifest: Any) -> list[SourceProviderSpec]:
    specs = getattr(manifest, "context_sources", [])
    if not isinstance(specs, list):
        return []
    result: list[SourceProviderSpec] = []
    extension_name = str(getattr(manifest, "name", "")).strip()
    for entry in specs:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        module = str(entry.get("module", "")).strip()
        if not name or not module:
            continue
        factory = str(entry.get("factory", "register_context_source_provider")).strip()
        description = str(entry.get("description", "")).strip()
        result.append(
            SourceProviderSpec(
                name=name,
                module=module,
                factory=factory,
                description=description,
                kinds=tuple(_as_str_list(entry.get("kinds"))),
                extension=extension_name,
            )
        )
    return result


def discover_source_provider_specs(config: Any = None) -> list[SourceProviderSpec]:
    """Return manifest-declared provider specs from enabled extensions."""
    specs: list[SourceProviderSpec] = []
    for manifest in load_enabled_extensions(config=config).values():
        specs.extend(_iter_manifest_source_specs(manifest))
    return sorted(specs, key=lambda spec: (spec.extension, spec.name))


@contextmanager
def _import_roots(roots: Iterable[Path]):
    original = list(sys.path)
    candidates: list[str] = []
    for root in roots:
        candidates.extend([str(root), str(root.parent)])
    sys.path = [candidate for candidate in candidates if Path(candidate).exists()] + original
    try:
        yield
    finally:
        sys.path = original


def load_source_provider(
    spec: SourceProviderSpec,
    *,
    manifest: Any | None = None,
) -> SourceProvider:
    """Import and instantiate one provider spec."""
    roots = list(getattr(manifest, "import_roots", []) or [])
    with _import_roots(roots):
        module = importlib.import_module(spec.module)
    factory = getattr(module, spec.factory)
    provider = factory()
    name = str(getattr(provider, "name", "")).strip()
    if not name:
        provider.name = spec.name
    if not hasattr(provider, "kinds"):
        provider.kinds = spec.kinds
    return provider


def load_source_providers(config: Any = None) -> dict[str, ContextSourceProvider]:
    """Load enabled providers that support status and sync, keyed by name.

    Research-only providers are intentionally excluded. Callers that need an
    explicitly selected research provider should use
    :func:`load_source_provider_by_name` and check ``ResearchSourceProvider``.
    """
    manifests = load_enabled_extensions(config=config)
    providers: dict[str, ContextSourceProvider] = {}
    for manifest in manifests.values():
        for spec in _iter_manifest_source_specs(manifest):
            try:
                provider = load_source_provider(spec, manifest=manifest)
            except Exception:
                continue
            if not isinstance(provider, ContextSourceProvider):
                continue
            providers[str(getattr(provider, "name", spec.name))] = provider
    return dict(sorted(providers.items()))


def load_source_provider_by_name(
    name: str,
    *,
    config: Any = None,
) -> SourceProvider:
    """Load exactly one explicitly selected provider.

    Discovery reads manifests only. Non-selected provider modules are never
    imported, which keeps provider selection useful as a network-consent
    boundary for research callers.
    """

    requested = name.strip()
    if not requested:
        raise ValueError("source provider name must be non-empty")
    matches: list[tuple[SourceProviderSpec, Any]] = []
    for manifest in load_enabled_extensions(config=config).values():
        for spec in _iter_manifest_source_specs(manifest):
            if spec.name == requested:
                matches.append((spec, manifest))
    if not matches:
        raise KeyError(f"unknown enabled source provider: {requested}")
    if len(matches) > 1:
        extensions = sorted(spec.extension for spec, _manifest in matches)
        raise ValueError(
            f"source provider {requested!r} is declared by multiple extensions: "
            + ", ".join(extensions)
        )
    spec, manifest = matches[0]
    return load_source_provider(spec, manifest=manifest)


def materialize_source_records(
    *,
    context_path: Path,
    provider_name: str,
    records: Iterable[ContextSourceRecord | dict[str, Any]],
    dry_run: bool = True,
) -> SourceSyncResult:
    """Write provider records under the v1 ``items/sources/<provider>/`` path."""
    assert_source_materialization_supported(context_path)
    provider_slug = safe_source_id(provider_name)
    target_dir = context_path.expanduser().resolve() / "items" / "sources" / provider_slug
    normalized: list[ContextSourceRecord] = []
    for entry in records:
        record = entry if isinstance(entry, ContextSourceRecord) else ContextSourceRecord.from_dict(entry)
        if not record.provider:
            record = ContextSourceRecord(**{**record.to_dict(), "provider": provider_slug})
        normalized.append(record)

    written: list[Path] = []
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
        for record in normalized:
            path = target_dir / f"{safe_source_id(record.kind)}-{safe_source_id(record.id)}.md"
            path.write_text(record.render_markdown(), encoding="utf-8")
            written.append(path)

    return SourceSyncResult(
        provider=provider_slug,
        dry_run=dry_run,
        target_dir=target_dir,
        records=tuple(normalized),
        written_paths=tuple(written),
    )
