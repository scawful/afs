"""Discovery and loading for extension-owned context source providers."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..plugins import load_enabled_extensions
from .models import (
    ContextSourceProvider,
    ContextSourceRecord,
    SourceProviderSpec,
    SourceSyncResult,
    safe_source_id,
)


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


def load_source_provider(spec: SourceProviderSpec, *, manifest: Any | None = None) -> ContextSourceProvider:
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
    """Load all enabled extension source providers, keyed by provider name."""
    manifests = load_enabled_extensions(config=config)
    providers: dict[str, ContextSourceProvider] = {}
    for manifest in manifests.values():
        for spec in _iter_manifest_source_specs(manifest):
            try:
                provider = load_source_provider(spec, manifest=manifest)
            except Exception:
                continue
            providers[str(getattr(provider, "name", spec.name))] = provider
    return dict(sorted(providers.items()))


def materialize_source_records(
    *,
    context_path: Path,
    provider_name: str,
    records: Iterable[ContextSourceRecord | dict[str, Any]],
    dry_run: bool = True,
) -> SourceSyncResult:
    """Write provider records under ``.context/items/sources/<provider>/``."""
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
