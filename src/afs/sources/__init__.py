"""Generic context source provider framework."""

from .models import (
    CONTEXT_SOURCE_KINDS,
    ContextSourceProvider,
    ContextSourceRecord,
    SourceProviderSpec,
    SourceSyncResult,
)
from .registry import (
    discover_source_provider_specs,
    load_source_provider,
    load_source_providers,
    materialize_source_records,
)

__all__ = [
    "CONTEXT_SOURCE_KINDS",
    "ContextSourceProvider",
    "ContextSourceRecord",
    "SourceProviderSpec",
    "SourceSyncResult",
    "discover_source_provider_specs",
    "load_source_provider",
    "load_source_providers",
    "materialize_source_records",
]
