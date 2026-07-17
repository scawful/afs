"""Generic context source provider framework."""

from .models import (
    CONTEXT_SOURCE_KINDS,
    ContextSourceProvider,
    ContextSourceRecord,
    SourceProviderSpec,
    SourceSyncResult,
)
from .registry import (
    assert_source_materialization_supported,
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
    "assert_source_materialization_supported",
    "discover_source_provider_specs",
    "load_source_provider",
    "load_source_providers",
    "materialize_source_records",
]
