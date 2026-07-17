"""Generic context source provider framework."""

from .models import (
    CONTEXT_SOURCE_KINDS,
    ContextSourceProvider,
    ContextSourceRecord,
    ResearchRequest,
    ResearchSourceProvider,
    SourceProviderSpec,
    SourceSyncResult,
)
from .registry import (
    assert_source_materialization_supported,
    discover_source_provider_specs,
    load_source_provider,
    load_source_provider_by_name,
    load_source_providers,
    materialize_source_records,
)
from .research import (
    ResearchProviderError,
    execute_research_provider,
    normalize_research_records,
)

__all__ = [
    "CONTEXT_SOURCE_KINDS",
    "ContextSourceProvider",
    "ContextSourceRecord",
    "ResearchRequest",
    "ResearchProviderError",
    "ResearchSourceProvider",
    "SourceProviderSpec",
    "SourceSyncResult",
    "assert_source_materialization_supported",
    "discover_source_provider_specs",
    "execute_research_provider",
    "load_source_provider",
    "load_source_provider_by_name",
    "load_source_providers",
    "materialize_source_records",
    "normalize_research_records",
]
