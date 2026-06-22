"""Compatibility shim for extension-owned entity extraction."""

from __future__ import annotations

import importlib

_ERROR = (
    "knowledge.entity_extractor requires the afs_scawful extension repo. "
    "Enable afs_scawful or see docs/EXTENSION_MIGRATION.md."
)

try:  # pragma: no cover - compatibility path when extension is installed
    _module = importlib.import_module("afs_scawful.knowledge.entity_extractor")
except Exception as exc:  # pragma: no cover - default path without extension
    raise RuntimeError(_ERROR) from exc

EntityExtractor = _module.EntityExtractor
ExtractedEntity = _module.ExtractedEntity
ExtractionResult = _module.ExtractionResult
extract_entities = _module.extract_entities
extract_with_stats = _module.extract_with_stats

__all__ = [
    "EntityExtractor",
    "ExtractedEntity",
    "ExtractionResult",
    "extract_entities",
    "extract_with_stats",
]
