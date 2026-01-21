"""Base classes for AFS service adapters."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import ServiceDefinition


@dataclass
class ServiceAdapter:
    platform_name: str

    def render(self, definition: ServiceDefinition) -> str:
        raise NotImplementedError

    def file_extension(self) -> str:
        return ""
