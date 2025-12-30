"""AFS context validator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .mapping import resolve_directory_map
from .models import MountType, ProjectMetadata
from .schema import DirectoryConfig


class AFSValidator:
    def __init__(
        self,
        context_root: Path,
        afs_directories: list[DirectoryConfig] | None = None,
    ) -> None:
        self.root = context_root
        self._afs_directories = afs_directories

    def check_integrity(self) -> dict[str, Any]:
        if not self.root.exists():
            return {
                "valid": False,
                "missing": [],
                "errors": ["context root does not exist"],
            }

        metadata = _load_metadata(self.root)
        directory_map = resolve_directory_map(
            afs_directories=self._afs_directories,
            metadata=metadata,
        )
        required_dirs = [directory_map.get(mt, mt.value) for mt in MountType]
        status: dict[str, Any] = {"valid": True, "missing": [], "errors": []}

        for directory in required_dirs:
            if not (self.root / directory).is_dir():
                status["valid"] = False
                status["missing"].append(directory)

        return status


def _load_metadata(context_root: Path) -> ProjectMetadata | None:
    metadata_path = context_root / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return ProjectMetadata.from_dict(payload)
