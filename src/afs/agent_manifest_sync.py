"""Copy-based sync for the shared agent harness manifest."""

from __future__ import annotations

import filecmp
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent_manifest import export_for_harness


@dataclass(frozen=True)
class ManifestSyncAction:
    action: str
    harness: str
    target: str
    source: str = ""
    status: str = "pending"
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "action": self.action,
            "harness": self.harness,
            "target": self.target,
            "source": self.source,
            "status": self.status,
            "detail": self.detail,
        }


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _selected_harnesses(data: dict[str, Any], harnesses: set[str] | None) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for harness in _as_list(data.get("harnesses")):
        if not isinstance(harness, dict):
            continue
        name = str(harness.get("name", "")).strip()
        if not name:
            continue
        if harnesses and name not in harnesses:
            continue
        selected.append(harness)
    return selected


def _harness_map(data: dict[str, Any], harnesses: set[str] | None) -> dict[str, dict[str, Any]]:
    return {
        str(harness["name"]): harness
        for harness in _selected_harnesses(data, harnesses)
        if str(harness.get("name", "")).strip()
    }


def _skill_state(source: Path, target: Path) -> tuple[str, str]:
    if not source.exists():
        return "error", "canonical skill path missing"
    if not source.is_dir():
        return "error", "canonical skill path is not a directory"
    if target.is_symlink():
        return "would_replace_symlink", "target is a symlink"
    if target.exists() and not target.is_dir():
        return "error", "target exists and is not a directory"
    if not target.exists():
        return "would_create", "target skill directory missing"
    for source_file in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = source_file.relative_to(source)
        target_file = target / relative
        if not target_file.exists():
            return "would_update", f"missing file: {relative}"
        if not filecmp.cmp(source_file, target_file, shallow=False):
            return "would_update", f"changed file: {relative}"
    return "up_to_date", "canonical files already copied"


def _copy_skill_tree(source: Path, target: Path) -> None:
    if target.is_symlink():
        target.unlink()
    if target.exists() and not target.is_dir():
        raise FileExistsError(f"target exists and is not a directory: {target}")
    target.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        destination = target / relative
        if item.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)


def _export_payload(data: dict[str, Any], harness_name: str) -> str:
    payload = export_for_harness(data, harness_name)
    payload["generated_by"] = "afs agent-manifest sync"
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _export_state(payload: str, target: Path) -> tuple[str, str]:
    if target.exists() and target.read_text(encoding="utf-8") == payload:
        return "up_to_date", "export already current"
    if target.exists():
        return "would_update", "export differs"
    return "would_create", "export missing"


def sync_manifest(
    data: dict[str, Any],
    *,
    apply: bool = False,
    harnesses: set[str] | None = None,
    sync_skills: bool = True,
    sync_exports: bool = True,
) -> list[ManifestSyncAction]:
    """Plan or apply manifest sync actions.

    Skill sync copies canonical skill directories into harness skill roots. It
    does not use symlinks and it leaves extra destination files alone.
    """
    actions: list[ManifestSyncAction] = []
    selected = _harness_map(data, harnesses)

    if sync_skills:
        for skill in _as_list(data.get("skills")):
            if not isinstance(skill, dict):
                continue
            name = str(skill.get("name", "")).strip()
            canonical = Path(str(skill.get("canonical_path", ""))).expanduser()
            if not name:
                continue
            for target_name in {str(item) for item in _as_list(skill.get("targets"))}:
                harness = selected.get(target_name)
                if harness is None:
                    continue
                for raw_root in _as_list(harness.get("skill_roots")):
                    root = Path(str(raw_root)).expanduser()
                    target = root / name
                    status, detail = _skill_state(canonical, target)
                    if apply and status.startswith("would_"):
                        _copy_skill_tree(canonical, target)
                        status = "synced"
                    actions.append(
                        ManifestSyncAction(
                            action="copy_skill",
                            harness=target_name,
                            source=str(canonical),
                            target=str(target),
                            status=status,
                            detail=detail,
                        )
                    )

    if sync_exports:
        for harness_name, harness in selected.items():
            for raw_target in _as_list(harness.get("manifest_exports")):
                target = Path(str(raw_target)).expanduser()
                try:
                    payload = _export_payload(data, harness_name)
                    status, detail = _export_state(payload, target)
                    if apply and status.startswith("would_"):
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(payload, encoding="utf-8")
                        status = "synced"
                except Exception as exc:
                    status = "error"
                    detail = str(exc)
                actions.append(
                    ManifestSyncAction(
                        action="write_export",
                        harness=harness_name,
                        target=str(target),
                        status=status,
                        detail=detail,
                    )
                )

    return actions
