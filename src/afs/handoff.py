"""Immutable, multi-stream conversation handoffs for AFS.

Schema v3 revisions are human-readable Markdown artifacts.  Legacy callers
continue to use :meth:`HandoffStore.create`, while new callers use
``create_revision`` with an explicit title.  Acknowledgement and closure are
append-only events, so a published revision is never rewritten.
"""

from __future__ import annotations

import base64
import json
import os
import re
import threading
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import (
    ArtifactRootResolver,
    MarkdownArtifact,
    MarkdownArtifactCodec,
    default_artifact_root,
    infer_scope_id,
    validate_scope_id,
)
from .context_paths import resolve_mount_root
from .models import MountType

HANDOFF_SCHEMA_VERSION = "3"
LEGACY_HANDOFF_SCHEMA_VERSION = "2"
_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PAYLOAD_PATTERN = re.compile(r"<!-- afs-handoff-payload:([A-Za-z0-9_-]+) -->")
_MANIFEST_LOCK = threading.RLock()
_EVENT_LOCK = threading.RLock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp_key(value: str) -> float:
    raw = value.strip()
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return float("-inf")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _safe_identifier(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not _SAFE_ID_PATTERN.fullmatch(normalized) or normalized in {".", ".."}:
        raise ValueError(
            f"{field_name} must be a safe identifier containing only letters, "
            "numbers, '.', '_' or '-'"
        )
    return normalized


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


@dataclass
class HandoffPacket:
    session_id: str
    agent_name: str
    timestamp: str
    accomplished: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    context_snapshot: dict[str, Any] = field(default_factory=dict)
    open_tasks: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    target_agent: str | None = None
    priority: str = "normal"
    schema_version: str = HANDOFF_SCHEMA_VERSION
    acknowledged_by: list[str] = field(default_factory=list)
    stream_id: str = ""
    revision_id: str = ""
    title: str = ""
    supersedes: list[str] = field(default_factory=list)
    closed: bool = False
    artifact_path: str = ""

    def __post_init__(self) -> None:
        # ``session_id`` is the public v1/v2 lookup key.  In v3 it aliases the
        # immutable revision id so existing supervisor and CLI callers keep
        # working without a second identifier migration.
        if not self.revision_id:
            self.revision_id = self.session_id
        if not self.session_id:
            self.session_id = self.revision_id
        if not self.stream_id:
            self.stream_id = self.revision_id
        if not self.title:
            self.title = f"Handoff {self.session_id}"

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp,
            "accomplished": list(self.accomplished),
            "blocked": list(self.blocked),
            "next_steps": list(self.next_steps),
            "context_snapshot": dict(self.context_snapshot),
            "open_tasks": list(self.open_tasks),
            "metadata": dict(self.metadata),
            "priority": self.priority,
            "schema_version": self.schema_version,
            "acknowledged_by": list(self.acknowledged_by),
            "stream_id": self.stream_id,
            "revision_id": self.revision_id,
            "title": self.title,
            "supersedes": list(self.supersedes),
            "closed": self.closed,
        }
        if self.target_agent is not None:
            result["target_agent"] = self.target_agent
        if self.artifact_path:
            result["artifact_path"] = self.artifact_path
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HandoffPacket:
        target_agent_raw = data.get("target_agent")
        target_agent = str(target_agent_raw) if target_agent_raw is not None else None
        metadata = data.get("metadata")
        context_snapshot = data.get("context_snapshot")
        return cls(
            session_id=str(data.get("session_id", data.get("revision_id", ""))),
            agent_name=str(data.get("agent_name", "")),
            timestamp=str(data.get("timestamp", "")),
            accomplished=_string_list(data.get("accomplished")),
            blocked=_string_list(data.get("blocked")),
            next_steps=_string_list(data.get("next_steps")),
            context_snapshot=dict(context_snapshot) if isinstance(context_snapshot, dict) else {},
            open_tasks=_dict_list(data.get("open_tasks")),
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
            target_agent=target_agent,
            priority=str(data.get("priority", "normal")).strip() or "normal",
            schema_version=str(data.get("schema_version", "1")).strip() or "1",
            acknowledged_by=_string_list(data.get("acknowledged_by")),
            stream_id=str(data.get("stream_id", "")),
            revision_id=str(data.get("revision_id", "")),
            title=str(data.get("title", "")),
            supersedes=_string_list(data.get("supersedes")),
            closed=bool(data.get("closed", False)),
            artifact_path=str(data.get("artifact_path", "")),
        )


@dataclass(frozen=True)
class HandoffStream:
    """A summary of one logical handoff stream."""

    stream_id: str
    title: str
    scope_id: str
    latest_revision_id: str
    revision_count: int
    closed: bool
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "title": self.title,
            "scope_id": self.scope_id,
            "latest_revision_id": self.latest_revision_id,
            "revision_count": self.revision_count,
            "closed": self.closed,
            "updated_at": self.updated_at,
        }


class HandoffStore:
    """Immutable handoff revisions under ``memory/projects/<scope>/handoffs``."""

    def __init__(
        self,
        context_path: Path,
        *,
        config: Any = None,
        scope_id: str | None = None,
        root_resolver: ArtifactRootResolver | None = None,
    ) -> None:
        self._context_path = context_path.expanduser().resolve()
        self._config = config
        self.scope_id = validate_scope_id(scope_id or infer_scope_id(self._context_path))
        resolver = root_resolver or default_artifact_root
        self._root = (
            resolver(
                self._context_path,
                scope_id=self.scope_id,
                collection="handoffs",
                config=config,
            )
            .expanduser()
            .resolve()
        )
        self._root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._codec = MarkdownArtifactCodec(self._root)
        self._events_path = self._root / "_events.jsonl"

        # v1/v2 used scratchpad/handoffs/<session>.json.  Keep it as a
        # read-through import surface and maintain the tiny manifest for old
        # clients, but all new revision content lives under memory.
        self._legacy_root = (
            resolve_mount_root(self._context_path, MountType.SCRATCHPAD, config=config) / "handoffs"
        )
        self._legacy_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._manifest_path = self._legacy_root / "_manifest.json"

    def create_revision(
        self,
        *,
        title: str,
        agent_name: str,
        stream_id: str | None = None,
        revision_id: str | None = None,
        supersedes: str | Sequence[str] | None = None,
        accomplished: list[str] | None = None,
        blocked: list[str] | None = None,
        next_steps: list[str] | None = None,
        context_snapshot: dict[str, Any] | None = None,
        open_tasks: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        target_agent: str | None = None,
        priority: str = "normal",
        project_id: str = "",
        task_id: str = "",
        sensitivity: str = "internal",
    ) -> HandoffPacket:
        """Create an immutable v3 revision, optionally continuing a stream."""

        return self._create_revision(
            title=title,
            agent_name=agent_name,
            stream_id=stream_id,
            revision_id=revision_id,
            supersedes=supersedes,
            accomplished=accomplished,
            blocked=blocked,
            next_steps=next_steps,
            context_snapshot=context_snapshot,
            open_tasks=open_tasks,
            metadata=metadata,
            target_agent=target_agent,
            priority=priority,
            project_id=project_id,
            task_id=task_id,
            sensitivity=sensitivity,
            schema_version=HANDOFF_SCHEMA_VERSION,
        )

    def _create_revision(
        self,
        *,
        title: str,
        agent_name: str,
        stream_id: str | None,
        revision_id: str | None,
        supersedes: str | Sequence[str] | None,
        accomplished: list[str] | None,
        blocked: list[str] | None,
        next_steps: list[str] | None,
        context_snapshot: dict[str, Any] | None,
        open_tasks: list[dict[str, Any]] | None,
        metadata: dict[str, Any] | None,
        target_agent: str | None,
        priority: str,
        project_id: str,
        task_id: str,
        sensitivity: str,
        schema_version: str,
    ) -> HandoffPacket:

        normalized_title = title.strip()
        if not normalized_title:
            raise ValueError("title is required for a handoff revision")
        normalized_agent = agent_name.strip()
        if not normalized_agent:
            raise ValueError("agent_name is required")
        normalized_supersedes = self._normalize_supersedes(supersedes)

        referenced: list[HandoffPacket] = []
        for previous_id in normalized_supersedes:
            previous = self.read(session_id=previous_id)
            if previous is None:
                raise ValueError(f"superseded revision does not exist: {previous_id}")
            referenced.append(previous)

        if stream_id is None and referenced:
            stream_id = referenced[0].stream_id
        normalized_stream_id = _safe_identifier(
            stream_id or uuid.uuid4().hex, field_name="stream_id"
        )
        for previous in referenced:
            if previous.stream_id != normalized_stream_id:
                raise ValueError("superseded revisions must belong to the same stream")

        normalized_revision_id = _safe_identifier(
            revision_id or uuid.uuid4().hex, field_name="revision_id"
        )
        if self.read(session_id=normalized_revision_id) is not None:
            raise FileExistsError(f"handoff revision already exists: {normalized_revision_id}")

        if self.list_revisions(normalized_stream_id):
            current = self.read(stream_id=normalized_stream_id)
            if current is not None and current.closed:
                raise ValueError(f"handoff stream is closed: {normalized_stream_id}")

        revision_claim = self._claim_revision_id(normalized_revision_id)

        packet = HandoffPacket(
            session_id=normalized_revision_id,
            revision_id=normalized_revision_id,
            stream_id=normalized_stream_id,
            title=normalized_title,
            agent_name=normalized_agent,
            timestamp=_now(),
            accomplished=list(accomplished or []),
            blocked=list(blocked or []),
            next_steps=list(next_steps or []),
            context_snapshot=dict(context_snapshot or {}),
            open_tasks=list(open_tasks or []),
            metadata=dict(metadata or {}),
            target_agent=target_agent,
            priority=priority,
            schema_version=schema_version,
            supersedes=normalized_supersedes,
        )
        try:
            artifact = self._write_revision(
                packet,
                project_id=project_id,
                task_id=task_id,
                sensitivity=sensitivity,
            )
        except BaseException:
            revision_claim.unlink(missing_ok=True)
            raise
        packet.artifact_path = str(artifact.path)
        self._update_legacy_manifest(packet.session_id)
        self._log_creation(packet)
        return packet

    def create(
        self,
        *,
        session_id: str | None = None,
        agent_name: str,
        accomplished: list[str] | None = None,
        blocked: list[str] | None = None,
        next_steps: list[str] | None = None,
        context_snapshot: dict[str, Any] | None = None,
        open_tasks: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        target_agent: str | None = None,
        priority: str = "normal",
        title: str | None = None,
        stream_id: str | None = None,
        supersedes: str | Sequence[str] | None = None,
    ) -> HandoffPacket:
        """Compatibility shim for the v1/v2 title-less creation API."""

        sid = _safe_identifier(session_id or uuid.uuid4().hex[:12], field_name="session_id")
        packet = self._create_revision(
            title=title or f"Handoff from {agent_name}",
            agent_name=agent_name,
            stream_id=stream_id,
            revision_id=sid,
            supersedes=supersedes,
            accomplished=accomplished,
            blocked=blocked,
            next_steps=next_steps,
            context_snapshot=context_snapshot,
            open_tasks=open_tasks,
            metadata=metadata,
            target_agent=target_agent,
            priority=priority,
            project_id="",
            task_id="",
            sensitivity="internal",
            schema_version=LEGACY_HANDOFF_SCHEMA_VERSION,
        )
        return packet

    def read(
        self,
        *,
        session_id: str | None = None,
        stream_id: str | None = None,
    ) -> HandoffPacket | None:
        if session_id is not None:
            try:
                safe_id = _safe_identifier(session_id, field_name="session_id")
            except ValueError:
                return None
            packet = self._read_revision(safe_id)
            if packet is None:
                packet = self._read_legacy(safe_id)
            return self._apply_events(packet) if packet is not None else None

        packets = self.list(stream_id=stream_id, limit=1)
        return packets[0] if packets else None

    def list(
        self,
        *,
        limit: int = 10,
        stream_id: str | None = None,
    ) -> list[HandoffPacket]:
        if limit <= 0:
            return []
        safe_stream: str | None = None
        if stream_id is not None:
            safe_stream = _safe_identifier(stream_id, field_name="stream_id")

        packets: dict[str, HandoffPacket] = {}
        for artifact in self._codec.iter_artifacts(kind="handoff"):
            packet = self._packet_from_artifact(artifact)
            if packet is None:
                continue
            packets[packet.revision_id] = packet
        legacy_ids = self._load_manifest()
        legacy_ids.extend(
            path.stem
            for path in self._legacy_root.glob("*.json")
            if path.name != self._manifest_path.name and _SAFE_ID_PATTERN.fullmatch(path.stem)
        )
        for legacy_id in dict.fromkeys(legacy_ids):
            if legacy_id in packets:
                continue
            packet = self._read_legacy(legacy_id)
            if packet is not None:
                packets[packet.revision_id] = packet

        selected = [
            self._apply_events(packet)
            for packet in packets.values()
            if safe_stream is None or packet.stream_id == safe_stream
        ]
        selected.sort(
            key=lambda packet: (_timestamp_key(packet.timestamp), packet.revision_id),
            reverse=True,
        )
        return selected[:limit]

    def list_revisions(self, stream_id: str) -> list[HandoffPacket]:
        return self.list(limit=1_000_000, stream_id=stream_id)

    def list_streams(self, *, limit: int = 100) -> list[HandoffStream]:
        grouped: dict[str, list[HandoffPacket]] = {}
        for packet in self.list(limit=1_000_000):
            grouped.setdefault(packet.stream_id, []).append(packet)
        streams: list[HandoffStream] = []
        for stream_id, revisions in grouped.items():
            latest = max(
                revisions,
                key=lambda packet: (_timestamp_key(packet.timestamp), packet.revision_id),
            )
            streams.append(
                HandoffStream(
                    stream_id=stream_id,
                    title=latest.title,
                    scope_id=self.scope_id,
                    latest_revision_id=latest.revision_id,
                    revision_count=len(revisions),
                    closed=latest.closed,
                    updated_at=latest.timestamp,
                )
            )
        streams.sort(key=lambda item: (item.updated_at, item.stream_id), reverse=True)
        return streams[: max(0, limit)]

    def pending_for_agent(self, name: str) -> list[HandoffPacket]:
        """Return unacknowledged, open revisions targeted at ``name``."""

        pending = [
            packet
            for packet in self.list(limit=1_000_000)
            if packet.target_agent == name
            and name not in packet.acknowledged_by
            and not packet.closed
        ]
        priority_order = {"critical": 0, "high": 1, "normal": 2, "low": 3}
        pending.sort(key=lambda packet: priority_order.get(packet.priority, 2))
        return pending

    def acknowledge(self, session_id: str, name: str) -> bool:
        packet = self.read(session_id=session_id)
        if packet is None:
            return False
        actor = name.strip()
        if not actor:
            raise ValueError("acknowledging name is required")
        if actor in packet.acknowledged_by:
            return True
        self._append_event(
            {
                "event_id": uuid.uuid4().hex,
                "kind": "acknowledged",
                "timestamp": _now(),
                "stream_id": packet.stream_id,
                "revision_id": packet.revision_id,
                "actor": actor,
            }
        )
        return True

    def close(self, identifier: str, *, actor: str, reason: str = "") -> bool:
        """Close the stream containing ``identifier`` without editing a revision."""

        packet = self.read(session_id=identifier)
        if packet is None:
            try:
                revisions = self.list_revisions(identifier)
            except ValueError:
                return False
            packet = revisions[0] if revisions else None
        if packet is None:
            return False
        normalized_actor = actor.strip()
        if not normalized_actor:
            raise ValueError("closing actor is required")
        if packet.closed:
            return True
        self._append_event(
            {
                "event_id": uuid.uuid4().hex,
                "kind": "closed",
                "timestamp": _now(),
                "stream_id": packet.stream_id,
                "revision_id": packet.revision_id,
                "actor": normalized_actor,
                "reason": reason.strip(),
            }
        )
        return True

    def _write_revision(
        self,
        packet: HandoffPacket,
        *,
        project_id: str,
        task_id: str,
        sensitivity: str,
    ) -> MarkdownArtifact:
        body = self._render_body(packet)
        return self._codec.create(
            kind="handoff",
            title=packet.title,
            body=body,
            scope_id=self.scope_id,
            project_id=project_id,
            task_id=task_id,
            agent_name=packet.agent_name,
            author_kind="agent",
            sensitivity=sensitivity,
            provenance={
                "source": "afs.handoff",
                "stream_id": packet.stream_id,
                "revision_id": packet.revision_id,
                "supersedes": packet.supersedes,
            },
            relative_dir=packet.stream_id,
        )

    @staticmethod
    def _render_body(packet: HandoffPacket) -> str:
        def section(name: str, values: Sequence[str]) -> list[str]:
            lines = [f"## {name}", ""]
            lines.extend(f"- {value}" for value in values)
            if not values:
                lines.append("_None._")
            lines.append("")
            return lines

        lines = [f"# {packet.title}", ""]
        lines.extend(section("Accomplished", packet.accomplished))
        lines.extend(section("Blocked", packet.blocked))
        lines.extend(section("Next steps", packet.next_steps))
        payload = json.dumps(
            packet.to_dict(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        encoded = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")
        lines.append(f"<!-- afs-handoff-payload:{encoded} -->")
        return "\n".join(lines)

    def _packet_from_artifact(self, artifact: MarkdownArtifact) -> HandoffPacket | None:
        match = _PAYLOAD_PATTERN.search(artifact.body)
        if match is None:
            return None
        encoded = match.group(1)
        encoded += "=" * (-len(encoded) % 4)
        try:
            data = json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        packet = HandoffPacket.from_dict(data)
        provenance = artifact.metadata.provenance or {}
        if (
            provenance.get("revision_id") != packet.revision_id
            or provenance.get("stream_id") != packet.stream_id
            or artifact.metadata.title != packet.title
            or artifact.metadata.scope_id != self.scope_id
        ):
            return None
        packet.artifact_path = str(artifact.path)
        return packet

    def _read_revision(self, revision_id: str) -> HandoffPacket | None:
        for artifact in self._codec.iter_artifacts(kind="handoff"):
            provenance = artifact.metadata.provenance or {}
            if provenance.get("revision_id") != revision_id:
                continue
            return self._packet_from_artifact(artifact)
        return None

    def _read_legacy(self, session_id: str) -> HandoffPacket | None:
        packet_path = self._legacy_root / f"{session_id}.json"
        try:
            packet_path.resolve().relative_to(self._legacy_root.resolve())
        except ValueError:
            return None
        if not packet_path.is_file():
            return None
        try:
            data = json.loads(packet_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(data, dict):
            return None
        packet = HandoffPacket.from_dict(data)
        packet.artifact_path = str(packet_path)
        return packet

    def _apply_events(self, packet: HandoffPacket) -> HandoffPacket:
        acknowledged = list(dict.fromkeys(packet.acknowledged_by))
        closed = packet.closed
        for event in self._load_events():
            kind = event.get("kind")
            if kind == "acknowledged" and event.get("revision_id") == packet.revision_id:
                actor = event.get("actor")
                if isinstance(actor, str) and actor and actor not in acknowledged:
                    acknowledged.append(actor)
            elif kind == "closed" and event.get("stream_id") == packet.stream_id:
                closed = True
        return replace(packet, acknowledged_by=acknowledged, closed=closed)

    def _load_events(self) -> list[dict[str, Any]]:
        if not self._events_path.exists():
            return []
        events: list[dict[str, Any]] = []
        try:
            with self._events_path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict):
                        events.append(item)
        except OSError:
            return []
        return events

    def _append_event(self, event: Mapping[str, Any]) -> None:
        payload = (json.dumps(dict(event), sort_keys=True) + "\n").encode("utf-8")
        if len(payload) > 64 * 1024:
            raise ValueError("handoff event exceeds 64 KiB")
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        with _EVENT_LOCK:
            fd = os.open(self._events_path, flags, 0o600)
            try:
                try:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_EX)
                except ImportError:  # pragma: no cover - non-POSIX fallback
                    pass
                view = memoryview(payload)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError("short append while writing handoff event")
                    view = view[written:]
                os.fsync(fd)
            finally:
                os.close(fd)

    def _claim_revision_id(self, revision_id: str) -> Path:
        claim_root = self._root / ".revision_ids"
        claim_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        claim_root = claim_root.resolve()
        try:
            claim_root.relative_to(self._root)
        except ValueError as exc:
            raise ValueError("revision id registry resolves outside the handoff root") from exc
        claim_path = claim_root / revision_id
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(claim_path, flags, 0o600)
        except FileExistsError as exc:
            raise FileExistsError(f"handoff revision already exists: {revision_id}") from exc
        os.close(fd)
        return claim_path

    @staticmethod
    def _normalize_supersedes(value: str | Sequence[str] | None) -> list[str]:
        if value is None:
            return []
        values = [value] if isinstance(value, str) else list(value)
        result: list[str] = []
        for item in values:
            if not isinstance(item, str):
                raise ValueError("supersedes entries must be revision identifiers")
            normalized = _safe_identifier(item, field_name="supersedes")
            if normalized not in result:
                result.append(normalized)
        return result

    def _load_manifest(self) -> list[str]:
        if not self._manifest_path.exists():
            return []
        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        if isinstance(data, list):
            return [str(item) for item in data if isinstance(item, str)]
        # Some experimental v2 manifests wrapped the ordered ids in a dict.
        if isinstance(data, dict):
            for key in ("sessions", "handoffs", "items", "ids"):
                value = data.get(key)
                if isinstance(value, list):
                    return [str(item) for item in value if isinstance(item, str)]
        return []

    def _update_legacy_manifest(self, session_id: str) -> None:
        lock_path = self._legacy_root / ".manifest.lock"
        with _MANIFEST_LOCK:
            lock_handle = lock_path.open("a+", encoding="utf-8")
            try:
                try:
                    import fcntl

                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                except ImportError:  # pragma: no cover - non-POSIX fallback
                    pass
                manifest = self._load_manifest()
                if session_id not in manifest:
                    manifest.append(session_id)
                temporary = self._legacy_root / f"._manifest.{uuid.uuid4().hex}.tmp"
                fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as handle:
                        json.dump(manifest, handle, indent=2)
                        handle.write("\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    temporary.replace(self._manifest_path)
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            finally:
                lock_handle.close()

    def _log_creation(self, packet: HandoffPacket) -> None:
        try:
            from .history import log_session_event

            log_session_event(
                "handoff",
                session_id=packet.session_id,
                metadata={
                    "agent_name": packet.agent_name,
                    "stream_id": packet.stream_id,
                    "revision_id": packet.revision_id,
                    "accomplished_count": len(packet.accomplished),
                    "blocked_count": len(packet.blocked),
                    "target_agent": packet.target_agent or "",
                    "priority": packet.priority,
                },
                context_root=self._context_path,
            )
        except Exception:
            pass


__all__ = [
    "HANDOFF_SCHEMA_VERSION",
    "HandoffPacket",
    "HandoffStore",
    "HandoffStream",
]
