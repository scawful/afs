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
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
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
_STREAM_LOCK = threading.RLock()


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


def _strict_string_list(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return list(value)


def _strict_dict_list(value: Any, *, field_name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{field_name} must be a list of objects")
    return [dict(item) for item in value]


def _strict_dict(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return dict(value)


@dataclass(frozen=True)
class _EventState:
    acknowledged_by_revision: dict[str, tuple[str, ...]]
    closed_streams: frozenset[str]


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
    """Immutable handoff revisions under ``memory/projects/<scope>/handoffs``.

    The store validates scope grammar and filesystem containment.  Registry
    authorization remains the caller's boundary: callers accepting requester-
    supplied scopes must authorize them before constructing this store.
    """

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
        self._manifest_path = self._legacy_root / "_manifest.json"
        self._legacy_codec: MarkdownArtifactCodec | None = None
        if self.scope_id == "common":
            self._legacy_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._legacy_codec = MarkdownArtifactCodec(self._legacy_root)

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
        if not isinstance(title, str):
            raise ValueError("title must be a string")
        normalized_title = title.strip()
        if not normalized_title:
            raise ValueError("title is required for a handoff revision")
        if not isinstance(agent_name, str):
            raise ValueError("agent_name must be a string")
        normalized_agent = agent_name.strip()
        if not normalized_agent:
            raise ValueError("agent_name is required")
        normalized_supersedes = self._normalize_supersedes(supersedes)
        normalized_accomplished = _strict_string_list(
            accomplished if accomplished is not None else [], field_name="accomplished"
        )
        normalized_blocked = _strict_string_list(
            blocked if blocked is not None else [], field_name="blocked"
        )
        normalized_next_steps = _strict_string_list(
            next_steps if next_steps is not None else [], field_name="next_steps"
        )
        normalized_context = _strict_dict(
            context_snapshot if context_snapshot is not None else {},
            field_name="context_snapshot",
        )
        normalized_tasks = _strict_dict_list(
            open_tasks if open_tasks is not None else [], field_name="open_tasks"
        )
        normalized_metadata = _strict_dict(
            metadata if metadata is not None else {}, field_name="metadata"
        )
        if target_agent is not None and not isinstance(target_agent, str):
            raise ValueError("target_agent must be a string or null")
        if not isinstance(priority, str) or not priority.strip():
            raise ValueError("priority must be a non-empty string")

        with self._stream_transaction():
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

            revisions = self.list_revisions(normalized_stream_id)
            if revisions and revisions[0].closed:
                raise ValueError(f"handoff stream is closed: {normalized_stream_id}")

            revision_claim = self._claim_revision_id(normalized_revision_id)
            packet = HandoffPacket(
                session_id=normalized_revision_id,
                revision_id=normalized_revision_id,
                stream_id=normalized_stream_id,
                title=normalized_title,
                agent_name=normalized_agent,
                timestamp=_now(),
                accomplished=normalized_accomplished,
                blocked=normalized_blocked,
                next_steps=normalized_next_steps,
                context_snapshot=normalized_context,
                open_tasks=normalized_tasks,
                metadata=normalized_metadata,
                target_agent=target_agent,
                priority=priority.strip(),
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
                revision_claim.rollback()
                raise
            revision_claim.close()
            packet.artifact_path = str(artifact.path)

        self._update_legacy_compatibility_best_effort(packet)
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
            if packet is None and self.scope_id == "common":
                packet = self._read_legacy(safe_id)
            if packet is None:
                return None
            return self._apply_events(packet, self._load_event_state())

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
        if self.scope_id == "common":
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

        event_state = self._load_event_state()
        selected = [
            self._apply_events(packet, event_state)
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

        if not isinstance(actor, str):
            raise ValueError("closing actor must be a string")
        normalized_actor = actor.strip()
        if not normalized_actor:
            raise ValueError("closing actor is required")
        if not isinstance(reason, str):
            raise ValueError("closing reason must be a string")
        with self._stream_transaction():
            packet = self.read(session_id=identifier)
            if packet is None:
                try:
                    revisions = self.list_revisions(identifier)
                except ValueError:
                    return False
                packet = revisions[0] if revisions else None
            if packet is None:
                return False
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
        body_lines = artifact.body.rstrip("\n").splitlines()
        if not body_lines:
            return None
        match = _PAYLOAD_PATTERN.fullmatch(body_lines[-1])
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
        required = {
            "session_id",
            "agent_name",
            "timestamp",
            "accomplished",
            "blocked",
            "next_steps",
            "context_snapshot",
            "open_tasks",
            "metadata",
            "priority",
            "schema_version",
            "acknowledged_by",
            "stream_id",
            "revision_id",
            "title",
            "supersedes",
            "closed",
        }
        if required.difference(data):
            return None
        if data.get("schema_version") not in {
            HANDOFF_SCHEMA_VERSION,
            LEGACY_HANDOFF_SCHEMA_VERSION,
        }:
            return None
        string_fields = (
            "session_id",
            "agent_name",
            "timestamp",
            "priority",
            "stream_id",
            "revision_id",
            "title",
        )
        if not all(isinstance(data.get(name), str) for name in string_fields):
            return None
        if data.get("target_agent") is not None and not isinstance(data.get("target_agent"), str):
            return None
        embedded_path = data.get("artifact_path")
        if embedded_path is not None and embedded_path != "":
            return None
        try:
            accomplished = _strict_string_list(data.get("accomplished"), field_name="accomplished")
            blocked = _strict_string_list(data.get("blocked"), field_name="blocked")
            next_steps = _strict_string_list(data.get("next_steps"), field_name="next_steps")
            context_snapshot = _strict_dict(
                data.get("context_snapshot"), field_name="context_snapshot"
            )
            open_tasks = _strict_dict_list(data.get("open_tasks"), field_name="open_tasks")
            metadata = _strict_dict(data.get("metadata"), field_name="metadata")
            acknowledged_by = _strict_string_list(
                data.get("acknowledged_by"), field_name="acknowledged_by"
            )
            supersedes = _strict_string_list(data.get("supersedes"), field_name="supersedes")
            revision_id = _safe_identifier(str(data["revision_id"]), field_name="revision_id")
            stream_id = _safe_identifier(str(data["stream_id"]), field_name="stream_id")
            normalized_supersedes = self._normalize_supersedes(supersedes)
        except ValueError:
            return None
        if not isinstance(data.get("closed"), bool):
            return None
        if (
            data["session_id"] != revision_id
            or not str(data["agent_name"]).strip()
            or not str(data["title"]).strip()
            or not str(data["priority"]).strip()
            or bool(data["closed"])
            or acknowledged_by
        ):
            return None
        timestamp = str(data["timestamp"])
        timestamp_value = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp_value)
        except ValueError:
            return None
        if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() is None:
            return None
        packet = HandoffPacket.from_dict(data)
        packet.accomplished = accomplished
        packet.blocked = blocked
        packet.next_steps = next_steps
        packet.context_snapshot = context_snapshot
        packet.open_tasks = open_tasks
        packet.metadata = metadata
        provenance = artifact.metadata.provenance or {}
        if (
            provenance.get("source") != "afs.handoff"
            or provenance.get("revision_id") != revision_id
            or provenance.get("stream_id") != stream_id
            or provenance.get("supersedes") != normalized_supersedes
            or artifact.metadata.title != packet.title
            or artifact.metadata.agent_name != packet.agent_name
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
        if self.scope_id != "common" or self._legacy_codec is None:
            return None
        packet_path = self._legacy_root / f"{session_id}.json"
        try:
            _, payload = self._legacy_codec._read_contained_text(packet_path)
            data = json.loads(payload)
        except (json.JSONDecodeError, OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        packet = HandoffPacket.from_dict(data)
        packet.artifact_path = str(packet_path)
        return packet

    def _apply_events(self, packet: HandoffPacket, event_state: _EventState) -> HandoffPacket:
        acknowledged = list(dict.fromkeys(packet.acknowledged_by))
        for actor in event_state.acknowledged_by_revision.get(packet.revision_id, ()):
            if actor not in acknowledged:
                acknowledged.append(actor)
        closed = packet.closed or packet.stream_id in event_state.closed_streams
        return replace(packet, acknowledged_by=acknowledged, closed=closed)

    def _load_event_state(self) -> _EventState:
        acknowledged: dict[str, list[str]] = {}
        closed_streams: set[str] = set()
        for event in self._load_events():
            kind = event.get("kind")
            if kind == "acknowledged":
                revision_id = event.get("revision_id")
                actor = event.get("actor")
                if isinstance(revision_id, str) and isinstance(actor, str) and actor:
                    actors = acknowledged.setdefault(revision_id, [])
                    if actor not in actors:
                        actors.append(actor)
            elif kind == "closed":
                stream_id = event.get("stream_id")
                if isinstance(stream_id, str) and stream_id:
                    closed_streams.add(stream_id)
        return _EventState(
            acknowledged_by_revision={key: tuple(value) for key, value in acknowledged.items()},
            closed_streams=frozenset(closed_streams),
        )

    def _load_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        try:
            fd = self._codec._open_control_file("_events.jsonl", os.O_RDONLY)
            with os.fdopen(fd, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict):
                        events.append(item)
        except (OSError, ValueError):
            return []
        return events

    def _append_event(self, event: Mapping[str, Any]) -> None:
        payload = (json.dumps(dict(event), sort_keys=True) + "\n").encode("utf-8")
        if len(payload) > 64 * 1024:
            raise ValueError("handoff event exceeds 64 KiB")
        flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
        with _EVENT_LOCK:
            fd = self._codec._open_control_file("_events.jsonl", flags)
            locked = False
            try:
                self._lock_fd(fd)
                locked = True
                view = memoryview(payload)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError("short append while writing handoff event")
                    view = view[written:]
                os.fsync(fd)
            finally:
                if locked:
                    self._unlock_fd(fd)
                os.close(fd)

    def _claim_revision_id(self, revision_id: str) -> Any:
        try:
            return self._codec._claim_identifier(".revision_ids", revision_id)
        except FileExistsError as exc:
            raise FileExistsError(f"handoff revision already exists: {revision_id}") from exc

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
        if self.scope_id != "common" or self._legacy_codec is None:
            return []
        try:
            fd = self._legacy_codec._open_control_file("_manifest.json", os.O_RDONLY)
            with os.fdopen(fd, encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError, ValueError):
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

    def _update_legacy_compatibility_best_effort(self, packet: HandoffPacket) -> None:
        if self.scope_id != "common":
            return
        try:
            self._write_legacy_compatibility(packet)
        except Exception:
            # Canonical Markdown is the source of truth.  Compatibility is
            # derived state and must never turn a committed write into a
            # reported failure.
            pass

    def _write_legacy_compatibility(self, packet: HandoffPacket) -> None:
        if self._legacy_codec is None:  # pragma: no cover - guarded by caller
            return
        with _MANIFEST_LOCK:
            lock_fd = self._legacy_codec._open_control_file(
                ".manifest.lock", os.O_RDWR | os.O_CREAT
            )
            locked = False
            try:
                self._lock_fd(lock_fd)
                locked = True
                compatibility = packet.to_dict()
                compatibility.pop("artifact_path", None)
                rendered_packet = json.dumps(compatibility, indent=2, allow_nan=False) + "\n"
                legacy_fd = self._legacy_codec._open_relative_directory(None)
                try:
                    if legacy_fd is not None:
                        self._legacy_codec._write_exclusive_at(
                            legacy_fd,
                            f"{packet.session_id}.json",
                            rendered_packet,
                        )
                    else:
                        self._legacy_codec._write_exclusive(
                            self._legacy_root / f"{packet.session_id}.json",
                            rendered_packet,
                        )
                finally:
                    if legacy_fd is not None:
                        os.close(legacy_fd)

                manifest = self._load_manifest()
                if packet.session_id not in manifest:
                    manifest.append(packet.session_id)
                self._replace_legacy_manifest(manifest)
            finally:
                if locked:
                    self._unlock_fd(lock_fd)
                os.close(lock_fd)

    def _replace_legacy_manifest(self, manifest: list[str]) -> None:
        if self._legacy_codec is None:  # pragma: no cover - guarded by caller
            return
        payload = json.dumps(manifest, indent=2) + "\n"
        temporary_name = f"_manifest.{uuid.uuid4().hex}.tmp"
        legacy_fd = self._legacy_codec._open_relative_directory(None)
        if legacy_fd is not None:
            try:
                self._legacy_codec._write_exclusive_at(legacy_fd, temporary_name, payload)
                try:
                    os.replace(
                        temporary_name,
                        "_manifest.json",
                        src_dir_fd=legacy_fd,
                        dst_dir_fd=legacy_fd,
                    )
                    os.fsync(legacy_fd)
                except BaseException:
                    try:
                        os.unlink(temporary_name, dir_fd=legacy_fd)
                    except FileNotFoundError:
                        pass
                    raise
            finally:
                os.close(legacy_fd)
            return

        temporary = self._legacy_root / temporary_name
        self._legacy_codec._write_exclusive(temporary, payload)
        try:
            temporary.replace(self._manifest_path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    @contextmanager
    def _stream_transaction(self) -> Iterator[None]:
        with _STREAM_LOCK:
            fd = self._codec._open_control_file(".streams.lock", os.O_RDWR | os.O_CREAT)
            if os.name == "nt" and os.fstat(fd).st_size == 0:  # pragma: no cover - Windows
                os.write(fd, b"0")
                os.fsync(fd)
            locked = False
            try:
                self._lock_fd(fd)
                locked = True
                yield
            finally:
                if locked:
                    self._unlock_fd(fd)
                os.close(fd)

    @staticmethod
    def _lock_fd(fd: int) -> None:
        try:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - Windows
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)

    @staticmethod
    def _unlock_fd(fd: int) -> None:
        try:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)
        except ImportError:  # pragma: no cover - Windows
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

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
