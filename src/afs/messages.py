"""Scope-aware inter-agent messages with legacy bus compatibility.

``MessageBus`` is the public API.  The older implementation remains readable
through :mod:`afs.hivemind` for one compatibility cycle, but new callers must
select a project scope.  Reads include only that project and the shared
``common`` scope unless ``all_projects=True`` is explicitly requested.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from .hivemind import (
    HivemindBus,
    HivemindMessage,
    HivemindSubscription,
)

COMMON_SCOPE = "common"

# Language-neutral public names.  The legacy class names remain importable for
# existing integrations and persisted JSON records.
Message = HivemindMessage
Subscription = HivemindSubscription


def normalize_scope_id(value: str) -> str:
    """Return a validated scope identifier suitable for persisted metadata."""
    normalized = value.strip()
    if not normalized:
        raise ValueError("scope_id is required")
    if len(normalized) > 128:
        raise ValueError("scope_id must be at most 128 characters")
    if any(ch.isspace() or ch in "/\\" for ch in normalized):
        raise ValueError("scope_id cannot contain whitespace or path separators")
    return normalized


class MessageBus(HivemindBus):
    """Project-scoped view over the durable file message queue.

    ``scope_id`` is an authorization boundary, not only a ranking hint.  The
    default view may read its own scope and ``common``.  Reading every project
    requires the explicit ``all_projects`` constructor flag.
    """

    def __init__(
        self,
        context_path: Path,
        *,
        scope_id: str,
        config: Any = None,
        additional_scope_ids: Iterable[str] = (),
        all_projects: bool = False,
        include_legacy: bool = False,
    ) -> None:
        super().__init__(context_path, config=config)
        self.scope_id = normalize_scope_id(scope_id)
        self.all_projects = bool(all_projects)
        self.include_legacy = bool(include_legacy)
        self.allowed_scope_ids = {
            COMMON_SCOPE,
            self.scope_id,
            *(normalize_scope_id(item) for item in additional_scope_ids),
        }

    def _scope_allowed(self, message: Message) -> bool:
        if self.all_projects:
            return bool(message.scope_id) or self.include_legacy
        if not message.scope_id:
            return self.include_legacy
        return message.scope_id in self.allowed_scope_ids

    def send(
        self,
        from_agent: str,
        msg_type: str,
        payload: dict[str, Any] | None = None,
        *,
        to: str | None = None,
        topic: str | None = None,
        ttl_hours: int | None = None,
        scope_id: str | None = None,
    ) -> Message:
        effective_scope = normalize_scope_id(scope_id or self.scope_id)
        if not self.all_projects and effective_scope not in self.allowed_scope_ids:
            raise PermissionError(
                f"scope {effective_scope!r} is outside this message bus view"
            )
        return super().send(
            from_agent,
            msg_type,
            payload,
            to=to,
            topic=topic,
            ttl_hours=ttl_hours,
            scope_id=effective_scope,
        )

    def read(
        self,
        *,
        agent_name: str | None = None,
        since: datetime | None = None,
        msg_type: str | None = None,
        topic: str | None = None,
        limit: int = 50,
    ) -> list[Message]:
        messages = super().read(
            agent_name=agent_name,
            since=since,
            msg_type=msg_type,
            topic=topic,
            limit=0,
        )
        visible = [message for message in messages if self._scope_allowed(message)]
        return visible[-limit:] if limit and len(visible) > limit else visible


__all__ = [
    "COMMON_SCOPE",
    "Message",
    "MessageBus",
    "Subscription",
    "normalize_scope_id",
]
