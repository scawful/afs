"""Content binding shared by AFS approval stores and extension senders."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any


def approval_content_hash(
    *, target_system: str, target_id: str, action: str, preview: Any
) -> str:
    """Hash the exact action envelope, preserving text and array order.

    Extensions must put final outgoing text, attachment digests and all delivery
    options in preview. Object key order is immaterial; Unicode, whitespace,
    destinations, message splitting and attachment order are material.
    """
    content = {
        "version": 1,
        "target_system": target_system,
        "target_id": target_id,
        "action": action,
        "preview": preview,
    }
    serialized = json.dumps(
        content, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def validate_approved_content(
    approval: dict[str, Any],
    *,
    target_system: str,
    target_id: str,
    action: str,
    preview: Any,
) -> str:
    """Require human authorization and compare the final dispatch envelope.

    Call immediately before sending, after every wrapper transformation. This
    validates content; execution claims and retries remain the store's job.
    """
    if approval.get("status") not in {"approved", "executing"} or approval.get(
        "human_confirmed"
    ) is not True:
        raise PermissionError("content approval is not human-confirmed and executable")
    digest = approval_content_hash(
        target_system=target_system, target_id=target_id, action=action, preview=preview
    )
    expected = approval.get("content_sha256")
    if not isinstance(expected, str) or not hmac.compare_digest(expected, digest):
        raise PermissionError("approved content differs from final dispatch payload")
    return digest
