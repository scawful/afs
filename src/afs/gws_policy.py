"""Google Workspace safety policy for AFS.

Classifies every GWS operation by risk tier and enforces guardrails so that
broad OAuth scopes don't translate into unchecked agent behaviour.

Risk tiers
----------
- **read**: Passive data retrieval — always allowed.
- **notify**: Low-side-effect actions like creating a draft — allowed but
  logged with extra context.
- **mutate**: State-changing actions (send email, create event, upload file,
  share, delete) — requires explicit confirmation unless the caller opts in
  to auto-approve via policy config.
- **dangerous**: Bulk/irreversible operations (delete, permission changes,
  forwarding rules) — always requires confirmation, never auto-approved.

Configuration lives in the AFS config TOML under ``[gws_policy]``.

Example::

    [gws_policy]
    enabled = true
    auto_approve_mutate = false
    rate_limit_sends_per_hour = 10
    blocked_recipients = ["*@competitor.com"]
    blocked_drive_actions = ["permissions.delete"]
    log_all = true
"""

from __future__ import annotations

import fnmatch
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# -------------------------------------------------------------------
# Risk classification
# -------------------------------------------------------------------

class RiskTier(str, Enum):
    READ = "read"
    NOTIFY = "notify"
    MUTATE = "mutate"
    DANGEROUS = "dangerous"


@dataclass(frozen=True)
class ActionClassification:
    """Result of classifying a GWS action."""

    tier: RiskTier
    service: str          # gmail, calendar, drive, sheets, …
    operation: str         # method name or gws sub-command
    description: str       # human-readable summary
    requires_confirm: bool


# Pattern → (tier, description template)
# Checked in order; first match wins.
_CLASSIFICATION_RULES: list[tuple[str, RiskTier, str]] = [
    # --- Gmail ---
    (r"gmail\s+.*\bdelete\b",                          RiskTier.DANGEROUS, "Delete Gmail message(s)"),
    (r"gmail\s+.*\bmodify\b.*removeLabelIds",          RiskTier.MUTATE,    "Modify Gmail labels"),
    (r"gmail\s+.*\btrash\b",                           RiskTier.DANGEROUS, "Trash Gmail message(s)"),
    (r"gmail\s+\+?send\b",                             RiskTier.MUTATE,    "Send email"),
    (r"gmail\s+\+?draft",                              RiskTier.NOTIFY,    "Create/update Gmail draft"),
    (r"gmail\s+.*\blist\b",                             RiskTier.READ,      "List Gmail messages"),
    (r"gmail\s+.*\bget\b",                              RiskTier.READ,      "Read Gmail message"),
    (r"gmail\s+\+?triage",                              RiskTier.READ,      "Gmail triage (read-only)"),
    (r"gmail\s+users\s+messages",                       RiskTier.READ,      "Gmail messages query"),
    (r"gmail\b",                                        RiskTier.READ,      "Gmail operation"),

    # --- Calendar ---
    (r"calendar\s+.*\bdelete\b",                        RiskTier.DANGEROUS, "Delete calendar event"),
    (r"calendar\s+\+?insert\b",                         RiskTier.MUTATE,    "Create calendar event"),
    (r"calendar\s+.*\bupdate\b",                        RiskTier.MUTATE,    "Update calendar event"),
    (r"calendar\s+.*\bpatch\b",                         RiskTier.MUTATE,    "Patch calendar event"),
    (r"calendar\s+\+?agenda\b",                         RiskTier.READ,      "View calendar agenda"),
    (r"calendar\s+.*\blist\b",                          RiskTier.READ,      "List calendar events"),
    (r"calendar\s+.*\bget\b",                           RiskTier.READ,      "Get calendar event"),
    (r"calendar\b",                                     RiskTier.READ,      "Calendar operation"),

    # --- Drive ---
    (r"drive\s+.*permissions\s+.*\bdelete\b",           RiskTier.DANGEROUS, "Remove Drive sharing permission"),
    (r"drive\s+.*permissions\s+.*\b(create|update)\b",  RiskTier.DANGEROUS, "Modify Drive sharing permissions"),
    (r"drive\s+.*\bdelete\b",                           RiskTier.DANGEROUS, "Delete Drive file"),
    (r"drive\s+files\s+create\b",                       RiskTier.MUTATE,    "Upload file to Drive"),
    (r"drive\s+files\s+update\b",                       RiskTier.MUTATE,    "Update Drive file"),
    (r"drive\s+files\s+copy\b",                         RiskTier.NOTIFY,    "Copy Drive file"),
    (r"drive\s+files\s+list\b",                         RiskTier.READ,      "List Drive files"),
    (r"drive\s+files\s+get\b",                          RiskTier.READ,      "Get Drive file metadata"),
    (r"drive\b",                                        RiskTier.READ,      "Drive operation"),

    # --- Sheets ---
    (r"sheets\s+.*\bclear\b",                           RiskTier.DANGEROUS, "Clear spreadsheet data"),
    (r"sheets\s+\+?append\b",                           RiskTier.MUTATE,    "Append to spreadsheet"),
    (r"sheets\s+.*\bupdate\b",                          RiskTier.MUTATE,    "Update spreadsheet"),
    (r"sheets\s+\+?read\b",                             RiskTier.READ,      "Read spreadsheet"),
    (r"sheets\b",                                       RiskTier.READ,      "Sheets operation"),
]


def classify_action(args: list[str]) -> ActionClassification:
    """Classify a raw gws command by risk tier."""
    command_str = " ".join(args)
    service = args[0] if args else "unknown"

    for pattern, tier, desc in _CLASSIFICATION_RULES:
        if re.search(pattern, command_str, re.IGNORECASE):
            requires_confirm = tier in (RiskTier.MUTATE, RiskTier.DANGEROUS)
            return ActionClassification(
                tier=tier,
                service=service,
                operation=command_str,
                description=desc,
                requires_confirm=requires_confirm,
            )

    # Default: treat unknown commands as mutate (fail-safe)
    return ActionClassification(
        tier=RiskTier.MUTATE,
        service=service,
        operation=command_str,
        description=f"Unclassified GWS operation: {command_str[:80]}",
        requires_confirm=True,
    )


# -------------------------------------------------------------------
# Rate limiting
# -------------------------------------------------------------------

class RateLimiter:
    """Sliding-window rate limiter for send-like operations."""

    def __init__(self, max_per_hour: int = 10):
        self.max_per_hour = max_per_hour
        self._timestamps: deque[float] = deque()

    def check(self) -> tuple[bool, str]:
        """Return (allowed, reason)."""
        now = time.monotonic()
        cutoff = now - 3600
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

        if len(self._timestamps) >= self.max_per_hour:
            return (False, f"Rate limit exceeded: {self.max_per_hour} sends/hour")
        return (True, "")

    def record(self) -> None:
        self._timestamps.append(time.monotonic())


# -------------------------------------------------------------------
# Recipient filtering
# -------------------------------------------------------------------

def check_recipient(to: str, blocked_patterns: list[str]) -> tuple[bool, str]:
    """Check if a recipient is allowed."""
    for pattern in blocked_patterns:
        if fnmatch.fnmatch(to.lower(), pattern.lower()):
            return (False, f"Recipient {to} matches blocked pattern: {pattern}")
    return (True, "")


# -------------------------------------------------------------------
# Policy configuration
# -------------------------------------------------------------------

@dataclass
class GWSPolicyConfig:
    """Runtime policy settings, loadable from TOML."""

    enabled: bool = True
    auto_approve_mutate: bool = False
    auto_approve_notify: bool = True
    rate_limit_sends_per_hour: int = 10
    blocked_recipients: list[str] = field(default_factory=list)
    blocked_drive_actions: list[str] = field(default_factory=list)
    log_all: bool = True
    # Which services are allowed at all (empty = all allowed)
    allowed_services: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GWSPolicyConfig:
        """Create from a config dict (e.g. TOML section)."""
        return cls(
            enabled=data.get("enabled", True),
            auto_approve_mutate=data.get("auto_approve_mutate", False),
            auto_approve_notify=data.get("auto_approve_notify", True),
            rate_limit_sends_per_hour=data.get("rate_limit_sends_per_hour", 10),
            blocked_recipients=data.get("blocked_recipients", []),
            blocked_drive_actions=data.get("blocked_drive_actions", []),
            log_all=data.get("log_all", True),
            allowed_services=data.get("allowed_services", []),
        )


# -------------------------------------------------------------------
# Policy enforcer
# -------------------------------------------------------------------

@dataclass
class PolicyDecision:
    """Result of a policy check."""

    allowed: bool
    reason: str
    classification: ActionClassification
    requires_confirm: bool = False
    # If True, the caller should present the action to the user for approval
    # before executing. The action was not blocked by policy, but needs a
    # human in the loop.
    needs_human_approval: bool = False


class GWSPolicyEnforcer:
    """Evaluates GWS actions against the configured policy.

    Designed to sit between the AFS MCP layer (or CLI) and the GWSClient
    so that every operation is classified, checked, rate-limited, and logged
    before it reaches the gws binary.
    """

    def __init__(
        self,
        config: GWSPolicyConfig | None = None,
        audit_callback: Callable[[str, ActionClassification, PolicyDecision], None] | None = None,
    ):
        self.config = config or GWSPolicyConfig()
        self._send_limiter = RateLimiter(self.config.rate_limit_sends_per_hour)
        self._audit_callback = audit_callback

    def evaluate(self, args: list[str], *, caller: str = "unknown") -> PolicyDecision:
        """Evaluate a gws command against policy.

        Returns a PolicyDecision indicating whether the action is allowed,
        needs human approval, or is blocked outright.
        """
        classification = classify_action(args)

        if not self.config.enabled:
            return PolicyDecision(
                allowed=True,
                reason="Policy enforcement disabled",
                classification=classification,
            )

        # --- Service allowlist ---
        if self.config.allowed_services:
            if classification.service not in self.config.allowed_services:
                decision = PolicyDecision(
                    allowed=False,
                    reason=f"Service '{classification.service}' not in allowed list: {self.config.allowed_services}",
                    classification=classification,
                )
                self._audit("BLOCKED", classification, decision, caller)
                return decision

        # --- Blocked Drive actions ---
        normalized_operation = _normalize_action_text(classification.operation)
        for blocked in self.config.blocked_drive_actions:
            if _normalize_action_text(blocked) in normalized_operation:
                decision = PolicyDecision(
                    allowed=False,
                    reason=f"Action matches blocked Drive action: {blocked}",
                    classification=classification,
                )
                self._audit("BLOCKED", classification, decision, caller)
                return decision

        # --- Read tier: always allowed ---
        if classification.tier == RiskTier.READ:
            decision = PolicyDecision(
                allowed=True,
                reason="Read operations are always allowed",
                classification=classification,
            )
            if self.config.log_all:
                self._audit("ALLOWED", classification, decision, caller)
            return decision

        # --- Notify tier ---
        if classification.tier == RiskTier.NOTIFY:
            decision = PolicyDecision(
                allowed=True,
                reason="Notify-tier action",
                classification=classification,
                needs_human_approval=not self.config.auto_approve_notify,
            )
            self._audit("ALLOWED", classification, decision, caller)
            return decision

        # --- Mutate tier ---
        if classification.tier == RiskTier.MUTATE:
            # Rate-limit sends
            if "send" in classification.operation.lower():
                ok, msg = self._send_limiter.check()
                if not ok:
                    decision = PolicyDecision(
                        allowed=False,
                        reason=msg,
                        classification=classification,
                    )
                    self._audit("RATE_LIMITED", classification, decision, caller)
                    return decision
                # Will record after actual execution

            decision = PolicyDecision(
                allowed=True,
                reason="Mutate-tier action — confirmation required" if not self.config.auto_approve_mutate else "Mutate-tier auto-approved",
                classification=classification,
                requires_confirm=not self.config.auto_approve_mutate,
                needs_human_approval=not self.config.auto_approve_mutate,
            )
            self._audit("PENDING_CONFIRM" if decision.needs_human_approval else "ALLOWED", classification, decision, caller)
            return decision

        # --- Dangerous tier: always requires confirmation ---
        decision = PolicyDecision(
            allowed=True,
            reason="Dangerous-tier action — human confirmation always required",
            classification=classification,
            requires_confirm=True,
            needs_human_approval=True,
        )
        self._audit("PENDING_CONFIRM", classification, decision, caller)
        return decision

    def record_send(self) -> None:
        """Record a successful send for rate limiting."""
        self._send_limiter.record()

    def check_recipient(self, to: str) -> tuple[bool, str]:
        """Check if a recipient is allowed by policy."""
        return check_recipient(to, self.config.blocked_recipients)

    def _audit(
        self,
        outcome: str,
        classification: ActionClassification,
        decision: PolicyDecision,
        caller: str,
    ) -> None:
        if self._audit_callback:
            self._audit_callback(outcome, classification, decision)


def _normalize_action_text(value: str) -> str:
    """Canonicalize action text so dotted config entries match spaced commands."""
    return re.sub(r"[^a-z0-9]+", "", value.lower())
