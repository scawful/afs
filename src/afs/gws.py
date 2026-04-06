"""Google Workspace CLI (gws) integration for AFS.

Provides a Python interface to the `gws` binary for calendar, gmail, drive,
sheets, and other Google Workspace operations. All methods fail gracefully
when gws is not installed or not authenticated.

Every operation is routed through :class:`~afs.gws_policy.GWSPolicyEnforcer`
so that read/notify/mutate/dangerous actions are classified, rate-limited,
and optionally gated behind human confirmation.

Usage:
    from afs.gws import GWSClient

    gws = GWSClient()
    if gws.available:
        agenda = gws.calendar_agenda()
        unread = gws.gmail_unread()
        gws.gmail_send(to="someone@google.com", subject="hi", body="hello")
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from typing import Any

from .gws_policy import (
    GWSPolicyConfig,
    GWSPolicyEnforcer,
    PolicyDecision,
    classify_action,
)

logger = logging.getLogger(__name__)


class GWSPolicyError(Exception):
    """Raised when an operation is blocked by GWS policy."""

    def __init__(self, decision: PolicyDecision):
        self.decision = decision
        super().__init__(
            f"GWS policy blocked: {decision.reason} "
            f"[tier={decision.classification.tier.value}, "
            f"op={decision.classification.description}]"
        )


class GWSConfirmationRequired(Exception):
    """Raised when an operation needs human approval before execution."""

    def __init__(self, decision: PolicyDecision):
        self.decision = decision
        super().__init__(
            f"GWS action requires confirmation: {decision.classification.description} "
            f"[tier={decision.classification.tier.value}]"
        )


class GWSClient:
    """Thin wrapper around the ``gws`` CLI binary with policy enforcement.

    Parameters
    ----------
    binary : str | None
        Path to the ``gws`` binary.  Resolved via ``shutil.which`` when *None*.
    policy_config : GWSPolicyConfig | None
        Override the default safety policy.
    confirm_callback : callable | None
        Called when an action needs human approval.  Receives a
        :class:`PolicyDecision` and should return *True* to proceed or
        *False* to abort.  When *None*, :class:`GWSConfirmationRequired` is
        raised instead.
    """

    def __init__(
        self,
        binary: str | None = None,
        policy_config: GWSPolicyConfig | None = None,
        confirm_callback: Any | None = None,
    ):
        self._binary = binary or shutil.which("gws")
        self._policy = GWSPolicyEnforcer(
            config=policy_config,
            audit_callback=self._audit_log,
        )
        self._confirm_callback = confirm_callback

    # ------------------------------------------------------------------
    # Policy helpers
    # ------------------------------------------------------------------

    @property
    def policy(self) -> GWSPolicyEnforcer:
        """Access the policy enforcer (e.g. for introspection or testing)."""
        return self._policy

    def _enforce_policy(self, args: list[str], *, caller: str = "afs") -> PolicyDecision:
        """Run policy checks and handle confirmation flow.

        Returns the decision if allowed.
        Raises GWSPolicyError if blocked.
        Raises GWSConfirmationRequired if human approval needed and no callback.
        """
        decision = self._policy.evaluate(args, caller=caller)

        if not decision.allowed:
            logger.warning("GWS policy blocked: %s — %s", decision.classification.description, decision.reason)
            raise GWSPolicyError(decision)

        if decision.needs_human_approval:
            if self._confirm_callback:
                approved = self._confirm_callback(decision)
                if not approved:
                    logger.info("GWS action declined by user: %s", decision.classification.description)
                    raise GWSPolicyError(PolicyDecision(
                        allowed=False,
                        reason="User declined confirmation",
                        classification=decision.classification,
                    ))
                logger.info("GWS action approved by user: %s", decision.classification.description)
            else:
                raise GWSConfirmationRequired(decision)

        return decision

    @staticmethod
    def _audit_log(outcome: str, classification: Any, decision: Any) -> None:
        logger.info(
            "GWS audit: %s | tier=%s | service=%s | op=%s | reason=%s",
            outcome,
            classification.tier.value,
            classification.service,
            classification.description,
            decision.reason,
        )

    @property
    def available(self) -> bool:
        """True if the gws binary is on PATH."""
        return self._binary is not None

    @property
    def authenticated(self) -> bool:
        """True if gws has valid credentials."""
        if not self.available:
            return False
        status = self.auth_status()
        return status.get("auth_method", "none") != "none"

    def _run(
        self,
        args: list[str],
        timeout: int = 15,
        *,
        _skip_policy: bool = False,
    ) -> subprocess.CompletedProcess:
        """Run a gws command and return the raw result.

        Policy enforcement happens here unless ``_skip_policy`` is set
        (used internally for auth-status checks that must not recurse).
        """
        if not self._binary:
            raise RuntimeError("gws binary not found")
        if not _skip_policy:
            self._enforce_policy(args)
        return subprocess.run(
            [self._binary] + args,
            capture_output=True, text=True, timeout=timeout,
        )

    def _run_json(
        self,
        args: list[str],
        timeout: int = 15,
        *,
        _skip_policy: bool = False,
    ) -> Any:
        """Run a gws command and parse JSON output.

        Handles both single JSON objects and NDJSON streams.
        Returns None on any failure.
        """
        try:
            result = self._run(args, timeout=timeout, _skip_policy=_skip_policy)
            if result.returncode != 0:
                return None
            output = result.stdout.strip()
            if not output:
                return None
            lines = output.splitlines()
            if len(lines) == 1:
                return json.loads(lines[0])
            items = []
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return items if items else None
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            return None

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def auth_status(self) -> dict[str, Any]:
        """Return gws auth status as a dict.

        Auth introspection bypasses policy — it is not a Workspace API call.
        """
        data = self._run_json(["auth", "status"], _skip_policy=True)
        return data if isinstance(data, dict) else {}

    # ------------------------------------------------------------------
    # Calendar
    # ------------------------------------------------------------------

    def calendar_agenda(self, max_events: int = 10) -> list[dict[str, Any]]:
        """Fetch today's calendar agenda."""
        data = self._run_json(["calendar", "+agenda", "--output-format", "json"])
        if isinstance(data, list):
            return data[:max_events]
        if isinstance(data, dict):
            items = data.get("items", data.get("events", []))
            return items[:max_events] if isinstance(items, list) else []
        return []

    def calendar_event_create(
        self, summary: str, start: str, end: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        """Create a calendar event. start/end are ISO 8601 datetimes."""
        event = {"summary": summary, "start": {"dateTime": start}, "end": {"dateTime": end}}
        event.update(kwargs)
        return self._run_json([
            "calendar", "+insert", "--json", json.dumps(event), "--output-format", "json",
        ])

    # ------------------------------------------------------------------
    # Gmail
    # ------------------------------------------------------------------

    def gmail_unread(self, max_results: int = 5, query: str = "is:unread category:primary") -> list[dict[str, Any]]:
        """Fetch recent unread messages from primary inbox."""
        data = self._run_json([
            "gmail", "users", "messages", "list",
            "--params", json.dumps({"userId": "me", "q": query, "maxResults": max_results}),
            "--output-format", "json",
        ])
        if isinstance(data, dict):
            messages = data.get("messages", [])
            return messages[:max_results] if isinstance(messages, list) else []
        if isinstance(data, list):
            return data[:max_results]
        return []

    def gmail_send(self, to: str, subject: str, body: str) -> dict[str, Any] | None:
        """Send an email via gws.

        Subject to recipient filtering and per-hour rate limiting.
        """
        # Recipient check happens before the normal policy evaluation
        ok, reason = self._policy.check_recipient(to)
        if not ok:
            raise GWSPolicyError(PolicyDecision(
                allowed=False,
                reason=reason,
                classification=classify_action(["gmail", "+send"]),
            ))

        result = self._run_json([
            "gmail", "+send",
            "--to", to,
            "--subject", subject,
            "--body", body,
            "--output-format", "json",
        ])
        # Record successful send for rate limiting
        if result is not None:
            self._policy.record_send()
        return result

    def gmail_triage(self) -> Any:
        """Run gmail triage helper."""
        return self._run_json(["gmail", "+triage", "--output-format", "json"])

    # ------------------------------------------------------------------
    # Drive
    # ------------------------------------------------------------------

    def drive_list(self, query: str = "", max_results: int = 10) -> list[dict[str, Any]]:
        """List Drive files matching a query."""
        params: dict[str, Any] = {"pageSize": max_results}
        if query:
            params["q"] = query
        data = self._run_json([
            "drive", "files", "list",
            "--params", json.dumps(params),
            "--output-format", "json",
        ])
        if isinstance(data, dict):
            files = data.get("files", [])
            return files[:max_results] if isinstance(files, list) else []
        if isinstance(data, list):
            return data[:max_results]
        return []

    def drive_upload(self, file_path: str, name: str | None = None) -> dict[str, Any] | None:
        """Upload a file to Drive."""
        args = ["drive", "files", "create", "--upload", file_path, "--output-format", "json"]
        if name:
            args.extend(["--json", json.dumps({"name": name})])
        return self._run_json(args)

    # ------------------------------------------------------------------
    # Sheets
    # ------------------------------------------------------------------

    def sheets_read(self, spreadsheet_id: str, range: str = "Sheet1") -> Any:
        """Read data from a spreadsheet."""
        return self._run_json([
            "sheets", "+read",
            "--spreadsheet-id", spreadsheet_id,
            "--range", range,
            "--output-format", "json",
        ])

    def sheets_append(self, spreadsheet_id: str, range: str, values: list[list[Any]]) -> Any:
        """Append rows to a spreadsheet."""
        return self._run_json([
            "sheets", "+append",
            "--spreadsheet-id", spreadsheet_id,
            "--range", range,
            "--json", json.dumps({"values": values}),
            "--output-format", "json",
        ])

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def raw(self, *args: str, timeout: int = 15) -> dict[str, Any] | list | None:
        """Run an arbitrary gws command and parse JSON output.

        This is the most dangerous entry point — full policy enforcement
        applies including classification, rate limiting, and confirmation.
        """
        return self._run_json(list(args), timeout=timeout)


# Singleton for convenience
_default_client: GWSClient | None = None


def get_client(policy_config: GWSPolicyConfig | None = None) -> GWSClient:
    """Return a shared GWSClient instance.

    On first call, attempts to load ``[gws_policy]`` from AFS config TOML.
    Pass *policy_config* to override.
    """
    global _default_client
    if _default_client is None:
        if policy_config is None:
            policy_config = _load_policy_from_config()
        _default_client = GWSClient(policy_config=policy_config)
    return _default_client


def _load_policy_from_config() -> GWSPolicyConfig | None:
    """Try to load gws_policy section from AFS config."""
    try:
        from .config import load_config
        cfg = load_config()
        section = cfg.get("gws_policy")
        if isinstance(section, dict):
            return GWSPolicyConfig.from_dict(section)
    except Exception:
        pass
    return None
