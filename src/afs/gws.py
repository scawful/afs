"""Google Workspace CLI (gws) integration for AFS.

Provides a Python interface to the `gws` binary for calendar, gmail, drive,
sheets, and other Google Workspace operations. All methods fail gracefully
when gws is not installed or not authenticated.

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
import shutil
import subprocess
from typing import Any


class GWSClient:
    """Thin wrapper around the `gws` CLI binary."""

    def __init__(self, binary: str | None = None):
        self._binary = binary or shutil.which("gws")

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

    def _run(self, args: list[str], timeout: int = 15) -> subprocess.CompletedProcess:
        """Run a gws command and return the raw result."""
        if not self._binary:
            raise RuntimeError("gws binary not found")
        return subprocess.run(
            [self._binary] + args,
            capture_output=True, text=True, timeout=timeout,
        )

    def _run_json(self, args: list[str], timeout: int = 15) -> Any:
        """Run a gws command and parse JSON output.

        Handles both single JSON objects and NDJSON streams.
        Returns None on any failure.
        """
        try:
            result = self._run(args, timeout=timeout)
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
        """Return gws auth status as a dict."""
        data = self._run_json(["auth", "status"])
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
        """Send an email via gws."""
        return self._run_json([
            "gmail", "+send",
            "--to", to,
            "--subject", subject,
            "--body", body,
            "--output-format", "json",
        ])

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
        """Run an arbitrary gws command and parse JSON output."""
        return self._run_json(list(args), timeout=timeout)


# Singleton for convenience
_default_client: GWSClient | None = None


def get_client() -> GWSClient:
    """Return a shared GWSClient instance."""
    global _default_client
    if _default_client is None:
        _default_client = GWSClient()
    return _default_client
