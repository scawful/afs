from __future__ import annotations

from types import SimpleNamespace

from afs.gws import GWSClient


class _FakeClient(GWSClient):
    def __init__(self, stdout: str, *, returncode: int = 0) -> None:
        self._binary = "gws"
        self._stdout = stdout
        self._returncode = returncode

    def _run(self, args: list[str], timeout: int = 15):
        return SimpleNamespace(returncode=self._returncode, stdout=self._stdout)


def test_gws_client_run_json_parses_ndjson() -> None:
    client = _FakeClient('{"kind":"a"}\n{"kind":"b"}\n')

    result = client.raw("calendar", "agenda")

    assert result == [{"kind": "a"}, {"kind": "b"}]


def test_gws_client_calendar_agenda_accepts_wrapped_items() -> None:
    client = _FakeClient('{"items":[{"summary":"Sync"},{"summary":"Review"}]}\n')

    result = client.calendar_agenda()

    assert [item["summary"] for item in result] == ["Sync", "Review"]


def test_gws_client_gmail_unread_accepts_message_list() -> None:
    client = _FakeClient('{"messages":[{"id":"m1"},{"id":"m2"}]}\n')

    result = client.gmail_unread(max_results=1)

    assert result == [{"id": "m1"}]
