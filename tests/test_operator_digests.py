from __future__ import annotations

from afs.operator_digests import digest_operator_output


def test_digest_operator_output_auto_detects_pytest_failures() -> None:
    payload = digest_operator_output(
        """
============================= test session starts =============================
FAILED tests/test_alpha.py::test_first - AssertionError: bad
FAILED tests/test_beta.py::test_second - ValueError: worse
========================= 88 passed, 2 failed in 4.21s ========================
""".strip()
    )

    assert payload["kind"] == "pytest"
    assert payload["details"]["counts"]["passed"] == 88
    assert payload["details"]["counts"]["failed"] == 2
    assert payload["details"]["failing_test_count"] == 2
    assert "pytest failed" in payload["summary"]


def test_digest_operator_output_auto_detects_traceback() -> None:
    payload = digest_operator_output(
        """
Traceback (most recent call last):
  File "/tmp/app.py", line 9, in main
    run()
  File "/tmp/lib.py", line 4, in run
    raise ValueError("boom")
ValueError: boom
""".strip()
    )

    assert payload["kind"] == "traceback"
    assert payload["summary"] == "ValueError: boom"
    assert payload["details"]["exception_type"] == "ValueError"
    assert payload["details"]["frame_count"] == 2
    assert payload["highlights"][1] == "/tmp/app.py:9 in main"


def test_digest_operator_output_auto_detects_grep_matches() -> None:
    payload = digest_operator_output(
        """
src/afs/context_pack.py:12:def build_context_pack():
src/afs/mcp_server.py:98:context pack handling lives here
""".strip()
    )

    assert payload["kind"] == "grep"
    assert payload["summary"] == "2 matches across 2 files"
    assert payload["details"]["match_count"] == 2
    assert payload["details"]["matches"][0]["line"] == 12


def test_digest_operator_output_auto_detects_diffstat() -> None:
    payload = digest_operator_output(
        """
 src/afs/context_pack.py | 12 +++++++++---
 src/afs/mcp_server.py   |  5 ++++-
 2 files changed, 14 insertions(+), 3 deletions(-)
""".strip()
    )

    assert payload["kind"] == "diffstat"
    assert payload["summary"] == "2 files changed, 14 insertions(+), 3 deletions(-)"
    assert payload["details"]["file_count"] == 2
    assert payload["details"]["insertions"] == 14
    assert payload["details"]["deletions"] == 3


def test_digest_operator_output_uses_generic_fallback() -> None:
    payload = digest_operator_output("line one\n\nline two")

    assert payload["kind"] == "generic"
    assert payload["summary"] == "line one"
    assert payload["highlights"] == ["line one", "line two"]
    assert payload["digest_text"].startswith("Kind: generic")
