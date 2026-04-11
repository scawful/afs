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


def test_digest_operator_output_auto_detects_tsc_diagnostics() -> None:
    payload = digest_operator_output(
        """
src/app.ts(12,5): error TS2322: Type 'string' is not assignable to type 'number'.
src/app.ts(18,1): error TS1005: ',' expected.
Found 2 errors in the same file, starting at: src/app.ts:12
""".strip()
    )

    assert payload["kind"] == "diagnostic"
    assert payload["summary"] == "2 errors across 1 file"
    assert payload["details"]["error_count"] == 2
    assert payload["details"]["path_count"] == 1
    assert payload["details"]["tools"] == ["tsc"]
    assert payload["details"]["entries"][0]["code"] == "TS2322"


def test_digest_operator_output_auto_detects_eslint_stylish_output() -> None:
    payload = digest_operator_output(
        """
/Users/scawful/src/lab/afs/extensions/vscode-afs/src/extension.ts
  10:5  error    Unexpected any. Specify a different type  @typescript-eslint/no-explicit-any
  14:1  warning  Missing return type on function           @typescript-eslint/explicit-function-return-type

✖ 2 problems (1 error, 1 warning)
""".strip()
    )

    assert payload["kind"] == "diagnostic"
    assert payload["summary"] == "1 error, 1 warning across 1 file"
    assert payload["details"]["error_count"] == 1
    assert payload["details"]["warning_count"] == 1
    assert payload["details"]["tools"] == ["eslint"]
    assert payload["details"]["entries"][0]["code"] == "@typescript-eslint/no-explicit-any"


def test_digest_operator_output_accepts_explicit_diagnostic_kind() -> None:
    payload = digest_operator_output(
        "src/afs/operator_digests.py:88:9: F401 `re` imported but unused",
        kind="diagnostic",
    )

    assert payload["kind"] == "diagnostic"
    assert payload["summary"] == "1 error across 1 file"
    assert payload["details"]["tools"] == ["ruff"]
    assert payload["details"]["entries"][0]["code"] == "F401"


def test_digest_operator_output_uses_generic_fallback() -> None:
    payload = digest_operator_output("line one\n\nline two")

    assert payload["kind"] == "generic"
    assert payload["summary"] == "line one"
    assert payload["highlights"] == ["line one", "line two"]
    assert payload["digest_text"].startswith("Kind: generic")
