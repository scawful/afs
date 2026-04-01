"""Tests for the tether-bridge agent module."""

from __future__ import annotations

import json
from pathlib import Path

from afs.agents import list_agents
from afs.agents import tether_bridge as bridge


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_tether_bridge_is_registered() -> None:
    names = [spec.name for spec in list_agents()]
    assert "tether-bridge" in names


def test_agent_name_and_description() -> None:
    assert bridge.AGENT_NAME == "tether-bridge"
    assert bridge.AGENT_DESCRIPTION


# ---------------------------------------------------------------------------
# findings_to_captures
# ---------------------------------------------------------------------------

def test_findings_to_captures_basic() -> None:
    findings = [
        {
            "type": "dirty_repo",
            "name": "my-project",
            "dirty_file_count": 15,
            "agent": "workspace-analyst",
            "path": "/home/user/src/my-project",
        },
        {
            "type": "stale_repo",
            "name": "old-lib",
            "days_stale": 120,
            "agent": "workspace-analyst",
        },
        {
            "type": "pending_approval",
            "name": "scribe-draft",
            "agent": "agent-supervisor",
            "detail": "Agent 'scribe-draft' is awaiting review",
        },
    ]
    captures = bridge.findings_to_captures(findings)
    assert len(captures) == 3

    # Every capture has required fields
    for capture in captures:
        assert "title" in capture
        assert "body" in capture
        assert "priority" in capture
        assert "source" in capture
        assert "tags" in capture
        assert "entry_type" in capture
        assert "status" in capture
        assert "created_at" in capture
        assert "deep_link" in capture
        assert capture["status"] == "inbox"


def test_empty_findings() -> None:
    captures = bridge.findings_to_captures([])
    assert captures == []


# ---------------------------------------------------------------------------
# Priority mapping
# ---------------------------------------------------------------------------

def test_priority_dirty_repo_high() -> None:
    finding = {"type": "dirty_repo", "dirty_file_count": 15}
    assert bridge._classify_priority(finding) == "high"


def test_priority_dirty_repo_not_high_under_threshold() -> None:
    finding = {"type": "dirty_repo", "dirty_file_count": 5}
    # <= 10 dirty files is not high for dirty_repo, falls to default medium
    assert bridge._classify_priority(finding) == "medium"


def test_priority_stale_repo_low() -> None:
    finding = {"type": "stale_repo"}
    assert bridge._classify_priority(finding) == "low"


def test_priority_pending_approval_medium() -> None:
    finding = {"type": "pending_approval"}
    assert bridge._classify_priority(finding) == "medium"


def test_priority_severity_fallback() -> None:
    assert bridge._classify_priority({"severity": "critical"}) == "high"
    assert bridge._classify_priority({"severity": "high"}) == "high"
    assert bridge._classify_priority({"severity": "low"}) == "low"
    assert bridge._classify_priority({}) == "medium"


# ---------------------------------------------------------------------------
# Title and body generation
# ---------------------------------------------------------------------------

def test_title_dirty_repo() -> None:
    finding = {"type": "dirty_repo", "name": "myrepo", "dirty_file_count": 7}
    title = bridge._build_title(finding)
    assert "Dirty repo" in title
    assert "myrepo" in title
    assert "7" in title


def test_title_stale_repo() -> None:
    finding = {"type": "stale_repo", "name": "oldlib", "days_stale": 90}
    title = bridge._build_title(finding)
    assert "Stale repo" in title
    assert "oldlib" in title
    assert "90" in title


def test_title_pending_approval() -> None:
    finding = {"type": "pending_approval", "name": "my-agent"}
    title = bridge._build_title(finding)
    assert "Pending approval" in title
    assert "my-agent" in title


def test_title_generic_fallback() -> None:
    finding = {"type": "unknown_type", "title": "Something happened"}
    title = bridge._build_title(finding)
    assert title == "Something happened"


def test_title_bare_minimum() -> None:
    title = bridge._build_title({})
    assert "unknown" in title.lower() or "finding" in title.lower()


def test_body_includes_path_and_type() -> None:
    finding = {
        "type": "dirty_repo",
        "path": "/home/user/src/foo",
        "description": "Has uncommitted changes",
    }
    body = bridge._build_body(finding)
    assert body is not None
    assert "/home/user/src/foo" in body
    assert "dirty_repo" in body


def test_body_empty_finding() -> None:
    body = bridge._build_body({})
    assert body is None


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

def test_tags_include_agent_and_type() -> None:
    finding = {"type": "dirty_repo", "agent": "workspace-analyst"}
    tags = bridge._build_tags(finding)
    assert "afs-agent" in tags
    assert "dirty-repo" in tags
    assert "from:workspace-analyst" in tags


def test_tags_merge_existing() -> None:
    finding = {"type": "test", "tags": ["custom-tag"]}
    tags = bridge._build_tags(finding)
    assert "custom-tag" in tags
    assert "afs-agent" in tags


# ---------------------------------------------------------------------------
# Entry type inference
# ---------------------------------------------------------------------------

def test_entry_type_actionable() -> None:
    for t in ("dirty_repo", "pending_approval", "missing_context"):
        assert bridge._infer_entry_type({"type": t}) == "task"


def test_entry_type_informational() -> None:
    for t in ("stale_repo", "health_report", "summary"):
        assert bridge._infer_entry_type({"type": t}) == "note"


def test_entry_type_default() -> None:
    assert bridge._infer_entry_type({}) == "task"


# ---------------------------------------------------------------------------
# Deep links
# ---------------------------------------------------------------------------

def test_generate_deep_link() -> None:
    link = bridge.generate_deep_link("Dirty repo: foo (15 files)")
    assert link.startswith("tether://capture?title=")
    assert "Dirty" in link or "Dirty%20" in link or "Dirty" in link


def test_deep_link_encoding() -> None:
    link = bridge.generate_deep_link("Has spaces & symbols!")
    assert "tether://capture?title=" in link
    # Spaces and special chars should be percent-encoded
    assert " " not in link.split("?title=")[1]


# ---------------------------------------------------------------------------
# Output file format
# ---------------------------------------------------------------------------

def test_write_captures_file(tmp_path: Path) -> None:
    captures = [
        {
            "title": "Test capture",
            "body": "Some body",
            "priority": "medium",
            "source": "test-agent",
            "tags": ["afs-agent"],
            "entry_type": "task",
            "status": "inbox",
            "created_at": "2026-03-24T00:00:00",
            "deep_link": "tether://capture?title=Test%20capture",
        }
    ]
    output_path = tmp_path / "captures.json"
    result_path = bridge.write_captures_file(captures, output_path)
    assert result_path == output_path.resolve()
    assert output_path.exists()

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["agent"] == "tether-bridge"
    assert data["count"] == 1
    assert len(data["captures"]) == 1
    assert data["captures"][0]["title"] == "Test capture"


def test_write_captures_file_creates_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested" / "captures.json"
    bridge.write_captures_file([], nested)
    assert nested.exists()
    data = json.loads(nested.read_text(encoding="utf-8"))
    assert data["count"] == 0
    assert data["captures"] == []


# ---------------------------------------------------------------------------
# Read findings from reports
# ---------------------------------------------------------------------------

def test_read_latest_findings_empty(tmp_path: Path) -> None:
    findings = bridge._read_latest_findings(tmp_path)
    assert findings == []


def test_read_latest_findings_nonexistent(tmp_path: Path) -> None:
    findings = bridge._read_latest_findings(tmp_path / "does-not-exist")
    assert findings == []


def test_read_latest_findings_from_reports(tmp_path: Path) -> None:
    # Create a fake agent report
    report = {
        "name": "context-audit",
        "status": "warn",
        "payload": {
            "invalid": [
                {"name": "broken-repo", "path": "/src/broken", "missing": ["scratchpad"]},
            ],
        },
    }
    (tmp_path / "context-audit.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    findings = bridge._read_latest_findings(tmp_path)
    assert len(findings) == 1
    assert findings[0]["name"] == "broken-repo"
    assert findings[0]["agent"] == "context-audit"


def test_read_latest_findings_skips_own_output(tmp_path: Path) -> None:
    # The tether_captures.json file should be skipped
    (tmp_path / "tether_captures.json").write_text(
        json.dumps({"version": 1, "captures": [{"title": "skip me"}]}),
        encoding="utf-8",
    )
    findings = bridge._read_latest_findings(tmp_path)
    assert findings == []


def test_read_latest_findings_includes_supervisor_pending(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state = {
        "name": "scribe-draft",
        "state": "awaiting_review",
        "started_at": "2026-03-24T00:00:00",
    }
    (state_dir / "scribe-draft.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    findings = bridge._read_latest_findings(tmp_path)
    assert len(findings) == 1
    assert findings[0]["type"] == "pending_approval"
    assert findings[0]["name"] == "scribe-draft"


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------

def test_main_dry_run(tmp_path: Path, monkeypatch) -> None:
    """main() with --dry-run should exit 0 and not write captures file."""
    captures_path = tmp_path / "captures.json"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    # Put a fake report in the reports dir
    report = {
        "name": "test-agent",
        "status": "ok",
        "payload": {
            "findings": [
                {"type": "dirty_repo", "name": "proj", "dirty_file_count": 20},
            ],
        },
    }
    (reports_dir / "test-agent.json").write_text(
        json.dumps(report), encoding="utf-8"
    )

    exit_code = bridge.main([
        "--dry-run",
        "--captures-output", str(captures_path),
        "--reports-dir", str(reports_dir),
        "--stdout",
        "--quiet",
    ])
    assert exit_code == 0
    # Dry run should NOT write captures file
    assert not captures_path.exists()


def test_main_writes_captures(tmp_path: Path, monkeypatch) -> None:
    """main() without --dry-run should write captures file."""
    captures_path = tmp_path / "captures.json"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    report = {
        "name": "test-agent",
        "status": "warn",
        "payload": {
            "invalid": [
                {"type": "stale_repo", "name": "dusty", "days_stale": 200},
            ],
        },
    }
    (reports_dir / "test-agent.json").write_text(
        json.dumps(report), encoding="utf-8"
    )

    exit_code = bridge.main([
        "--captures-output", str(captures_path),
        "--reports-dir", str(reports_dir),
        "--quiet",
    ])
    assert exit_code == 0
    assert captures_path.exists()

    data = json.loads(captures_path.read_text(encoding="utf-8"))
    assert data["count"] == 1
    assert data["captures"][0]["priority"] == "low"  # stale_repo = low


def test_main_no_findings(tmp_path: Path) -> None:
    """main() with empty reports dir should still exit 0."""
    captures_path = tmp_path / "captures.json"
    reports_dir = tmp_path / "empty_reports"
    reports_dir.mkdir()

    exit_code = bridge.main([
        "--captures-output", str(captures_path),
        "--reports-dir", str(reports_dir),
        "--quiet",
    ])
    assert exit_code == 0
    # No captures to write, so file should not be created
    assert not captures_path.exists()


def test_main_output_report(tmp_path: Path) -> None:
    """main() with --output writes an agent report."""
    report_output = tmp_path / "report.json"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    exit_code = bridge.main([
        "--dry-run",
        "--reports-dir", str(reports_dir),
        "--output", str(report_output),
        "--quiet",
    ])
    assert exit_code == 0
    assert report_output.exists()

    data = json.loads(report_output.read_text(encoding="utf-8"))
    assert data["name"] == "tether-bridge"
    assert "metrics" in data
    assert "captures_generated" in data["metrics"]
