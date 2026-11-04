from __future__ import annotations

import hashlib
import json
import subprocess
from argparse import Namespace

import pytest

from afs.worktree_report import load_lifecycle_records, parse_worktree_porcelain, worktree_report


@pytest.fixture
def episode(tmp_path, monkeypatch):
    monkeypatch.setattr("afs.worktree_report._owner_inactive", lambda pid: pid == 12345)
    repo, worktree = tmp_path / "repo", tmp_path / "worktree"
    repo.mkdir()

    def git(*args, cwd=repo):
        return subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=t@example.invalid",
                               "-C", str(cwd), *args], check=True, text=True, capture_output=True).stdout.strip()

    git("init", "-q", "-b", "main")
    (repo / "code.asm").write_text("old")
    git("add", ".")
    git("commit", "-qm", "base")
    git("worktree", "add", "-qb", "task", str(worktree))
    base = git("rev-parse", "main")
    handoff = tmp_path / "handoff.md"
    handoff.write_text("Retained evidence")

    def artifact(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    contract = {"repo": str(repo), "worktree": str(worktree), "branch": "task",
                "base_commit": base, "head_commit": base, "allowed_scope": ["code.asm"],
                "owner_state": "inactive", "owner_pid": 12345, "lease_path": str(tmp_path / "absent-lease"),
                "artifacts": [artifact(handoff)], "intended_exit": "cleanup-ready",
                "review_status": "pending", "adoption_status": "pending"}
    mission = {"mission_id": "mission_test", "owner": "test-owner", "status": "done",
               "metadata": {"worktree_exit": [contract]}}
    path = tmp_path / "mission.json"

    def report():
        path.write_text(json.dumps(mission))
        return worktree_report(repo, base_ref="main", mission_records=(path,))["worktrees"][0]

    return repo, worktree, git, contract, mission, path, report, artifact


def test_exact_merged_head_can_only_be_review_candidate(episode):
    repo, wt, git, contract, mission, path, report, artifact = episode
    row = report()
    assert row["exit_plan"]["ready_for_review"] is True
    assert row["exit_plan"]["execution_authorized"] is False
    assert "review" in row["candidate"]
    assert len(row["head"]) == 40
    assert row["ahead"] == row["behind"] == 0
    assert wt.is_dir() and git("rev-parse", "task") == contract["head_commit"]


@pytest.mark.parametrize("filename", ["untracked.asm", "ignored.log"])
def test_unique_untracked_and_ignored_content_blocks_cleanup(episode, filename):
    repo, wt, git, contract, mission, path, report, artifact = episode
    if filename.endswith("log"):
        (repo / ".git/info/exclude").write_text("*.log\n")
    (wt / filename).write_text("must survive")
    row = report()
    assert not row["candidate"]
    assert any("content must be preserved" in b for b in row["exit_plan"]["blockers"])
    assert (wt / filename).read_text() == "must survive"


@pytest.mark.parametrize("change", ["owner", "owner_pid", "base", "branch", "retention", "lease", "repo", "head"])
def test_missing_or_stale_exit_evidence_blocks(episode, change, monkeypatch):
    repo, wt, git, contract, mission, path, report, artifact = episode
    if change == "owner":
        mission["status"] = "active"
    elif change == "owner_pid":
        contract.pop("owner_pid")
    elif change in {"base", "head"}:
        contract[f"{change}_commit"] = "0" * 40
    elif change == "branch":
        contract["branch"] = "someone-else"
    elif change == "retention":
        contract["artifacts"][0]["sha256"] = "bad"
    elif change == "lease":
        monkeypatch.setattr("afs.leases.inspect", lambda _: {"status": "held"})
    else:
        contract["repo"] = str(repo.parent)
    assert not report()["exit_plan"]["ready_for_review"]


def test_deleted_upstream_is_not_merge_or_cleanup_proof(episode, monkeypatch):
    repo, wt, git, contract, mission, path, report, artifact = episode
    (wt / "code.asm").write_text("new")
    git("add", ".", cwd=wt)
    git("commit", "-qm", "unique", cwd=wt)
    contract["head_commit"] = git("rev-parse", "HEAD", cwd=wt)
    monkeypatch.setattr("afs.worktree_report._gone_branches", lambda _: {"task"})
    row = report()
    assert row["upstream_gone"] and not row["merged"] and not row["candidate"]
    assert any("preserve commits" in b for b in row["exit_plan"]["blockers"])


def test_merge_plan_checks_exact_verification_scope_and_conflicts(episode):
    repo, wt, git, contract, mission, path, report, artifact = episode
    (wt / "code.asm").write_text("new")
    git("add", ".", cwd=wt)
    git("commit", "-qm", "edit", cwd=wt)
    head = git("rev-parse", "HEAD", cwd=wt)
    contract.update(head_commit=head, intended_exit="merge-ready")
    evidence = repo.parent / "verification.json"
    receipt = {"base_commit": contract["base_commit"], "head_commit": head,
               "passed": True, "checks": ["focused tests"], "conflicts": "unknown"}
    evidence.write_text(json.dumps(receipt))
    contract["verification"] = artifact(evidence)
    assert not report()["exit_plan"]["ready_for_review"]
    receipt["conflicts"] = "none"
    evidence.write_text(json.dumps(receipt))
    contract["verification"] = artifact(evidence)
    row = report()
    assert row["ahead"] == 1 and row["changed_paths"] == ["code.asm"]
    assert row["exit_plan"]["ready_for_review"] and not row["candidate"]
    contract["allowed_scope"] = ["other.asm"]
    assert not report()["exit_plan"]["ready_for_review"]
    contract["allowed_scope"] = ["code.asm"]
    git("commit", "--allow-empty", "-qm", "target advanced")
    assert not report()["exit_plan"]["ready_for_review"]


def test_missing_and_locked_worktrees_retained(episode):
    repo, wt, git, contract, mission, path, report, artifact = episode
    git("worktree", "lock", str(wt))
    assert not report()["exit_plan"]["ready_for_review"]


def test_record_collisions_rejected_and_newline_paths_parse(episode):
    repo, wt, git, contract, mission, path, report, artifact = episode
    report()
    with pytest.raises(ValueError, match="Conflicting"):
        load_lifecycle_records((path, path))
    text = "worktree /a\nquoted\0HEAD abc\0branch refs/heads/main\0\0"
    assert parse_worktree_porcelain(text)[0].path == "/a\nquoted"


def test_failed_git_inspection_never_creates_candidate(episode, monkeypatch):
    from afs import worktree_report as module
    original = module._status
    monkeypatch.setattr(module, "_status", lambda _: (_ for _ in ()).throw(ValueError("unavailable")))
    row = episode[6]()
    assert row["inspection_errors"] and not row["exit_plan"]["ready_for_review"]
    monkeypatch.setattr(module, "_status", original)


def test_explicit_audit_uses_existing_history_with_start_and_result(episode, monkeypatch, capsys):
    from afs.cli.coordination import worktrees_report_command
    events = []
    monkeypatch.setattr("afs.history.log_session_event", lambda op, **kwargs: events.append((op, kwargs)) or "event")
    repo, wt, git, contract, mission, path, report, artifact = episode
    report()
    args = Namespace(repo=str(repo), older_than_days=7, base_ref="main", mission_record=[str(path)],
                     audit_context=str(repo.parent / "host-context"), all=True, json=True, limit=25)
    assert worktrees_report_command(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["dry_run"] is True
    assert [e[0] for e in events] == ["worktree_report_started", "worktree_report_completed"]
    assert events[-1][1]["metadata"]["report_sha256"] == hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()


def test_branch_without_worktree_is_visible_and_preserved(episode):
    repo, wt, git, contract, mission, path, report, artifact = episode
    git("branch", "retained")
    result = worktree_report(repo, base_ref="main")
    assert result["unattached_branches"][0]["branch"] == "retained"
    assert result["unattached_branches"][0]["intended_exit"] == "retained"
    assert git("rev-parse", "retained") == contract["head_commit"]


def test_report_failure_is_audited_without_claiming_success(episode, monkeypatch):
    from afs.cli.coordination import worktrees_report_command
    events = []
    monkeypatch.setattr("afs.history.log_session_event", lambda op, **kwargs: events.append(op) or "event")
    args = Namespace(repo=str(episode[0]/"missing"), older_than_days=7, base_ref="main", mission_record=[],
                     audit_context=str(episode[0].parent/"host-context"), all=True, json=True, limit=25)
    with pytest.raises(ValueError):
        worktrees_report_command(args)
    assert events == ["worktree_report_started", "worktree_report_failed"]


def test_owner_pid_probe_fails_closed(monkeypatch):
    from afs.worktree_report import _owner_inactive
    assert not _owner_inactive(None)
    assert not _owner_inactive(0)
    assert not _owner_inactive(True)
    monkeypatch.setattr("os.kill", lambda pid, signal: None)
    assert not _owner_inactive(123)
    def denied(pid, signal):
        raise PermissionError("denied")
    monkeypatch.setattr("os.kill", denied)
    assert not _owner_inactive(123)
    def absent(pid, signal):
        raise ProcessLookupError("gone")
    monkeypatch.setattr("os.kill", absent)
    assert _owner_inactive(123)


@pytest.mark.parametrize("data", [[], {"metadata": {}},
    {"mission_id": "mission_x", "owner": "owner", "status": "done", "metadata": []},
    {"mission_id": "mission_x", "owner": "owner", "status": "done",
     "metadata": {"worktree_exit": [{"worktree": "/tmp/example", "intended_exit": []}]}}])
def test_malformed_lifecycle_fails_closed(tmp_path, data):
    path = tmp_path / "mission.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_lifecycle_records((path,))


@pytest.mark.parametrize("flag,count", [("--skip-worktree", "skip_worktree"),
                                       ("--assume-unchanged", "assume_unchanged")])
def test_hidden_tracked_content_cannot_qualify_for_cleanup(episode, flag, count):
    repo, wt, git, contract, mission, path, report, artifact = episode
    git("update-index", flag, "code.asm", cwd=wt)
    (wt / "code.asm").write_text("unique work hidden from ordinary Git status")
    assert git("status", "--porcelain", cwd=wt) == ""
    row = report()
    assert row["dirty"] is None
    assert row["index_visibility"][count] == 1
    assert not row["exit_plan"]["ready_for_review"] and not row["candidate"]
    assert any("visibility flags" in b for b in row["exit_plan"]["blockers"])
    assert (wt / "code.asm").read_text().startswith("unique work")


@pytest.mark.parametrize("setting", ["core.sparseCheckout", "index.sparse", "core.ignoreStat"])
def test_sparse_or_visibility_assumption_state_blocks_even_clean_status(episode, setting):
    repo, wt, git, contract, mission, path, report, artifact = episode
    git("config", setting, "true", cwd=wt)
    row = report()
    assert row["index_visibility"]["settings"][setting] is True
    assert not row["exit_plan"]["ready_for_review"] and not row["candidate"]


def test_submodule_content_needs_separate_preservation_verification(episode):
    repo, wt, git, contract, mission, path, report, artifact = episode
    head = git("rev-parse", "HEAD", cwd=wt)
    git("update-index", "--add", "--cacheinfo", f"160000,{head},nested", cwd=wt)
    git("commit", "-qm", "gitlink", cwd=wt)
    git("merge", "--ff-only", "task")
    contract.update(base_commit=git("rev-parse", "main"), head_commit=git("rev-parse", "task"))
    nested = wt / "nested"
    nested.mkdir()
    git("init", "-q", cwd=nested)
    (nested / "unique.log").write_text("nested unique content")
    (nested / ".gitignore").write_text("*.log\n")
    git("config", "submodule.nested.ignore", "all", cwd=wt)
    row = report()
    assert row["index_visibility"]["submodules"] == 1
    assert not row["candidate"]
    assert any("Submodule content" in b for b in row["exit_plan"]["blockers"])
    assert (nested / "unique.log").read_text() == "nested unique content"


def test_rename_inventory_includes_out_of_scope_source_deletion(episode):
    repo, wt, git, contract, mission, path, report, artifact = episode
    git("mv", "code.asm", "allowed.asm", cwd=wt)
    git("commit", "-qm", "rename", cwd=wt)
    head = git("rev-parse", "HEAD", cwd=wt)
    contract.update(head_commit=head, intended_exit="merge-ready", allowed_scope=["allowed.asm"])
    evidence = repo.parent / "verification.json"
    evidence.write_text(json.dumps({"base_commit": contract["base_commit"], "head_commit": head,
                                   "passed": True, "checks": ["focused"], "conflicts": "none"}))
    contract["verification"] = artifact(evidence)
    row = report()
    assert set(row["changed_paths"]) == {"code.asm", "allowed.asm"}
    assert not row["exit_plan"]["ready_for_review"]
    assert "Committed changes outside declared scope" in row["exit_plan"]["blockers"]
    contract["allowed_scope"] = ["code.asm", "allowed.asm"]
    assert report()["exit_plan"]["ready_for_review"] is True


def test_json_human_limit_cannot_break_audit_payload_hash(episode, monkeypatch, capsys):
    from afs.cli.coordination import worktrees_report_command
    repo, wt, git, contract, mission, path, report, artifact = episode
    report()
    git("worktree", "add", "-qb", "second", str(repo.parent / "second"))
    events = []
    monkeypatch.setattr("afs.history.log_session_event", lambda op, **kw: events.append((op, kw)) or "event")
    args = Namespace(repo=str(repo), older_than_days=7, base_ref="main", mission_record=[str(path)],
                     audit_context=str(repo.parent / "host-context"), all=True, json=True, limit=1)
    assert worktrees_report_command(args) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["linked_count"] == len(emitted["worktrees"]) == 2
    assert hashlib.sha256(json.dumps(emitted, sort_keys=True).encode()).hexdigest() == events[-1][1]["metadata"]["report_sha256"]
