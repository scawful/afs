"""Tests for workspace analyst agent — repo discovery, git analysis, health reporting."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from afs.agents.workspace_analyst import (
    AGENT_NAME,
    RepoHealth,
    _analyze_repo,
    _find_repos,
    _run_git,
    _write_report,
    main,
)


@pytest.fixture(autouse=True)
def _isolated_agent_registry(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AFS_AGENT_REGISTRY_PATH", str(tmp_path / "agent_registry.json"))
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a real git repo for testing."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@test.com"],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test"],
        capture_output=True, check=True,
    )
    # Create initial commit
    (repo / "file.txt").write_text("hello", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "initial"],
        capture_output=True, check=True,
    )
    return repo


# ---------------------------------------------------------------------------
# _run_git
# ---------------------------------------------------------------------------


class TestRunGit:
    def test_valid_command(self, git_repo: Path) -> None:
        result = _run_git(git_repo, "status", "--porcelain")
        assert result is not None

    def test_invalid_repo(self, tmp_path: Path) -> None:
        result = _run_git(tmp_path, "status", "--porcelain")
        assert result is None

    def test_timeout_returns_none(self, git_repo: Path) -> None:
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("git", 10)):
            result = _run_git(git_repo, "status")
        assert result is None


# ---------------------------------------------------------------------------
# _analyze_repo
# ---------------------------------------------------------------------------


class TestAnalyzeRepo:
    def test_clean_repo(self, git_repo: Path) -> None:
        health = _analyze_repo(git_repo)
        assert health is not None
        assert health.name == "test-repo"
        assert health.status == "ok"
        assert health.uncommitted_files == 0
        assert health.untracked_files == 0

    def test_dirty_repo(self, git_repo: Path) -> None:
        (git_repo / "file.txt").write_text("modified", encoding="utf-8")
        health = _analyze_repo(git_repo)
        assert health is not None
        assert health.status == "dirty"
        assert health.uncommitted_files == 1

    def test_untracked_files(self, git_repo: Path) -> None:
        (git_repo / "new.txt").write_text("new", encoding="utf-8")
        health = _analyze_repo(git_repo)
        assert health is not None
        assert health.untracked_files == 1

    def test_non_git_dir_returns_none(self, tmp_path: Path) -> None:
        regular_dir = tmp_path / "not-a-repo"
        regular_dir.mkdir()
        health = _analyze_repo(regular_dir)
        assert health is None

    def test_branch_detection(self, git_repo: Path) -> None:
        health = _analyze_repo(git_repo)
        assert health is not None
        # Default branch could be main or master depending on git config
        assert health.branch in ("main", "master")

    def test_stale_branch_detection(self, git_repo: Path) -> None:
        # Create and merge a feature branch
        subprocess.run(
            ["git", "-C", str(git_repo), "checkout", "-b", "feature-done"],
            capture_output=True, check=True,
        )
        (git_repo / "feature.txt").write_text("feature", encoding="utf-8")
        subprocess.run(["git", "-C", str(git_repo), "add", "."], capture_output=True, check=True)
        subprocess.run(
            ["git", "-C", str(git_repo), "commit", "-m", "feature"],
            capture_output=True, check=True,
        )
        # Merge back to original branch
        main_branch = subprocess.run(
            ["git", "-C", str(git_repo), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip()
        subprocess.run(
            ["git", "-C", str(git_repo), "checkout", "-"],
            capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "-C", str(git_repo), "merge", "feature-done"],
            capture_output=True, check=True,
        )
        health = _analyze_repo(git_repo)
        assert health is not None
        assert "feature-done" in health.stale_branches

    def test_to_dict(self) -> None:
        health = RepoHealth(
            path="/some/path",
            name="test",
            status="dirty",
            branch="main",
            uncommitted_files=2,
        )
        d = health.to_dict()
        assert d["path"] == "/some/path"
        assert d["name"] == "test"
        assert d["status"] == "dirty"
        assert d["uncommitted_files"] == 2


# ---------------------------------------------------------------------------
# _find_repos
# ---------------------------------------------------------------------------


class TestFindRepos:
    def test_finds_direct_repo(self, git_repo: Path) -> None:
        repos = _find_repos([git_repo])
        assert len(repos) == 1
        assert repos[0] == git_repo

    def test_finds_nested_repos(self, tmp_path: Path) -> None:
        # Create repos at depth 1
        for name in ["repo-a", "repo-b"]:
            repo = tmp_path / "workspace" / name
            repo.mkdir(parents=True)
            subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)

        repos = _find_repos([tmp_path / "workspace"])
        assert len(repos) == 2
        names = {r.name for r in repos}
        assert names == {"repo-a", "repo-b"}

    def test_deduplicates_repos(self, git_repo: Path) -> None:
        repos = _find_repos([git_repo, git_repo])
        assert len(repos) == 1

    def test_missing_root_skipped(self, tmp_path: Path) -> None:
        repos = _find_repos([tmp_path / "nonexistent"])
        assert len(repos) == 0

    def test_max_depth_respected(self, tmp_path: Path) -> None:
        # Create a deeply nested repo
        deep_repo = tmp_path / "a" / "b" / "c" / "d" / "deep-repo"
        deep_repo.mkdir(parents=True)
        subprocess.run(["git", "init", str(deep_repo)], capture_output=True, check=True)

        repos = _find_repos([tmp_path], max_depth=2)
        assert len(repos) == 0

        repos = _find_repos([tmp_path], max_depth=5)
        assert len(repos) == 1


# ---------------------------------------------------------------------------
# _write_report
# ---------------------------------------------------------------------------


class TestWriteReport:
    def test_writes_json_report(self, tmp_path: Path) -> None:
        results = [
            RepoHealth(path="/a", name="repo-a", status="ok"),
            RepoHealth(path="/b", name="repo-b", status="dirty", uncommitted_files=3),
        ]
        report_path = _write_report(results, tmp_path / "output")
        assert report_path.exists()
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert data["summary"]["total_repos"] == 2
        assert data["summary"]["dirty"] == 1
        assert data["summary"]["ok"] == 1
        assert len(data["repos"]) == 2

    def test_attention_needed_filters(self, tmp_path: Path) -> None:
        results = [
            RepoHealth(path="/a", name="ok-repo", status="ok"),
            RepoHealth(path="/b", name="dirty-repo", status="dirty"),
            RepoHealth(path="/c", name="behind-repo", status="ok", behind=5),
            RepoHealth(
                path="/d", name="stale-branches", status="ok",
                stale_branches=["old-feature"],
            ),
        ]
        report_path = _write_report(results, tmp_path / "output")
        data = json.loads(report_path.read_text(encoding="utf-8"))
        attention = data["attention_needed"]
        attention_names = {r["name"] for r in attention}
        assert "ok-repo" not in attention_names
        assert "dirty-repo" in attention_names
        assert "behind-repo" in attention_names
        assert "stale-branches" in attention_names

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        output = tmp_path / "nested" / "deep" / "output"
        _write_report([], output)
        assert output.exists()


# ---------------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------------


class TestWorkspaceAnalystCLI:
    def test_empty_scan(self, tmp_path: Path) -> None:
        rc = main([
            "--scan-roots", str(tmp_path / "empty"),
            "--context-root", str(tmp_path / "context"),
            "--stdout", "--quiet",
        ])
        assert rc == 0

    def test_scan_with_repos(self, git_repo: Path, tmp_path: Path) -> None:
        context_root = tmp_path / "context"
        rc = main([
            "--scan-roots", str(git_repo.parent),
            "--context-root", str(context_root),
            "--stdout", "--quiet",
        ])
        assert rc == 0
        report = context_root / "scratchpad" / "afs_agents" / "workspace_health.json"
        assert report.exists()
        data = json.loads(report.read_text(encoding="utf-8"))
        assert data["summary"]["total_repos"] >= 1
