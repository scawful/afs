from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import afs.cli.insights as insights_cli
from afs.agents.guardrails import ApprovalGate
from afs.artifacts import NoteStore
from afs.cli import build_parser
from afs.context_layout import scaffold_v2
from afs.history import append_history_event, resolve_history_root
from afs.insights import InsightStore, reflect_evidence
from afs.project_registry import ProjectRegistry


def _workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, ProjectRegistry]:
    context = tmp_path / ".context"
    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir()
    other.mkdir()
    scaffold_v2(context)
    registry = ProjectRegistry(context)
    registry.register(project)
    registry.register(other)
    config = tmp_path / "afs.toml"
    config.write_text(
        f'[general]\ncontext_root = "{context}"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AFS_CONFIG_PATH", str(config))
    monkeypatch.setenv(
        "AFS_AGENT_APPROVALS_PATH",
        str(tmp_path / "approvals.json"),
    )
    return context, project, other, registry


def _append_failure(
    context: Path,
    *,
    scope_id: str,
    project_id: str,
    event_id: str,
) -> None:
    append_history_event(
        resolve_history_root(context),
        "session",
        "afs.session",
        op="turn_failed",
        event_id=event_id,
        context_root=context,
        metadata={
            "scope_id": scope_id,
            "project_id": project_id,
            "scope_attribution": "registry",
            "status": "failed",
            "prompt_preview": "must-never-enter-insight-evidence",
        },
    )


def _run(argv: list[str]) -> int:
    args = build_parser(argv).parse_args(argv)
    return args.func(args)


def test_research_terminal_output_collapses_untrusted_control_characters() -> None:
    rendered = insights_cli._terminal_text(
        "src/bad\n\x1b[2Jforged\u202e.py",
    )

    assert rendered == "src/bad [2Jforged .py"
    assert "\n" not in rendered
    assert "\x1b" not in rendered
    assert "\u202e" not in rendered


def test_insight_show_renders_full_bounded_candidate_without_truncation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context, project, _other, registry = _workspace(tmp_path, monkeypatch)
    record = registry.resolve(project)
    assert record is not None
    for index in (1, 2):
        _append_failure(
            context,
            scope_id=record.scope_id,
            project_id=record.project_id,
            event_id=f"unsafe-render-{index}",
        )
    store = InsightStore(
        context,
        scope_id=record.scope_id,
        requester_path=project,
    )
    packet = store.build_evidence_packet()
    candidate = reflect_evidence(packet)
    assert candidate is not None
    long_insight = "Evidence " + ("x" * 16_000) + " visible-tail"
    candidate["insight"] = long_insight
    artifact = store.create_candidate(candidate, evidence=packet)

    assert _run(["insights", "show", artifact.metadata.artifact_id, "--path", str(project)]) == 0
    shown = capsys.readouterr().out
    assert long_insight in shown
    assert "visible-tail" in shown


def test_unsafe_legacy_candidate_is_escaped_for_inspection_and_can_be_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context, project, _other, registry = _workspace(tmp_path, monkeypatch)
    record = registry.resolve(project)
    assert record is not None
    for index in (1, 2):
        _append_failure(
            context,
            scope_id=record.scope_id,
            project_id=record.project_id,
            event_id=f"unsafe-legacy-{index}",
        )
    store = InsightStore(
        context,
        scope_id=record.scope_id,
        requester_path=project,
    )
    packet = store.build_evidence_packet()
    candidate = reflect_evidence(packet)
    assert candidate is not None
    artifact = store.create_candidate(candidate, evidence=packet)
    artifact.path.write_text(
        artifact.path.read_text(encoding="utf-8")
        + "\x1b]8;;https://evil.example\x07forged link\n",
        encoding="utf-8",
    )

    show_args = ["insights", "show", artifact.metadata.artifact_id, "--path", str(project)]
    assert _run(show_args) == 2
    rendered = capsys.readouterr()
    assert rendered.out == ""
    assert "cannot be rendered safely" in rendered.err
    assert "--json" in rendered.err

    assert _run([*show_args, "--json"]) == 0
    escaped = capsys.readouterr().out
    assert "\\u001b" in escaped
    assert "\\u0007" in escaped
    assert "\x1b" not in escaped

    assert (
        _run(
            [
                "insights",
                "accept",
                artifact.metadata.artifact_id,
                "--path",
                str(project),
                "--because",
                "unsafe terminal content must never enter durable memory",
            ]
        )
        == 2
    )
    assert "Could not accept insight" in capsys.readouterr().err

    prompts: list[str] = []

    def confirm(prompt: str) -> str | None:
        prompts.append(prompt)
        match = re.search(r"Type '([^']+)'", prompt)
        return match.group(1) if match else None

    monkeypatch.setattr(insights_cli, "_TTY_READER", confirm)
    assert (
        _run(
            [
                "insights",
                "reject",
                artifact.metadata.artifact_id,
                "--path",
                str(project),
                "--because",
                "unsafe terminal content should be quarantined without promotion",
            ]
        )
        == 0
    )
    rejected = capsys.readouterr()
    assert "Warning: rejecting a candidate" in rejected.err
    assert prompts
    assert "\x1b" not in prompts[0]
    assert "\x07" not in prompts[0]
    assert store.show(artifact.metadata.artifact_id).status == "rejected"  # type: ignore[union-attr]


def test_insights_research_is_local_and_scoped_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _context, project, other, _registry = _workspace(tmp_path, monkeypatch)
    (project / "visible.md").write_text("research-marker visible", encoding="utf-8")
    (other / "private.md").write_text("research-marker private", encoding="utf-8")

    def forbidden_embedder(*_args, **_kwargs):
        raise AssertionError("local research must not construct an embedder")

    monkeypatch.setattr("afs.embeddings.create_embed_fn", forbidden_embedder)

    assert (
        _run(
            [
                "insights",
                "research",
                "research-marker",
                "--path",
                str(project),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    rendered = json.dumps(payload)
    assert payload["research_source"] == "local_code_and_context"
    assert payload["network_requested"] is False
    assert payload["embedding_requested"] is False
    assert "visible.md" in rendered
    assert "private.md" not in rendered


def test_insights_research_refreshes_changed_code_unless_reuse_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _context, project, _other, _registry = _workspace(tmp_path, monkeypatch)
    source = project / "changing.md"
    source.write_text("old-research-canary", encoding="utf-8")
    base = ["insights", "research", "old-research-canary", "--path", str(project), "--json"]
    assert _run(base) == 0
    assert "old-research-canary" in capsys.readouterr().out

    source.write_text("new-research-canary", encoding="utf-8")
    updated = [
        "insights",
        "research",
        "new-research-canary",
        "--path",
        str(project),
        "--json",
    ]
    assert _run(updated) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rebuilt"] is True
    assert "new-research-canary" in json.dumps(payload)


def test_insights_research_runs_only_selected_bounded_internet_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context, project, _other, _registry = _workspace(tmp_path, monkeypatch)
    extension = tmp_path / "afs_web"
    extension.mkdir()
    (extension / "__init__.py").write_text("", encoding="utf-8")
    sentinel = tmp_path / "forbidden-imported"
    shadow_sentinel = tmp_path / "shadow-runner-imported"
    shadow_runner = project / "afs" / "sources"
    shadow_runner.mkdir(parents=True)
    (project / "afs" / "__init__.py").write_text("", encoding="utf-8")
    (shadow_runner / "__init__.py").write_text("", encoding="utf-8")
    (shadow_runner / "research_runner.py").write_text(
        f"from pathlib import Path\nPath({str(shadow_sentinel)!r}).write_text('bad')\n",
        encoding="utf-8",
    )
    (extension / "forbidden.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('bad')\n",
        encoding="utf-8",
    )
    (extension / "provider.py").write_text(
        "print('provider import chatter')\n"
        "class Provider:\n"
        "    name = 'example_web'\n"
        "    kinds = ('doc',)\n"
        "    def status(self): return {'status': 'ok'}\n"
        "    def sync(self, *, query='', limit=50): return []\n"
        "    def research(self, request):\n"
        "        assert request.network_allowed is True\n"
        "        assert request.allowed_domains == ('example.com',)\n"
        "        return [{'id': 'web-1', 'kind': 'doc', "
        "'title': 'Bounded web evidence', "
        "'body': 'provider result', "
        "'uri': 'https://docs.example.com/guide'}]\n"
        "def register_context_source_provider():\n"
        "    print('provider factory chatter')\n"
        "    return Provider()\n",
        encoding="utf-8",
    )
    (extension / "extension.toml").write_text(
        'name = "afs_web"\n'
        '[[context_sources]]\nname = "example_web"\n'
        'module = "afs_web.provider"\n'
        '[[context_sources]]\nname = "forbidden_web"\n'
        'module = "afs_web.forbidden"\n',
        encoding="utf-8",
    )
    config = tmp_path / "afs.toml"
    config.write_text(
        f'[general]\ncontext_root = "{context}"\n'
        "[extensions]\n"
        'enabled_extensions = ["afs_web"]\n'
        f'extension_dirs = ["{tmp_path}"]\n'
        "auto_discover = false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AFS_CONFIG_PATH", str(config))

    assert (
        _run(
            [
                "insights",
                "research",
                "official guidance",
                "--path",
                str(project),
                "--internet-provider",
                "example_web",
                "--allow-domain",
                "example.com",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["network_requested"] is True
    assert payload["remote_content_transmission_requested"] is True
    assert payload["internet"]["provider"] == "example_web"
    assert payload["internet"]["records"][0]["uri"] == (
        "https://docs.example.com/guide"
    )
    assert not sentinel.exists()
    assert not shadow_sentinel.exists()


def test_insights_reflect_writes_one_readable_scoped_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context, project, other, registry = _workspace(tmp_path, monkeypatch)
    record = registry.resolve(project)
    other_record = registry.resolve(other)
    assert record is not None and other_record is not None
    _append_failure(
        context,
        scope_id=record.scope_id,
        project_id=record.project_id,
        event_id="project-failure-1",
    )
    _append_failure(
        context,
        scope_id=record.scope_id,
        project_id=record.project_id,
        event_id="project-failure-2",
    )
    _append_failure(
        context,
        scope_id=other_record.scope_id,
        project_id=other_record.project_id,
        event_id="other-private-failure",
    )

    argv = ["insights", "reflect", "--path", str(project), "--json"]
    assert _run(argv) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "candidate_created"
    assert first["evidence"]["evidence_ids"] == [
        "project-failure-1",
        "project-failure-2",
    ]
    assert "other-private-failure" not in json.dumps(first)
    assert "must-never-enter-insight-evidence" not in json.dumps(first)
    path = Path(first["candidate"]["path"])
    assert path.parent == (
        context
        / "scratchpad"
        / "projects"
        / record.project_id
        / "insights"
        / "candidates"
    )
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{6}Z--[a-z0-9-]+--[0-9a-f]{10}\.md",
        path.name,
    )

    # Identical history produces the same candidate rather than scheduled spam.
    assert _run(argv) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["status"] == "candidate_existing"
    assert second["candidate"]["metadata"]["artifact_id"] == (
        first["candidate"]["metadata"]["artifact_id"]
    )
    assert second["bound_evidence_digest"] == first["bound_evidence_digest"]

    _append_failure(
        context,
        scope_id=record.scope_id,
        project_id=record.project_id,
        event_id="project-failure-3",
    )
    assert _run(argv) == 0
    rolling = json.loads(capsys.readouterr().out)
    assert rolling["status"] == "candidate_existing"
    assert rolling["candidate_status"] == "pending"
    assert rolling["bound_evidence_digest"] == first["bound_evidence_digest"]
    assert rolling["inspected_evidence_digest"] != rolling["bound_evidence_digest"]


def test_insight_accept_requires_rationale_and_human_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context, project, _other, registry = _workspace(tmp_path, monkeypatch)
    record = registry.resolve(project)
    assert record is not None
    for index in (1, 2):
        _append_failure(
            context,
            scope_id=record.scope_id,
            project_id=record.project_id,
            event_id=f"accept-failure-{index}",
        )
    assert _run(["insights", "reflect", "--path", str(project), "--json"]) == 0
    created = json.loads(capsys.readouterr().out)
    candidate_id = created["candidate"]["metadata"]["artifact_id"]

    assert (
        _run(["insights", "accept", candidate_id, "--path", str(project)])
        == 2
    )
    assert "rationale is required" in capsys.readouterr().err

    for unsafe_rationale, expected in (
        ("x" * 4097, "no more than 4096"),
        ("looks valid\x1b[2Jforged prompt", "control or formatting"),
    ):
        assert (
            _run(
                [
                    "insights",
                    "accept",
                    candidate_id,
                    "--path",
                    str(project),
                    "--because",
                    unsafe_rationale,
                ]
            )
            == 2
        )
        assert expected in capsys.readouterr().err
    assert ApprovalGate().all_requests() == []

    monkeypatch.setattr(insights_cli, "_TTY_READER", lambda _prompt: None)
    assert (
        _run(
            [
                "insights",
                "accept",
                candidate_id,
                "--path",
                str(project),
                "--because",
                "the repeated failures share the same attributed operation",
            ]
        )
        == 2
    )
    assert "interactive human confirmation" in capsys.readouterr().err
    store = InsightStore(
        context,
        scope_id=record.scope_id,
        requester_path=project,
    )
    assert store.show(candidate_id).status == "pending"  # type: ignore[union-attr]

    def confirm(prompt: str) -> str | None:
        match = re.search(r"Type '([^']+)'", prompt)
        return match.group(1) if match else None

    monkeypatch.setattr(insights_cli, "_TTY_READER", confirm)
    assert (
        _run(
            [
                "insights",
                "accept",
                candidate_id,
                "--path",
                str(project),
                "--because",
                "the repeated failures share the same attributed operation",
                "--json",
            ]
        )
        == 0
    )
    accepted = json.loads(capsys.readouterr().out)
    assert accepted["decision"] == "accept"
    assert "/memory/projects/" in accepted["path"]
    accepted_record = store.show(candidate_id)
    assert accepted_record is not None
    assert accepted_record.status == "accepted"
    assert accepted_record.review is not None
    assert accepted_record.review["rationale"] == (
        "the repeated failures share the same attributed operation"
    )
    assert accepted_record.review["authenticated"] is True
    assert accepted_record.review["human_confirmed"] is True
    assert accepted_record.review["via"] == "controlling_terminal"
    approval = next(
        request
        for request in ApprovalGate().all_requests()
        if request.request_id == accepted["review_request_id"]
    )
    assert approval.status == "approved"
    assert approval.action.startswith("insight_accept_")


def test_insight_reject_is_an_approved_human_operation_and_never_writes_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context, project, _other, registry = _workspace(tmp_path, monkeypatch)
    record = registry.resolve(project)
    assert record is not None
    for index in (1, 2):
        _append_failure(
            context,
            scope_id=record.scope_id,
            project_id=record.project_id,
            event_id=f"reject-failure-{index}",
        )
    assert _run(["insights", "reflect", "--path", str(project), "--json"]) == 0
    created = json.loads(capsys.readouterr().out)
    candidate_id = created["candidate"]["metadata"]["artifact_id"]

    def confirm(prompt: str) -> str | None:
        match = re.search(r"Type '([^']+)'", prompt)
        return match.group(1) if match else None

    monkeypatch.setattr(insights_cli, "_TTY_READER", confirm)
    assert (
        _run(
            [
                "insights",
                "reject",
                candidate_id,
                "--path",
                str(project),
                "--because",
                "frequency alone does not justify a durable lesson",
                "--json",
            ]
        )
        == 0
    )
    rejected = json.loads(capsys.readouterr().out)
    approval = next(
        request
        for request in ApprovalGate().all_requests()
        if request.request_id == rejected["review_request_id"]
    )
    assert approval.status == "approved"
    assert approval.action.startswith("insight_reject_")
    store = InsightStore(context, scope_id=record.scope_id, requester_path=project)
    rejected_record = store.show(candidate_id)
    assert rejected_record is not None
    assert rejected_record.status == "rejected"
    assert rejected_record.review is not None
    assert rejected_record.review["decision"] == "rejected"
    assert rejected_record.review["human_confirmed"] is True
    assert NoteStore(context, scope_id=record.scope_id).list() == []


def test_insight_review_fails_closed_when_candidate_changes_during_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context, project, _other, registry = _workspace(tmp_path, monkeypatch)
    record = registry.resolve(project)
    assert record is not None
    for index in (1, 2):
        _append_failure(
            context,
            scope_id=record.scope_id,
            project_id=record.project_id,
            event_id=f"mutation-failure-{index}",
        )
    assert _run(["insights", "reflect", "--path", str(project), "--json"]) == 0
    created = json.loads(capsys.readouterr().out)
    candidate_id = created["candidate"]["metadata"]["artifact_id"]
    candidate_path = Path(created["candidate"]["path"])

    def mutate_then_confirm(prompt: str) -> str | None:
        candidate_path.write_text(
            candidate_path.read_text(encoding="utf-8") + "\nchanged after display\n",
            encoding="utf-8",
        )
        match = re.search(r"Type '([^']+)'", prompt)
        return match.group(1) if match else None

    monkeypatch.setattr(insights_cli, "_TTY_READER", mutate_then_confirm)
    argv = [
        "insights",
        "accept",
        candidate_id,
        "--path",
        str(project),
        "--because",
        "the reviewed evidence supports this lesson",
    ]
    assert _run(argv) == 2
    assert "changed after review" in capsys.readouterr().err
    store = InsightStore(context, scope_id=record.scope_id, requester_path=project)
    current = store.show(candidate_id)
    assert current is not None and current.status == "pending"
    assert NoteStore(context, scope_id=record.scope_id).list() == []

    # The approved request is bound to the old content digest and cannot be
    # reused for the now-mutated candidate.
    monkeypatch.setattr(insights_cli, "_TTY_READER", lambda _prompt: None)
    assert _run(argv) == 2
    assert "interactive human confirmation" in capsys.readouterr().err


@pytest.mark.parametrize("failure_phase", ["after_archive", "after_decision"])
def test_insight_accept_repairs_crash_interrupted_lifecycle(
    failure_phase: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context, project, _other, registry = _workspace(tmp_path, monkeypatch)
    record = registry.resolve(project)
    assert record is not None
    for index in (1, 2):
        _append_failure(
            context,
            scope_id=record.scope_id,
            project_id=record.project_id,
            event_id=f"crash-{failure_phase}-{index}",
        )
    assert _run(["insights", "reflect", "--path", str(project), "--json"]) == 0
    created = json.loads(capsys.readouterr().out)
    candidate_id = created["candidate"]["metadata"]["artifact_id"]
    rationale = "repair the exact human-reviewed promotion after interruption"

    def confirm(prompt: str) -> str | None:
        match = re.search(r"Type '([^']+)'", prompt)
        return match.group(1) if match else None

    monkeypatch.setattr(insights_cli, "_TTY_READER", confirm)
    original_accept = InsightStore.accept
    original_create_note = InsightStore._create_accepted_note
    if failure_phase == "after_archive":
        def interrupted_accept(
            self,
            identifier,
            *,
            expected_digest,
            approval_gate=None,
            approval_request_id="",
        ):
            del approval_gate, approval_request_id
            current = self.show(identifier)
            assert current is not None and current.status == "pending"
            self._archive(
                current.artifact,
                destination="accepted",
                expected_digest=expected_digest,
            )
            raise RuntimeError("simulated hard stop after archive")

        monkeypatch.setattr(InsightStore, "accept", interrupted_accept)
    else:
        monkeypatch.setattr(
            InsightStore,
            "_create_accepted_note",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("simulated hard stop after decision")
            ),
        )

    argv = [
        "insights",
        "accept",
        candidate_id,
        "--path",
        str(project),
        "--because",
        rationale,
        "--json",
    ]
    with pytest.raises(RuntimeError, match="simulated hard stop"):
        _run(argv)
    interrupted = InsightStore(
        context,
        scope_id=record.scope_id,
        requester_path=project,
    ).show(candidate_id)
    assert interrupted is not None and interrupted.status == "accepted"
    assert NoteStore(context, scope_id=record.scope_id).list() == []

    monkeypatch.setattr(InsightStore, "accept", original_accept)
    monkeypatch.setattr(InsightStore, "_create_accepted_note", original_create_note)
    monkeypatch.setattr(
        insights_cli,
        "_TTY_READER",
        lambda _prompt: (_ for _ in ()).throw(
            AssertionError("crash retry must reuse the exact durable approval")
        ),
    )
    assert _run(argv) == 0
    repaired = json.loads(capsys.readouterr().out)
    assert repaired["decision"] == "accept"
    assert Path(repaired["path"]).is_file()
    assert len(NoteStore(context, scope_id=record.scope_id).list()) == 1


def test_common_reflection_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context, project, _other, registry = _workspace(tmp_path, monkeypatch)
    record = registry.resolve(project)
    assert record is not None
    for index in (1, 2):
        append_history_event(
            resolve_history_root(context),
            "session",
            "afs.session",
            op="turn_failed",
            event_id=f"common-failure-{index}",
            context_root=context,
            metadata={
                "scope_id": "common",
                "scope_attribution": "common",
                "status": "failed",
            },
        )

    assert _run(["insights", "reflect", "--path", str(project), "--json"]) == 0
    project_result = json.loads(capsys.readouterr().out)
    assert project_result["status"] == "no_candidate"

    assert (
        _run(
            [
                "insights",
                "reflect",
                "--path",
                str(project),
                "--common",
                "--json",
            ]
        )
        == 0
    )
    common_result = json.loads(capsys.readouterr().out)
    assert common_result["status"] == "candidate_created"
    assert common_result["evidence"]["scope_id"] == "common"


def test_insights_parser_never_exposes_all_projects() -> None:
    parser = build_parser(["insights", "research", "query"])
    with pytest.raises(SystemExit):
        parser.parse_args(["insights", "research", "query", "--all-projects"])
