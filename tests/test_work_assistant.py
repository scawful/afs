from __future__ import annotations

from pathlib import Path

from afs.personal_context import load_personal_context
from afs.work_assistant import WorkAssistantStore, enrich_logged_event


def test_store_tracks_people_relationships_reviewers_and_approvals(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    context_root.mkdir()
    store = WorkAssistantStore(context_root)

    person_id = store.upsert_person(
        {
            "display_name": "Doc Owner",
            "email": "owner@example.com",
            "roles": ["owner"],
            "permissions": ["can review docs"],
        }
    )
    store.upsert_relationship(
        person_id=person_id,
        scope_type="doc",
        scope_id="doc-123",
        relationship_type="owner",
        allowed_review_targets=["docs"],
        permission_class="can_approve",
        provenance=[{"source": "test"}],
        confidence=0.9,
    )

    reviewers = store.suggest_reviewers(
        target_type="docs",
        scope_type="doc",
        scope_id="doc-123",
    )
    assert reviewers[0]["person_id"] == person_id
    assert reviewers[0]["reason"] == "owner for doc:doc-123"

    approval_id = store.create_approval(
        target_system="google-docs",
        target_id="doc-123",
        action="edit_doc",
        summary="Apply suggested intro edit",
        preview={"replace": "old", "with": "new"},
        affected_people=[person_id],
        permission_required="doc edit approval",
        requested_by="agent",
    )
    approvals = store.list_approvals()
    assert approvals[0]["approval_id"] == approval_id
    assert approvals[0]["preview"]["replace"] == "old"

    assert store.approve(approval_id, approved_by="human")
    assert store.list_approvals(status="pending") == []
    assert store.list_approvals(status="approved")[0]["approved_by"] == "human"

    sample_id = store.record_communication_sample(
        person_id=person_id,
        source_system="google-docs",
        source_id="doc-123",
        channel="design_doc",
        purpose="technical_requirements",
        text="Short, direct requirement with concrete acceptance criteria.",
        style_notes=["direct", "specific"],
        provenance=[{"source": "test"}],
        confidence=0.8,
    )
    samples = store.list_communication_samples(purpose="technical_requirements")
    assert samples[0]["sample_id"] == sample_id
    assert samples[0]["display_name"] == "Doc Owner"
    assert samples[0]["style_notes"] == ["direct", "specific"]
    style_summary = store.communication_style_summary(purpose="technical_requirements")
    assert style_summary["sample_count"] == 1
    assert style_summary["purposes"] == {"technical_requirements": 1}
    assert style_summary["style_notes"] == ["direct", "specific"]
    assert any("explicit approval" in line for line in style_summary["guidance"])
    assert store.summary()["communication_samples"] == 1


def test_enrich_logged_event_extracts_people_routes_approvals_and_activity(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    context_root.mkdir()

    counts = enrich_logged_event(
        context_root,
        {
            "id": "evt1",
            "type": "context",
            "source": "gws",
            "op": "snapshot",
            "timestamp": "2026-04-28T12:00:00+00:00",
            "metadata": {
                "target_system": "google-docs",
                "target_type": "docs",
                "target_id": "doc-123",
                "owner": {"display_name": "Doc Owner", "email": "owner@example.com"},
                "reviewers": [{"display_name": "Reviewer", "email": "reviewer@example.com"}],
                "approval_request": {
                    "action": "edit_doc",
                    "summary": "Apply suggested doc edit",
                    "permission_required": "doc edit approval",
                    "preview": {"diff": "-old\n+new"},
                },
                "communication_sample": {
                    "person": {"display_name": "Doc Owner", "email": "owner@example.com"},
                    "channel": "doc_comment",
                    "purpose": "responding_to_comments",
                    "text": "Let's keep this concrete: what changed, what was verified, and what's still risky.",
                    "style_notes": ["concrete", "findings-first"],
                },
            },
        },
    )

    assert counts["people"] == 2
    assert counts["relationships"] == 2
    assert counts["review_routes"] == 2
    assert counts["approvals"] == 1
    assert counts["communication_samples"] == 1
    assert counts["activity"] == 1

    store = WorkAssistantStore(context_root)
    people = store.list_people()
    assert {person["display_name"] for person in people} == {"Doc Owner", "Reviewer"}

    reviewers = store.suggest_reviewers(
        target_type="docs",
        scope_type="google-docs",
        scope_id="doc-123",
    )
    assert {reviewer["display_name"] for reviewer in reviewers} == {"Doc Owner", "Reviewer"}
    assert store.list_approvals()[0]["summary"] == "Apply suggested doc edit"
    samples = store.list_communication_samples()
    assert samples[0]["purpose"] == "responding_to_comments"
    assert "what changed" in samples[0]["text_excerpt"]
    assert store.list_activity()[0]["event_id"] == "evt1"


def test_communication_sample_explicit_person_wins_over_context_owner(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    context_root.mkdir()

    counts = enrich_logged_event(
        context_root,
        {
            "id": "evt-person-sample",
            "type": "context.log",
            "metadata": {
                "target_system": "google-docs",
                "target_type": "design_doc",
                "owner": {"display_name": "Doc Owner", "email": "owner@example.com"},
                "communication_sample": {
                    "person": {"display_name": "Comment Author", "email": "author@example.com"},
                    "purpose": "responding_to_comments",
                    "text": "Concrete, evidence-backed reply.",
                },
            },
        },
    )

    assert counts["people"] == 2
    store = WorkAssistantStore(context_root)
    samples = store.list_communication_samples()
    assert samples[0]["display_name"] == "Comment Author"


def test_communication_preflight_merges_style_personal_context_and_approvals(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    context_root.mkdir()
    personal_root = tmp_path / "personal"
    personal_root.mkdir()
    (personal_root / "profile.toml").write_text('name = "Test User"\n', encoding="utf-8")
    (personal_root / "samples.md").write_text(
        "Findings first. Exact evidence. Short follow-up.",
        encoding="utf-8",
    )
    (personal_root / "manifest.toml").write_text(
        """
[modes.work]
tone = "direct and specific"
work_context = true
load = ["samples.md"]
style_instructions = ["avoid generic corporate filler"]
communication_sources = ["approved PR comments"]
posting_policy = "Ask before posting externally."
""".strip()
        + "\n",
        encoding="utf-8",
    )

    store = WorkAssistantStore(context_root)
    store.record_communication_sample(
        source_system="github",
        source_id="comment-1",
        channel="pr_review",
        purpose="responding_to_comments",
        text="Concise reply with exact file evidence.",
        style_notes=["concise"],
    )
    approval_id = store.create_approval(
        target_system="github",
        target_id="PR-1",
        action="post_pr_comment",
        summary="Post drafted PR response",
    )
    personal = load_personal_context("work", context_root=personal_root)

    preflight = store.communication_preflight(
        purpose="responding_to_comments",
        personal_context=personal,
        context_path=context_root,
    )

    assert preflight["style"]["sample_count"] == 1
    assert preflight["personal_context"]["loaded"] is True
    assert preflight["personal_context"]["mode"] == "work"
    assert preflight["personal_context"]["style_instructions"] == [
        "avoid generic corporate filler"
    ]
    assert preflight["pending_approvals"][0]["approval_id"] == approval_id
    assert preflight["approval_guardrail"]["requires_explicit_approval"] is True
    assert preflight["approval_guardrail"]["ready_to_post"] is False
    assert preflight["missing_style_evidence"] is False
    assert any("Personal posting policy" in line for line in preflight["guidance"])


def test_communication_preflight_flags_missing_style_evidence(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    context_root.mkdir()

    preflight = WorkAssistantStore(context_root).communication_preflight()

    assert preflight["missing_style_evidence"] is True
    assert preflight["checklist"][0]["status"] == "not_loaded"
    assert preflight["checklist"][1]["status"] == "missing"
    assert any("Style evidence is missing" in line for line in preflight["guidance"])
