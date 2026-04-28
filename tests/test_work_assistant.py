from __future__ import annotations

from pathlib import Path

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
            },
        },
    )

    assert counts["people"] == 2
    assert counts["relationships"] == 2
    assert counts["review_routes"] == 2
    assert counts["approvals"] == 1
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
    assert store.list_activity()[0]["event_id"] == "evt1"
