from __future__ import annotations

from afs.model_prompts import build_model_system_prompt


def test_build_model_system_prompt_includes_session_state_summary() -> None:
    prompt = build_model_system_prompt(
        base_prompt="Base behavior.",
        workflow="edit_fast",
        tool_profile="edit_and_verify",
        session_state={
            "project": "afs",
            "profile": "default",
            "scratchpad": {
                "state_text": "Investigating MCP registry split.",
                "deferred_text": "Follow up on prompt packing.",
            },
            "diff": {"available": True, "total_changes": 7},
            "memory": {
                "memory_manifest": [
                    {"topic": "tag:mcp", "entry_count": 3, "latest": "2026-04-01T00:00:00Z"},
                    {"topic": "domain:session", "entry_count": 2, "latest": "2026-03-31T00:00:00Z"},
                ]
            },
            "tasks": {"total": 2, "counts": {"pending": 2}},
            "handoff": {"available": True, "next_steps": ["Ship the MCP refactor."]},
        },
        pack_state={
            "available": True,
            "query": "mcp registry split",
            "task": "Wire prompt context into the harness.",
            "model": "codex",
            "workflow": "edit_fast",
            "tool_profile": "edit_and_verify",
            "pack_mode": "focused",
            "estimated_tokens": 420,
        },
        skills_state={
            "available": True,
            "matches": [
                {
                    "name": "agentic-context",
                    "score": 9,
                    "triggers": ["context", "harness"],
                    "enforcement": [
                        "Use the indexed context before asking for repeated repo state.",
                        "Keep the scratchpad current for handoff.",
                    ],
                    "verification": [
                        "Write the updated handoff note before ending the session.",
                    ],
                },
                {"name": "github:github", "score": 4, "triggers": ["repo"]},
            ],
        },
    )

    assert "Base behavior." in prompt
    assert "## Execution Profile" in prompt
    assert "## Session Pack" in prompt
    assert "Query focus: mcp registry split" in prompt
    assert "Task focus: Wire prompt context into the harness." in prompt
    assert "Pack settings: model=codex, workflow=edit_fast, tool_profile=edit_and_verify, pack_mode=focused" in prompt
    assert "CLI follow-up: `afs query <text> --path <workspace>`" in prompt
    assert "CLI rebuild: `afs index rebuild --path <workspace>`" in prompt
    assert "## Relevant Skills" in prompt
    assert "- agentic-context (score=9) triggers=context, harness" in prompt
    assert "## Skill Enforcement" in prompt
    assert "- agentic-context: Use the indexed context before asking for repeated repo state." in prompt
    assert "## Skill Verification" in prompt
    assert "- agentic-context: Write the updated handoff note before ending the session." in prompt
    assert "## Session Context" in prompt
    assert "Project: afs (profile: default)" in prompt
    assert "Scratchpad state: Investigating MCP registry split." in prompt
    assert "Deferred: Follow up on prompt packing." in prompt
    assert "Recent changes: 7 files changed" in prompt
    assert "Memory topics: tag:mcp, domain:session" in prompt
    assert "Tasks: 2 (pending=2)" in prompt
    assert "Last session next steps:" in prompt
    assert "- Ship the MCP refactor." in prompt


def test_build_model_system_prompt_budget_drops_dynamic_sections_first() -> None:
    prompt = build_model_system_prompt(
        base_prompt="Base behavior.",
        session_state={
            "project": "afs",
            "profile": "default",
            "scratchpad": {"state_text": "x" * 1200},
        },
        token_budget=8,
    )

    assert "Base behavior." in prompt
    assert "## Session Context" not in prompt
