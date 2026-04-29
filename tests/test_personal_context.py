from __future__ import annotations

from pathlib import Path

from afs.personal_context import load_personal_context, render_personal_context


def test_personal_context_renders_work_communication_contract(tmp_path: Path) -> None:
    root = tmp_path / "personal"
    root.mkdir()
    (root / "profile.toml").write_text('name = "Test User"\n', encoding="utf-8")
    (root / "samples.md").write_text("Use short, concrete replies.\n", encoding="utf-8")
    (root / "manifest.toml").write_text(
        """
[modes.work]
tone = "direct and specific"
work_context = true
load = ["samples.md"]
style_instructions = ["findings first", "avoid generic corporate filler"]
communication_sources = ["samples.md", "recent approved work comments"]
posting_policy = "Ask for explicit approval before posting externally."
""".strip()
        + "\n",
        encoding="utf-8",
    )

    payload = load_personal_context("work", context_root=root)
    assert payload.work_context is True
    assert payload.style_instructions == ["findings first", "avoid generic corporate filler"]
    assert payload.communication_sources == ["samples.md", "recent approved work comments"]
    assert payload.posting_policy == "Ask for explicit approval before posting externally."

    rendered = render_personal_context("work", context_root=root)
    assert "## Work communication instructions" in rendered
    assert "inspect the loaded samples and profile" in rendered
    assert "Do not post, send, submit, or edit" in rendered
    assert "findings first" in rendered
    assert "recent approved work comments" in rendered
