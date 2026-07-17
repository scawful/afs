from __future__ import annotations

import json
from pathlib import Path

from afs.cli import main
from afs.context_layout import scaffold_v2
from afs.project_registry import ProjectRegistry


def _v2_fixture(tmp_path: Path) -> tuple[Path, Path, Path, str, str]:
    context = tmp_path / "central"
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    scaffold_v2(context)
    registry = ProjectRegistry(context)
    alpha_record = registry.register(alpha, name="Alpha")
    beta_record = registry.register(beta, name="Beta")
    config = tmp_path / "afs.toml"
    config.write_text(
        "[general]\n"
        f'context_root = "{context}"\n\n'
        "[extensions]\n"
        "auto_discover = false\n\n"
        "[plugins]\n"
        "auto_discover = false\n",
        encoding="utf-8",
    )
    return config, alpha, beta, alpha_record.project_id, beta_record.project_id


def test_session_handoff_create_list_and_read_are_project_scoped_in_v2(
    tmp_path: Path,
    capsys,
) -> None:
    config, alpha, beta, alpha_id, beta_id = _v2_fixture(tmp_path)

    assert main(
        [
            "session",
            "handoff",
            "create",
            "--config",
            str(config),
            "--path",
            str(alpha),
            "--accomplished",
            "alpha-only",
            "--json",
        ]
    ) == 0
    created = json.loads(capsys.readouterr().out)
    revision_id = created["revision_id"]
    artifact = Path(created["artifact_path"])
    assert artifact.is_relative_to(
        config.parent / "central" / "memory" / "projects" / alpha_id / "handoffs"
    )
    assert not (
        config.parent / "central" / "memory" / "common" / "handoffs"
    ).exists()

    assert main(
        [
            "session",
            "handoff",
            "list",
            "--config",
            str(config),
            "--path",
            str(beta),
            "--json",
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out) == []
    beta_root = (
        config.parent / "central" / "memory" / "projects" / beta_id / "handoffs"
    )
    assert not list(beta_root.rglob("*.md"))

    assert main(
        [
            "session",
            "handoff",
            "read",
            "--config",
            str(config),
            "--path",
            str(beta),
            "--session-id",
            revision_id,
            "--json",
        ]
    ) == 1
    assert capsys.readouterr().out == "no handoff packet found\n"

    assert main(
        [
            "session",
            "handoff",
            "read",
            "--config",
            str(config),
            "--path",
            str(alpha),
            "--session-id",
            revision_id,
            "--json",
        ]
    ) == 0
    restored = json.loads(capsys.readouterr().out)
    assert restored["revision_id"] == revision_id
    assert restored["accomplished"] == ["alpha-only"]


def test_session_handoff_v1_keeps_common_compatibility_scope(
    tmp_path: Path,
    capsys,
) -> None:
    project = tmp_path / "legacy"
    context = project / ".context"
    context.mkdir(parents=True)
    config = tmp_path / "legacy.toml"
    config.write_text(
        "[extensions]\n"
        "auto_discover = false\n\n"
        "[plugins]\n"
        "auto_discover = false\n",
        encoding="utf-8",
    )

    assert main(
        [
            "session",
            "handoff",
            "create",
            "--config",
            str(config),
            "--path",
            str(project),
            "--accomplished",
            "legacy",
            "--json",
        ]
    ) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["accomplished"] == ["legacy"]
    assert Path(created["artifact_path"]).is_relative_to(
        context / "memory" / "common" / "handoffs"
    )
