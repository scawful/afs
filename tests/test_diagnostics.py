"""Tests for the AFS diagnostics engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from afs.config import load_config_model
from afs.context_paths import resolve_agent_output_root
from afs.diagnostics import (
    DiagnosticResult,
    check_config,
    check_context_health,
    check_context_index,
    check_context_root,
    check_dependencies,
    check_mcp_registration,
    check_python_environment,
    check_services,
    check_skills,
    format_results_json,
    format_results_text,
    run_all_checks,
    write_doctor_snapshot,
)
from afs.manager import AFSManager
from afs.models import MountType, ProjectMetadata
from afs.schema import (
    AFSConfig,
    GeneralConfig,
    ProfileConfig,
    ProfilesConfig,
    ServiceConfig,
    ServicesConfig,
)
from afs.services import manager as services_manager


def _write_config(config_path: Path, context_root: Path) -> None:
    config_path.write_text(
        "[profiles]\n"
        "active_profile = \"default\"\n"
        "auto_apply = false\n\n"
        "[plugins]\n"
        "auto_discover = false\n"
        "enabled_plugins = []\n\n"
        "[extensions]\n"
        "auto_discover = false\n"
        "enabled_extensions = []\n\n"
        f"[general]\ncontext_root = \"{context_root}\"\n",
        encoding="utf-8",
    )


def test_diagnostic_result_to_dict() -> None:
    result = DiagnosticResult(
        name="test",
        status="ok",
        message="all good",
        fix_available=True,
        fix_description="do something",
    )
    d = result.to_dict()
    assert d["name"] == "test"
    assert d["status"] == "ok"
    assert d["fix_available"] is True
    assert "_fix_fn" not in d


def test_check_python_environment() -> None:
    result = check_python_environment()
    assert result.status in {"ok", "warn"}
    assert result.name == "python"


def test_check_dependencies() -> None:
    result = check_dependencies()
    assert result.status in {"ok", "warn"}
    assert result.name == "dependencies"


def test_check_skills_warns_with_structured_details_but_counts_good_skills(
    tmp_path: Path,
    monkeypatch,
) -> None:
    skill_root = tmp_path / "skills"
    good = skill_root / "good" / "SKILL.md"
    invalid = skill_root / "invalid" / "SKILL.md"
    good.parent.mkdir(parents=True)
    invalid.parent.mkdir(parents=True)
    good.write_text("---\nname: good\n---\n", encoding="utf-8")
    invalid.write_text(
        "---\nname: invalid\nenforcement:\n"
        + "".join(f"  - rule {index}\n" for index in range(17))
        + "---\n",
        encoding="utf-8",
    )
    config = AFSConfig(
        general=GeneralConfig(context_root=tmp_path / "context"),
        profiles=ProfilesConfig(
            active_profile="default",
            profiles={"default": ProfileConfig(skill_roots=[skill_root])},
        ),
    )
    monkeypatch.setattr(
        "afs.diagnostics._load_runtime",
        lambda config_path=None: (
            config,
            AFSManager(config=config),
            config.general.context_root,
        ),
    )

    result = check_skills()

    assert result.status == "warn"
    assert result.name == "skills"
    assert "valid skill(s) loaded with 1 warning(s)" in result.message
    assert result.details[0]["code"] == "skill_invalid"
    assert result.details[0]["path"] == str(invalid)
    assert result.to_dict()["details"] == result.details


def test_check_skills_uses_repo_runtime_precedence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    user_config = home / ".config" / "afs" / "config.toml"
    user_skill = tmp_path / "user-skills" / "good" / "SKILL.md"
    local_invalid = tmp_path / "local-skills" / "invalid" / "SKILL.md"
    user_config.parent.mkdir(parents=True)
    repo.mkdir()
    user_skill.parent.mkdir(parents=True)
    local_invalid.parent.mkdir(parents=True)
    user_skill.write_text("---\nname: user-good\n---\n", encoding="utf-8")
    local_invalid.write_text(
        "---\nname: local-invalid\nenforcement:\n"
        + "".join(f"  - rule {index}\n" for index in range(17))
        + "---\n",
        encoding="utf-8",
    )
    user_config.write_text(
        "[profiles]\n"
        'active_profile = "user"\n'
        "[profiles.user]\n"
        f'skill_roots = ["{user_skill.parent.parent}"]\n',
        encoding="utf-8",
    )
    (repo / "afs.toml").write_text(
        "[profiles]\n"
        'active_profile = "local"\n'
        "[profiles.local]\n"
        f'skill_roots = ["{local_invalid.parent.parent}"]\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    for name in (
        "AFS_CONFIG_PATH",
        "AFS_PREFER_REPO_CONFIG",
        "AFS_PREFER_USER_CONFIG",
        "AFS_SKILL_ROOTS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(repo)

    result = check_skills()

    assert result.status == "warn"
    assert "1 warning(s)" in result.message
    assert result.details[0]["path"] == str(local_invalid)
    assert str(user_skill) not in result.message


def test_check_config_valid(tmp_path: Path) -> None:
    config_path = tmp_path / "afs.toml"
    _write_config(config_path, tmp_path / "context")
    result = check_config(config_path)
    assert result.status in {"ok", "warn"}
    assert result.name == "config"


def test_check_context_root_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "afs.toml"
    _write_config(config_path, tmp_path / "nonexistent" / ".context")
    result = check_context_root(config_path)
    assert result.status in {"error", "warn"}
    assert result.fix_available


def test_check_context_root_fix(tmp_path: Path) -> None:
    missing = tmp_path / "fixme" / ".context"
    config_path = tmp_path / "afs.toml"
    _write_config(config_path, missing)
    result = check_context_root(config_path)
    assert result.fix_available
    assert result._fix_fn is not None
    msg = result._fix_fn()
    assert "Created" in msg


def test_check_context_root_fix_respects_metadata_directory_mapping(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    context_root.mkdir()
    metadata = ProjectMetadata(
        directories={
            MountType.MEMORY.value: "memory",
            MountType.KNOWLEDGE.value: "knowledge",
            MountType.SCRATCHPAD.value: "notes",
        }
    )
    (context_root / "metadata.json").write_text(
        json.dumps(metadata.to_dict()),
        encoding="utf-8",
    )
    (context_root / "memory").mkdir()
    (context_root / "knowledge").mkdir()
    config_path = tmp_path / "afs.toml"
    _write_config(config_path, context_root)

    result = check_context_root(config_path)

    assert result.status == "warn"
    assert "scratchpad=notes" in result.message
    assert result._fix_fn is not None
    result._fix_fn()
    assert (context_root / "notes").exists()
    assert not (context_root / "scratchpad").exists()


def test_check_context_health_fix_seeds_untracked_mount_provenance(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    config_path = tmp_path / "afs.toml"
    _write_config(config_path, context_root)
    config = load_config_model(config_path=config_path, merge_user=False)
    manager = AFSManager(config=config)
    manager.ensure(context_root=context_root)
    source = tmp_path / "docs"
    source.mkdir()
    manager.mount(source, MountType.KNOWLEDGE, alias="docs", context_path=context_root)

    metadata_path = context_root / "metadata.json"
    metadata = ProjectMetadata.from_dict(
        json.loads(metadata_path.read_text(encoding="utf-8"))
    )
    metadata.mount_provenance = {}
    metadata_path.write_text(json.dumps(metadata.to_dict()), encoding="utf-8")

    result = check_context_health(config_path)

    assert result.status == "warn"
    assert "untracked=1" in result.message
    assert result._fix_fn is not None
    result._fix_fn()

    repaired = manager.context_health(context_root)
    assert repaired["provenance"]["untracked_mounts"] == []


def test_check_context_index_fix_rebuilds_missing_index_entries(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    config_path = tmp_path / "afs.toml"
    _write_config(config_path, context_root)
    config = load_config_model(config_path=config_path, merge_user=False)
    manager = AFSManager(config=config)
    manager.ensure(context_root=context_root)
    knowledge_root = manager.resolve_mount_root(context_root, MountType.KNOWLEDGE)
    (knowledge_root / "guide.md").write_text("# Guide\n", encoding="utf-8")

    result = check_context_index(config_path)

    assert result.status == "warn"
    assert result.fix_available
    assert result._fix_fn is not None
    result._fix_fn()

    healthy = check_context_index(config_path)
    assert healthy.status == "ok"


def test_check_context_index_ignores_volatile_mount_drift(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    config_path = tmp_path / "afs.toml"
    _write_config(config_path, context_root)
    config = load_config_model(config_path=config_path, merge_user=False)
    manager = AFSManager(config=config)
    manager.ensure(context_root=context_root)
    knowledge_root = manager.resolve_mount_root(context_root, MountType.KNOWLEDGE)
    scratchpad_root = manager.resolve_mount_root(context_root, MountType.SCRATCHPAD)
    (knowledge_root / "guide.md").write_text("# Guide\n", encoding="utf-8")
    (scratchpad_root / "note.md").write_text("draft\n", encoding="utf-8")

    manager.rebuild_context_index(context_root)
    (scratchpad_root / "note.md").write_text("draft updated\n", encoding="utf-8")

    result = check_context_index(config_path)

    assert result.status == "ok"


def test_check_context_index_warns_when_stable_mount_changes(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    config_path = tmp_path / "afs.toml"
    _write_config(config_path, context_root)
    config = load_config_model(config_path=config_path, merge_user=False)
    manager = AFSManager(config=config)
    manager.ensure(context_root=context_root)
    knowledge_root = manager.resolve_mount_root(context_root, MountType.KNOWLEDGE)
    (knowledge_root / "guide.md").write_text("# Guide\n", encoding="utf-8")

    manager.rebuild_context_index(context_root)
    (knowledge_root / "guide.md").write_text("# Guide\nupdated\n", encoding="utf-8")

    result = check_context_index(config_path)

    assert result.status == "warn"
    assert "stale" in result.message


def test_check_mcp_registration() -> None:
    result = check_mcp_registration()
    assert result.name == "mcp_registration"
    assert result.status in {"ok", "warn"}


def test_check_services_warns_for_stopped_autostart_service(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = AFSConfig(
        general=GeneralConfig(context_root=tmp_path / "context"),
        services=ServicesConfig(
            enabled=True,
            services={
                "history-memory": ServiceConfig(
                    name="history-memory",
                    auto_start=True,
                ),
            },
        ),
    )
    manager = AFSManager(config=config)
    monkeypatch.setattr(
        "afs.diagnostics._load_runtime",
        lambda _config_path=None: (config, manager, config.general.context_root),
    )
    monkeypatch.setattr(services_manager, "STATE_DIR", tmp_path / "service-state")

    result = check_services()

    assert result.status == "warn"
    assert "history-memory=stopped" in result.message


def test_run_all_checks_returns_list(tmp_path: Path) -> None:
    config_path = tmp_path / "afs.toml"
    _write_config(config_path, tmp_path / "context")
    results = run_all_checks(config_path=config_path)
    assert isinstance(results, list)
    assert len(results) >= 8
    for r in results:
        assert isinstance(r, DiagnosticResult)
        assert r.status in {"ok", "warn", "error"}


def test_run_all_checks_auto_fix(tmp_path: Path) -> None:
    missing = tmp_path / "autofix" / ".context"
    config_path = tmp_path / "afs.toml"
    _write_config(config_path, missing)
    results = run_all_checks(config_path=config_path, auto_fix=True)
    context_result = next(r for r in results if r.name == "context_root")
    assert context_result.fix_applied


def test_format_results_text() -> None:
    results = [
        DiagnosticResult(name="foo", status="ok", message="fine"),
        DiagnosticResult(
            name="bar",
            status="warn",
            message="needs fix",
            fix_available=True,
            fix_description="run something",
        ),
    ]
    text = format_results_text(results)
    assert "AFS Doctor" in text
    assert "foo" in text
    assert "bar" in text
    assert "run something" in text


def test_format_results_json() -> None:
    results = [
        DiagnosticResult(name="foo", status="ok", message="fine"),
    ]
    output = format_results_json(results)
    parsed = json.loads(output)
    assert "checks" in parsed
    assert parsed["checks"][0]["name"] == "foo"


def test_write_doctor_snapshot_writes_agent_report(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    config_path = tmp_path / "afs.toml"
    _write_config(config_path, context_root)

    path = write_doctor_snapshot(
        config_path=config_path,
        results=[DiagnosticResult(name="config", status="warn", message="needs attention")],
    )

    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "warn"
    assert payload["payload"]["context_root"] == str(context_root)
    assert payload["payload"]["checks"][0]["name"] == "config"


@pytest.mark.parametrize("linked_component", ["root", "leaf"])
def test_v2_doctor_snapshot_rejects_linked_common_output(
    tmp_path: Path,
    linked_component: str,
) -> None:
    context_root = tmp_path / "context"
    project = tmp_path / "project"
    project.mkdir()
    config = AFSConfig(general=GeneralConfig(context_root=context_root))
    AFSManager(config=config).ensure(
        path=project,
        context_root=context_root,
        layout_version=2,
    )
    config_path = tmp_path / "afs.toml"
    _write_config(config_path, context_root)
    output_root = resolve_agent_output_root(context_root, config=config, scope_id="common")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    poison = outside / "doctor_snapshot.json"
    poison.write_text("do-not-overwrite", encoding="utf-8")

    try:
        if linked_component == "root":
            output_root.symlink_to(outside, target_is_directory=True)
        else:
            output_root.mkdir()
            (output_root / "doctor_snapshot.json").symlink_to(poison)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="symbolic link|reparse point"):
        write_doctor_snapshot(config_path=config_path, results=[])

    assert poison.read_text(encoding="utf-8") == "do-not-overwrite"
