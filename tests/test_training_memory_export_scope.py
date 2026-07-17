from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from afs.agents import memory_export as memory_export_agent
from afs.cli.training import training_memory_export_command
from afs.context_layout import scaffold_v2
from afs.project_registry import ProjectRegistry
from afs.schema import AFSConfig, GeneralConfig
from afs.training_export_scope import resolve_memory_export_tree


@dataclass
class _ExportResult:
    total_entries: int = 0
    exported: int = 0
    skipped: int = 0
    filtered: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return f"exported: {self.exported}"


def _v2_memory_fixture(
    tmp_path: Path,
) -> tuple[AFSConfig, Path, Path, Path, str, str]:
    context_root = tmp_path / ".context"
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    scaffold_v2(context_root)
    registry = ProjectRegistry(context_root)
    alpha_record = registry.register(alpha)
    beta_record = registry.register(beta)
    memory_root = context_root / "memory"
    entries = {
        memory_root / "common" / "common.json": "common",
        memory_root / "projects" / alpha_record.project_id / "alpha.json": "alpha",
        memory_root / "projects" / beta_record.project_id / "beta.json": "beta",
    }
    for path, content in entries.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    config = AFSConfig(general=GeneralConfig(context_root=context_root))
    return (
        config,
        context_root,
        alpha,
        beta,
        alpha_record.project_id,
        beta_record.project_id,
    )


def _exported_files(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*.json"))
    }


def test_memory_export_agent_parser_exposes_v2_scope_controls() -> None:
    args = memory_export_agent.build_parser().parse_args(
        ["--path", "/tmp/project", "--memory-root", "/tmp/admin"]
    )

    assert args.path == "/tmp/project"
    assert args.memory_root == "/tmp/admin"


def test_v2_memory_export_tree_contains_current_project_and_common_only(
    tmp_path: Path,
) -> None:
    config, context_root, alpha, _beta, alpha_id, beta_id = _v2_memory_fixture(tmp_path)

    with resolve_memory_export_tree(
        context_root,
        config=config,
        requester_path=alpha,
    ) as export_tree:
        assert _exported_files(export_tree.export_root) == {
            "common/common.json": "common",
            f"projects/{alpha_id}/alpha.json": "alpha",
        }
        assert export_tree.scope_id == f"project:{alpha_id}"
        assert beta_id not in "\n".join(_exported_files(export_tree.export_root))


def test_v2_memory_export_requires_registered_requester_unless_admin_override(
    tmp_path: Path,
) -> None:
    config, context_root, _alpha, beta, _alpha_id, beta_id = _v2_memory_fixture(tmp_path)
    unregistered = tmp_path / "unregistered"
    unregistered.mkdir()

    with pytest.raises(PermissionError, match="not registered"):
        with resolve_memory_export_tree(
            context_root,
            config=config,
            requester_path=unregistered,
        ):
            pass

    beta_root = context_root / "memory" / "projects" / beta_id
    with resolve_memory_export_tree(
        context_root,
        config=config,
        requester_path=unregistered,
        memory_root_override=beta_root,
    ) as export_tree:
        assert _exported_files(export_tree.export_root) == {"beta.json": "beta"}
        assert export_tree.explicit_override is True
        assert export_tree.source_roots == (beta_root.resolve(),)


def test_v1_memory_export_retains_recursive_legacy_root(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    nested = context_root / "memory" / "nested" / "entry.json"
    nested.parent.mkdir(parents=True)
    nested.write_text("legacy", encoding="utf-8")
    config = AFSConfig(general=GeneralConfig(context_root=context_root))

    with resolve_memory_export_tree(
        context_root,
        config=config,
        requester_path=tmp_path / "not-registered",
    ) as export_tree:
        assert export_tree.export_root == context_root / "memory"
        assert _exported_files(export_tree.export_root) == {"nested/entry.json": "legacy"}


def test_cli_memory_export_passes_only_authorized_v2_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, context_root, alpha, _beta, alpha_id, beta_id = _v2_memory_fixture(tmp_path)
    seen: list[dict[str, str]] = []

    def fake_export(root: Path, _output: Path, **_kwargs: Any) -> _ExportResult:
        files = _exported_files(root)
        seen.append(files)
        return _ExportResult(total_entries=len(files), exported=len(files))

    monkeypatch.setattr("afs.config.load_config_model", lambda **_kwargs: config)
    monkeypatch.setattr("afs.training.export_memory_to_dataset", fake_export, raising=False)
    args = argparse.Namespace(
        config=None,
        context_root=str(context_root),
        path=str(alpha),
        memory_root=None,
        output=str(tmp_path / "output.jsonl"),
        domain="memory",
        allow_raw=False,
        allow_raw_tag=None,
        default_instruction=None,
        include_tag=None,
        exclude_tag=None,
        limit=None,
        require_quality=None,
        no_require_quality=None,
        min_quality_score=None,
        score_profile=None,
        enable_asar=False,
        no_redact=False,
    )

    assert training_memory_export_command(args) == 0
    assert seen == [
        {
            "common/common.json": "common",
            f"projects/{alpha_id}/alpha.json": "alpha",
        }
    ]
    assert beta_id not in repr(seen)


def test_cli_memory_export_refuses_unregistered_v2_requester(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, context_root, _alpha, _beta, _alpha_id, _beta_id = _v2_memory_fixture(
        tmp_path
    )
    unregistered = tmp_path / "unregistered"
    unregistered.mkdir()
    called = False

    def fake_export(_root: Path, _output: Path, **_kwargs: Any) -> _ExportResult:
        nonlocal called
        called = True
        return _ExportResult()

    monkeypatch.setattr("afs.config.load_config_model", lambda **_kwargs: config)
    monkeypatch.setattr("afs.training.export_memory_to_dataset", fake_export, raising=False)
    args = argparse.Namespace(
        config=None,
        context_root=str(context_root),
        path=str(unregistered),
        memory_root=None,
        output=str(tmp_path / "output.jsonl"),
        domain="memory",
        allow_raw=False,
        allow_raw_tag=None,
        default_instruction=None,
        include_tag=None,
        exclude_tag=None,
        limit=None,
        require_quality=None,
        no_require_quality=None,
        min_quality_score=None,
        score_profile=None,
        enable_asar=False,
        no_redact=False,
    )

    assert training_memory_export_command(args) == 2
    assert called is False
    assert "memory export refused" in capsys.readouterr().err


def test_agent_memory_export_passes_only_authorized_v2_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, context_root, alpha, _beta, alpha_id, beta_id = _v2_memory_fixture(tmp_path)
    seen: list[dict[str, str]] = []

    def fake_export(root: Path, _output: Path, **_kwargs: Any) -> _ExportResult:
        files = _exported_files(root)
        seen.append(files)
        return _ExportResult(total_entries=len(files), exported=len(files))

    monkeypatch.setattr("afs.training.export_memory_to_dataset", fake_export, raising=False)
    args = argparse.Namespace(
        context_root=str(context_root),
        path=str(alpha),
        memory_root=None,
        dataset_output=str(tmp_path / "output.jsonl"),
        domain="memory",
        allow_raw=False,
        default_instruction=None,
        limit=None,
    )

    result = memory_export_agent._run_export(args, config)

    assert seen == [
        {
            "common/common.json": "common",
            f"projects/{alpha_id}/alpha.json": "alpha",
        }
    ]
    assert beta_id not in repr(seen)
    assert result.payload["scope_id"] == f"project:{alpha_id}"
    assert result.payload["explicit_memory_root"] is False
