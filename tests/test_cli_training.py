from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from afs.cli import build_parser
from afs.cli.training import (
    training_extract_sessions_command,
    training_freshness_gate_command,
    training_generate_router_command,
)


def test_training_parser_registers_phase_two_commands() -> None:
    parser = build_parser()
    root_subparsers = next(
        action
        for action in parser._actions
        if getattr(action, "dest", None) == "command"
    )
    training_parser = root_subparsers.choices["training"]
    training_subparsers = next(
        action
        for action in training_parser._actions
        if getattr(action, "dest", None) == "training_command"
    )

    assert "freshness-gate" in training_subparsers.choices
    assert "extract-sessions" in training_subparsers.choices
    assert "generate-router-data" in training_subparsers.choices


def test_training_freshness_gate_json_output(monkeypatch, tmp_path: Path, capsys) -> None:
    class _Report:
        ready = True

        def to_dict(self) -> dict[str, object]:
            return {"ready": True, "overall_score": 0.9}

    monkeypatch.setattr(
        "afs.cli.training._resolve_training_runtime",
        lambda args: (SimpleNamespace(config=SimpleNamespace()), tmp_path / ".context"),
    )
    monkeypatch.setattr(
        "afs.training_integration.freshness_gate.check_training_readiness",
        lambda context_path, config, afs_config: _Report(),
    )
    args = SimpleNamespace(
        config=None,
        path=str(tmp_path),
        context_root=None,
        context_dir=None,
        min_score=0.3,
        decay_hours=168.0,
        mount=None,
        warn_only=False,
        json=True,
    )

    assert training_freshness_gate_command(args) == 0
    assert json.loads(capsys.readouterr().out)["ready"] is True


def test_training_extract_sessions_json_output(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(
        "afs.cli.training._resolve_training_runtime",
        lambda args: (SimpleNamespace(config=SimpleNamespace()), tmp_path / ".context"),
    )
    monkeypatch.setattr(
        "afs.training_integration.session_source.extract_from_sessions",
        lambda context_path, output_path, config, session_limit, afs_config: SimpleNamespace(
            sessions_scanned=2,
            sessions_with_data=1,
            samples_extracted=3,
            samples_filtered=1,
            output_path=output_path,
        ),
    )
    output_path = tmp_path / "sessions.jsonl"
    args = SimpleNamespace(
        config=None,
        path=str(tmp_path),
        context_root=None,
        context_dir=None,
        output=str(output_path),
        quality_floor=0.6,
        session_limit=20,
        json=True,
    )

    assert training_extract_sessions_command(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sessions_scanned"] == 2
    assert payload["output_path"] == str(output_path)


def test_training_generate_router_data_uses_runtime_config(monkeypatch, tmp_path: Path, capsys) -> None:
    config_object = SimpleNamespace(name="config")
    monkeypatch.setattr(
        "afs.cli._utils.load_runtime_config_from_args",
        lambda args, start_dir: (config_object, tmp_path / "afs.toml"),
    )
    monkeypatch.setattr(
        "afs.generators.router_from_capabilities.generate_router_dataset",
        lambda output_path, config, afs_config: SimpleNamespace(
            agents_processed=4,
            agents_with_capabilities=3,
            samples_generated=9,
            agent_sample_counts={"context-warm": 4, "history-memory": 5},
            output_path=output_path,
        )
        if afs_config is config_object
        else None,
    )
    output_path = tmp_path / "router.jsonl"
    args = SimpleNamespace(
        config=None,
        path=str(tmp_path),
        output=str(output_path),
        samples_per_agent=10,
        include_all=False,
        json=True,
    )

    assert training_generate_router_command(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["samples_generated"] == 9
    assert payload["output_path"] == str(output_path)
