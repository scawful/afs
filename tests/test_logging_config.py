from __future__ import annotations

import logging
from pathlib import Path

import pytest

from afs import logging_config
from afs.context_layout import audit_layout, scaffold_v2


def _close_handlers(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()


def test_default_logging_routes_v2_under_runtime_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context_root = tmp_path / ".context"
    scaffold_v2(context_root)
    monkeypatch.setattr(logging_config, "_default_context_root", lambda: context_root)

    logger = logging_config.setup_logging(
        name="v2-test",
        enable_json=False,
        enable_console=False,
        enable_rotation=False,
    )
    try:
        assert (context_root / ".afs" / "logs" / "afs").is_dir()
        assert not (context_root / "logs").exists()
        assert audit_layout(context_root).valid is True
    finally:
        _close_handlers(logger)


def test_default_logging_keeps_v1_logs_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context_root = tmp_path / ".context"
    monkeypatch.setattr(logging_config, "_default_context_root", lambda: context_root)

    logger = logging_config.setup_logging(
        name="v1-test",
        enable_json=False,
        enable_console=False,
        enable_rotation=False,
    )
    try:
        assert (context_root / "logs" / "afs").is_dir()
    finally:
        _close_handlers(logger)


def test_explicit_logging_directory_remains_unchanged(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context_root = tmp_path / ".context"
    scaffold_v2(context_root)
    explicit = tmp_path / "explicit-logs"
    monkeypatch.setattr(logging_config, "_default_context_root", lambda: context_root)

    logger = logging_config.setup_logging(
        name="explicit-test",
        log_dir=explicit,
        enable_json=False,
        enable_console=False,
        enable_rotation=False,
    )
    try:
        assert explicit.is_dir()
        assert not (context_root / ".afs" / "logs").exists()
    finally:
        _close_handlers(logger)


def test_default_logging_rejects_linked_v2_runtime_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context_root = tmp_path / ".context"
    scaffold_v2(context_root)
    outside = tmp_path / "outside-logs"
    outside.mkdir()
    try:
        (context_root / ".afs" / "logs").symlink_to(
            outside,
            target_is_directory=True,
        )
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"symlinks unavailable: {exc}")
    monkeypatch.setattr(logging_config, "_default_context_root", lambda: context_root)

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        logging_config.setup_logging(
            name="linked-root",
            enable_json=False,
            enable_console=False,
            enable_rotation=False,
        )

    assert list(outside.iterdir()) == []


def test_default_logging_rejects_linked_v2_log_leaf(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context_root = tmp_path / ".context"
    scaffold_v2(context_root)
    log_dir = context_root / ".afs" / "logs" / "afs"
    log_dir.mkdir(parents=True)
    outside = tmp_path / "outside.log"
    outside.write_text("outside\n", encoding="utf-8")
    try:
        (log_dir / "linked-leaf.log").symlink_to(outside)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"symlinks unavailable: {exc}")
    monkeypatch.setattr(logging_config, "_default_context_root", lambda: context_root)

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        logging_config.setup_logging(
            name="linked-leaf",
            enable_json=False,
            enable_console=False,
            enable_rotation=True,
        )

    assert outside.read_text(encoding="utf-8") == "outside\n"
