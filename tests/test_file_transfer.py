"""SSH transport cannot turn filenames or content into remote commands."""

import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from afs.file_transfer import push_text


def test_push_quotes_remote_arguments_and_sends_exact_stdin(monkeypatch) -> None:
    captured = {}

    def run(argv, **kwargs):
        captured.update(argv=argv, **kwargs)
        return subprocess.CompletedProcess(argv, 0, b'{"written":true}', b"")

    monkeypatch.setattr("afs.file_transfer.subprocess.run", run)
    destination = "notes/$(touch unwanted) 'quote'.md"
    result = push_text(
        "hello\r\n世界\n", host="user@alias", mount="scratchpad",
        destination=destination, if_match="missing", context_root="/a context",
    )
    assert captured["input"] == "hello\r\n世界\n".encode()
    command = shlex.split(captured["argv"][-1])
    assert command[:5] == ["afs", "fs", "write", "scratchpad", destination]
    assert command[command.index("--if-match") + 1] == "missing"
    assert "--delete" not in command
    assert result["receipt"]["written"] is True


def test_push_conflict_is_not_retried(monkeypatch) -> None:
    calls = []

    def run(argv, **_kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 1, b"write conflict", b"")

    monkeypatch.setattr("afs.file_transfer.subprocess.run", run)
    with pytest.raises(RuntimeError, match="write conflict"):
        push_text("text", host="alias", mount="scratchpad", destination="note",
                  if_match="missing", context_root="/context")
    assert len(calls) == 1


@pytest.mark.parametrize("host", ["-oProxyCommand=bad", "host; bad", "user@host bad"])
def test_push_rejects_option_or_shell_host_injection(host: str) -> None:
    with pytest.raises(ValueError, match="host"):
        push_text("text", host=host, mount="scratchpad", destination="note",
                  if_match="missing", context_root="/context")


def test_push_drives_real_destination_cli_through_local_transport(monkeypatch, tmp_path: Path) -> None:
    from afs.manager import AFSManager
    from afs.schema import AFSConfig, GeneralConfig

    root = tmp_path / "context"
    project = tmp_path / "project"
    project.mkdir()
    manager = AFSManager(config=AFSConfig(general=GeneralConfig(context_root=root)))
    manager.ensure(path=project, context_root=root)
    config = tmp_path / "afs.toml"
    config.write_text(f'[general]\ncontext_root = "{root}"\n[history]\nenabled = false\n')
    monkeypatch.setenv("AFS_CONFIG_PATH", str(config))
    monkeypatch.setenv("AFS_CONTEXT_ROOT", str(root))
    monkeypatch.setenv("AFS_HISTORY_DISABLED", "1")
    monkeypatch.setenv("AFS_PYTHON", sys.executable)
    original_run = subprocess.run

    def local_transport(argv, **kwargs):
        assert argv[0] == "ssh"
        return original_run(shlex.split(argv[-1]), **kwargs)

    monkeypatch.setattr("afs.file_transfer.subprocess.run", local_transport)
    args = {
        "host": "fixture", "mount": "scratchpad", "destination": "note.md",
        "if_match": "missing", "context_root": str(root), "project_path": str(project),
        "remote_afs": str(Path(__file__).resolve().parents[1] / "scripts" / "afs"),
    }
    push_text("exact\r\n世界\n", **args)
    target = root / "scratchpad" / "note.md"
    assert target.read_bytes() == "exact\r\n世界\n".encode()
    with pytest.raises(RuntimeError, match="write conflict"):
        push_text("stale replacement", **args)
    assert target.read_bytes() == "exact\r\n世界\n".encode()
