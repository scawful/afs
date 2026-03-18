from __future__ import annotations

import json
import os
import shlex
import stat
import subprocess
from pathlib import Path

SHELL_INIT = Path(__file__).parent.parent / "scripts" / "afs-shell-init.sh"
AFS_CHECK = Path(__file__).parent.parent / "scripts" / "afs-check"


def _write_fake_python(root: Path, log_path: Path) -> Path:
    python_bin = root / ".venv" / "bin" / "python"
    python_bin.parent.mkdir(parents=True, exist_ok=True)
    python_bin.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "Path(os.environ['FAKE_PYTHON_LOG']).write_text(\n"
        "    json.dumps({'args': sys.argv[1:], 'stdin': sys.stdin.read()}),\n"
        "    encoding='utf-8',\n"
        ")\n"
        "print('ok')\n",
        encoding="utf-8",
    )
    python_bin.chmod(python_bin.stat().st_mode | stat.S_IXUSR)
    return python_bin


def _run_shell_init_helper(
    tmp_path: Path,
    helper_call: str,
    *,
    create_context: bool = True,
) -> dict[str, object]:
    fake_root = tmp_path / "afs-root"
    log_path = tmp_path / "python-log.json"
    _write_fake_python(fake_root, log_path)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if create_context:
        (workspace / ".context").mkdir()

    env = os.environ.copy()
    env["AFS_ROOT"] = str(fake_root)
    env["AFS_VENV"] = str(fake_root / ".venv")
    env["FAKE_PYTHON_LOG"] = str(log_path)

    command = (
        f"source {shlex.quote(str(SHELL_INIT))} && "
        f"cd {shlex.quote(str(workspace))} && "
        f"{helper_call}"
    )
    result = subprocess.run(
        ["bash", "-lc", command],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(log_path.read_text(encoding="utf-8"))


def test_afs_shell_init_has_valid_bash_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SHELL_INIT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_afs_task_passes_title_as_argument(tmp_path: Path) -> None:
    title = "Fix quote's and \"double\" safely"
    payload = _run_shell_init_helper(
        tmp_path,
        shlex.join(["afs-task", title, "7"]),
    )

    assert payload["args"] == ["-", ".context", title, "7"]
    assert "TaskQueue" in str(payload["stdin"])
    assert title not in str(payload["stdin"])


def test_afs_say_passes_payload_as_arguments(tmp_path: Path) -> None:
    message = "quote's-and-\"double\""
    path_value = "C:\\temp\\value"
    payload = _run_shell_init_helper(
        tmp_path,
        shlex.join(
            [
                "afs-say",
                "worker",
                "finding",
                f"message={message}",
                f"path={path_value}",
            ]
        ),
    )

    assert payload["args"] == [
        "-",
        ".context",
        "worker",
        "finding",
        f"message={message}",
        f"path={path_value}",
    ]
    assert "HivemindBus" in str(payload["stdin"])
    assert message not in str(payload["stdin"])


def test_afs_check_prefers_venv_ruff(tmp_path: Path) -> None:
    root = tmp_path / "afs-copy"
    scripts_dir = root / "scripts"
    scripts_dir.mkdir(parents=True)
    copied = scripts_dir / "afs-check"
    copied.write_text(AFS_CHECK.read_text(encoding="utf-8"), encoding="utf-8")
    copied.chmod(copied.stat().st_mode | stat.S_IXUSR)

    (root / "src").mkdir()
    venv_bin = root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    log_path = tmp_path / "ruff-log.txt"

    (venv_bin / "ruff").write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$0 $*\" > \"$FAKE_RUFF_LOG\"\n",
        encoding="utf-8",
    )
    (venv_bin / "ruff").chmod((venv_bin / "ruff").stat().st_mode | stat.S_IXUSR)
    (venv_bin / "python").write_text(
        "#!/usr/bin/env bash\n"
        "exit 0\n",
        encoding="utf-8",
    )
    (venv_bin / "python").chmod((venv_bin / "python").stat().st_mode | stat.S_IXUSR)

    env = os.environ.copy()
    env["AFS_VENV"] = str(root / ".venv")
    env["FAKE_RUFF_LOG"] = str(log_path)

    result = subprocess.run(
        ["bash", str(copied), "--lint-only"],
        cwd=root,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert log_path.read_text(encoding="utf-8").strip().endswith("check src/")


def test_afs_check_has_valid_bash_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(AFS_CHECK)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
