from __future__ import annotations

import json
import os
import shlex
import stat
import subprocess
from pathlib import Path

SHELL_INIT = Path(__file__).parent.parent / "scripts" / "afs-shell-init.sh"
AFS_CHECK = Path(__file__).parent.parent / "scripts" / "afs-check"
AFS_CLIENT_SESSION = Path(__file__).parent.parent / "scripts" / "afs-client-session"
AFS_SESSION_NOTIFY = Path(__file__).parent.parent / "scripts" / "afs-session-notify"
AFS_SESSION_VERIFY = Path(__file__).parent.parent / "scripts" / "afs-session-verify"


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


def _write_fake_afs_cli(
    root: Path,
    bootstrap_json: Path,
    bootstrap_markdown: Path,
    pack_json: Path,
    pack_markdown: Path,
    skills_json: Path,
    prompt_json: Path,
    prompt_text: Path,
    payload_json: Path,
    context_root: Path,
) -> Path:
    afs_cli = root / "scripts" / "afs"
    afs_cli.parent.mkdir(parents=True, exist_ok=True)
    workspace_path = context_root.parent / "workspace"
    afs_cli.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [ -n \"${FAKE_AFS_LOG:-}\" ]; then\n"
        "  printf '%s\\n' \"$*\" >> \"$FAKE_AFS_LOG\"\n"
        "fi\n"
        "if [ \"$#\" -ge 3 ] && [ \"$1\" = \"session\" ] && [ \"$2\" = \"prepare-client\" ]; then\n"
        f"  cat <<'JSON'\n"
        "{\n"
        f"  \"client\": \"fake-client\",\n"
        "  \"session_id\": \"sess-from-prepare\",\n"
        f"  \"context_path\": \"{context_root}\",\n"
        "  \"bootstrap\": {\n"
        "    \"artifact_paths\": {\n"
        f"      \"json\": \"{bootstrap_json}\",\n"
        f"      \"markdown\": \"{bootstrap_markdown}\"\n"
        "    }\n"
        "  },\n"
        "  \"pack\": {\n"
        "    \"artifact_paths\": {\n"
        f"      \"json\": \"{pack_json}\",\n"
        f"      \"markdown\": \"{pack_markdown}\"\n"
        "    }\n"
        "  },\n"
        "  \"skills\": {\n"
        "    \"artifact_paths\": {\n"
        f"      \"json\": \"{skills_json}\"\n"
        "    }\n"
        "  },\n"
        "  \"prompt\": {\n"
        "    \"artifact_paths\": {\n"
        f"      \"json\": \"{prompt_json}\",\n"
        f"      \"text\": \"{prompt_text}\"\n"
        "    }\n"
        "  },\n"
        "  \"artifact_paths\": {\n"
        f"    \"json\": \"{payload_json}\"\n"
        "  },\n"
        "  \"cli_hints\": {\n"
        f"    \"workspace_path\": \"{workspace_path}\",\n"
        f"    \"query_shortcut\": \"afs query <text> --path {workspace_path}\",\n"
        f"    \"query_canonical\": \"afs context query <text> --path {workspace_path}\",\n"
        f"    \"index_rebuild\": \"afs index rebuild --path {workspace_path}\",\n"
        "    \"notes\": []\n"
        "  }\n"
        "}\n"
        "JSON\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$#\" -ge 3 ] && [ \"$1\" = \"session\" ] && [ \"$2\" = \"hook\" ]; then\n"
        "  if [ \"$3\" = \"session_end\" ] && [ -n \"${FAKE_SESSION_END_HOOK_EXIT:-}\" ]; then\n"
        "    exit \"${FAKE_SESSION_END_HOOK_EXIT}\"\n"
        "  fi\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$#\" -ge 3 ] && [ \"$1\" = \"session\" ] && [ \"$2\" = \"event\" ]; then\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$#\" -ge 2 ] && [ \"$1\" = \"agents\" ] && [ \"$2\" = \"wait\" ]; then\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$#\" -ge 2 ] && [ \"$1\" = \"agents\" ] && [ \"$2\" = \"monitor\" ]; then\n"
        "  exit 0\n"
        "fi\n"
        "exit 1\n",
        encoding="utf-8",
    )
    afs_cli.chmod(afs_cli.stat().st_mode | stat.S_IXUSR)
    return afs_cli


def _write_fake_client(path: Path, log_path: Path) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "payload = {\n"
        "    'args': sys.argv[1:],\n"
        "    'AFS_MCP_ALLOWED_ROOTS': os.environ.get('AFS_MCP_ALLOWED_ROOTS'),\n"
        "    'AFS_SESSION_BOOTSTRAP_JSON': os.environ.get('AFS_SESSION_BOOTSTRAP_JSON'),\n"
        "    'AFS_SESSION_BOOTSTRAP_MARKDOWN': os.environ.get('AFS_SESSION_BOOTSTRAP_MARKDOWN'),\n"
        "    'AFS_SESSION_PACK_JSON': os.environ.get('AFS_SESSION_PACK_JSON'),\n"
        "    'AFS_SESSION_PACK_MARKDOWN': os.environ.get('AFS_SESSION_PACK_MARKDOWN'),\n"
        "    'AFS_SESSION_SKILLS_JSON': os.environ.get('AFS_SESSION_SKILLS_JSON'),\n"
        "    'AFS_SESSION_SYSTEM_PROMPT_JSON': os.environ.get('AFS_SESSION_SYSTEM_PROMPT_JSON'),\n"
        "    'AFS_SESSION_SYSTEM_PROMPT_TEXT': os.environ.get('AFS_SESSION_SYSTEM_PROMPT_TEXT'),\n"
        "    'AFS_SESSION_CLIENT_PAYLOAD_JSON': os.environ.get('AFS_SESSION_CLIENT_PAYLOAD_JSON'),\n"
        "    'AFS_SESSION_QUERY_HINT': os.environ.get('AFS_SESSION_QUERY_HINT'),\n"
        "    'AFS_SESSION_CONTEXT_QUERY_HINT': os.environ.get('AFS_SESSION_CONTEXT_QUERY_HINT'),\n"
        "    'AFS_SESSION_INDEX_REBUILD_HINT': os.environ.get('AFS_SESSION_INDEX_REBUILD_HINT'),\n"
        "    'AFS_SESSION_EVENT_BIN': os.environ.get('AFS_SESSION_EVENT_BIN'),\n"
        "    'AFS_SESSION_DEFAULT_TURN_ID': os.environ.get('AFS_SESSION_DEFAULT_TURN_ID'),\n"
        "    'AFS_ACTIVE_CONTEXT_ROOT': os.environ.get('AFS_ACTIVE_CONTEXT_ROOT'),\n"
        "    'AFS_SESSION_ID': os.environ.get('AFS_SESSION_ID'),\n"
        "    'GEMINI_SYSTEM_MD': os.environ.get('GEMINI_SYSTEM_MD'),\n"
        "}\n"
        f"Path({str(log_path)!r}).write_text(json.dumps(payload), encoding='utf-8')\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


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


def _run_client_session(
    tmp_path: Path,
    *,
    client_label: str = "gemini",
    env_overrides: dict[str, str] | None = None,
    wrapper_args: list[str] | None = None,
    client_args: list[str] | None = None,
    expected_returncode: int = 0,
) -> dict[str, object]:
    root = tmp_path / "afs-copy"
    scripts_dir = root / "scripts"
    scripts_dir.mkdir(parents=True)

    copied = scripts_dir / "afs-client-session"
    copied.write_text(AFS_CLIENT_SESSION.read_text(encoding="utf-8"), encoding="utf-8")
    copied.chmod(copied.stat().st_mode | stat.S_IXUSR)
    copied_notify = scripts_dir / "afs-session-notify"
    copied_notify.write_text(AFS_SESSION_NOTIFY.read_text(encoding="utf-8"), encoding="utf-8")
    copied_notify.chmod(copied_notify.stat().st_mode | stat.S_IXUSR)

    bootstrap_json = tmp_path / "bootstrap.json"
    bootstrap_markdown = tmp_path / "bootstrap.md"
    pack_json = tmp_path / f"session_pack_{client_label}.json"
    pack_markdown = tmp_path / f"session_pack_{client_label}.md"
    skills_json = tmp_path / f"session_skills_{client_label}.json"
    prompt_json = tmp_path / f"session_system_prompt_{client_label}.json"
    prompt_text = tmp_path / f"session_system_prompt_{client_label}.txt"
    payload_json = tmp_path / f"session_client_{client_label}.json"
    context_root = tmp_path / "context"
    pack_json.write_text("{}", encoding="utf-8")
    pack_markdown.write_text("# pack\n", encoding="utf-8")
    skills_json.write_text("{}", encoding="utf-8")
    prompt_json.write_text("{}", encoding="utf-8")
    prompt_text.write_text(f"# {client_label} system prompt\n\nUse AFS context.\n", encoding="utf-8")
    payload_json.write_text("{}", encoding="utf-8")
    _write_fake_afs_cli(
        root,
        bootstrap_json,
        bootstrap_markdown,
        pack_json,
        pack_markdown,
        skills_json,
        prompt_json,
        prompt_text,
        payload_json,
        context_root,
    )

    client_log = tmp_path / "client-log.json"
    client = tmp_path / "fake-client"
    _write_fake_client(client, client_log)
    afs_log = tmp_path / "afs-log.txt"

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    env["FAKE_AFS_LOG"] = str(afs_log)
    wrapper_args = wrapper_args or []
    client_args = client_args or ["ping"]

    result = subprocess.run(
        [
            "bash",
            str(copied),
            client_label,
            str(client),
            "FAKE_CLIENT_CMD",
            *wrapper_args,
            "--",
            *client_args,
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == expected_returncode, result.stderr or result.stdout
    payload = json.loads(client_log.read_text(encoding="utf-8"))
    payload["_afs_calls"] = afs_log.read_text(encoding="utf-8").splitlines() if afs_log.exists() else []
    payload["_workspace"] = str(workspace)
    payload["_returncode"] = result.returncode
    return payload


def test_afs_shell_init_has_valid_bash_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SHELL_INIT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_afs_shell_init_exposes_afs_dev_alias(tmp_path: Path) -> None:
    fake_root = tmp_path / "afs-root"
    (fake_root / "scripts").mkdir(parents=True)

    result = subprocess.run(
        [
            "bash",
            "-lc",
            f"AFS_ROOT={shlex.quote(str(fake_root))} "
            f"source {shlex.quote(str(SHELL_INIT))} && alias afs-dev",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert f"alias afs-dev='{fake_root / 'scripts' / 'afs'}'" in result.stdout.strip()


def test_afs_client_session_has_valid_bash_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(AFS_CLIENT_SESSION)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_afs_session_notify_has_valid_bash_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(AFS_SESSION_NOTIFY)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_afs_session_verify_has_valid_bash_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(AFS_SESSION_VERIFY)],
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


def test_afs_verify_shell_helper_delegates_to_session_verify(tmp_path: Path) -> None:
    fake_root = tmp_path / "afs-root"
    scripts_dir = fake_root / "scripts"
    scripts_dir.mkdir(parents=True)
    verify_log = tmp_path / "verify-log.txt"
    fake_verify = scripts_dir / "afs-session-verify"
    fake_verify.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" > \"$FAKE_VERIFY_LOG\"\n",
        encoding="utf-8",
    )
    fake_verify.chmod(fake_verify.stat().st_mode | stat.S_IXUSR)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".context").mkdir()

    env = os.environ.copy()
    env["AFS_ROOT"] = str(fake_root)
    env["AFS_VENV"] = str(fake_root / ".venv")
    env["FAKE_VERIFY_LOG"] = str(verify_log)

    command = (
        f"source {shlex.quote(str(SHELL_INIT))} && "
        f"cd {shlex.quote(str(workspace))} && "
        "afs-verify --summary 'ran pytest' -- python3 -V"
    )
    result = subprocess.run(
        ["bash", "-lc", command],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert verify_log.read_text(encoding="utf-8").strip() == "--summary ran pytest -- python3 -V"


def test_afs_session_verify_records_passed_event(tmp_path: Path) -> None:
    root = tmp_path / "afs-copy"
    scripts_dir = root / "scripts"
    scripts_dir.mkdir(parents=True)

    copied_verify = scripts_dir / "afs-session-verify"
    copied_verify.write_text(AFS_SESSION_VERIFY.read_text(encoding="utf-8"), encoding="utf-8")
    copied_verify.chmod(copied_verify.stat().st_mode | stat.S_IXUSR)
    copied_notify = scripts_dir / "afs-session-notify"
    copied_notify.write_text(AFS_SESSION_NOTIFY.read_text(encoding="utf-8"), encoding="utf-8")
    copied_notify.chmod(copied_notify.stat().st_mode | stat.S_IXUSR)

    bootstrap_json = tmp_path / "bootstrap.json"
    bootstrap_markdown = tmp_path / "bootstrap.md"
    pack_json = tmp_path / "session_pack_codex.json"
    pack_markdown = tmp_path / "session_pack_codex.md"
    skills_json = tmp_path / "session_skills_codex.json"
    prompt_json = tmp_path / "session_system_prompt_codex.json"
    prompt_text = tmp_path / "session_system_prompt_codex.txt"
    payload_json = tmp_path / "session_client_codex.json"
    context_root = tmp_path / "context"
    _write_fake_afs_cli(
        root,
        bootstrap_json,
        bootstrap_markdown,
        pack_json,
        pack_markdown,
        skills_json,
        prompt_json,
        prompt_text,
        payload_json,
        context_root,
    )

    afs_log = tmp_path / "afs-log.txt"
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    env = os.environ.copy()
    env["AFS_ROOT"] = str(root)
    env["AFS_CLI"] = str(root / "scripts" / "afs")
    env["AFS_SESSION_NOTIFY"] = str(copied_notify)
    env["AFS_SESSION_CLIENT"] = "codex"
    env["AFS_SESSION_ID"] = "sess-verify"
    env["AFS_SESSION_CLIENT_PAYLOAD_JSON"] = str(payload_json)
    env["FAKE_AFS_LOG"] = str(afs_log)

    result = subprocess.run(
        ["bash", str(copied_verify), "--summary", "pytest clean", "--", "python3", "-c", "raise SystemExit(0)"],
        cwd=workspace,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    calls = afs_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 1
    assert calls[0].startswith(
        "session event verification_recorded "
        f"--cwd {workspace} "
        "--client codex "
        "--session-id sess-verify "
        f"--payload-file {payload_json} "
    )
    assert "--verification-status passed " in calls[0]
    assert "--summary pytest clean" in calls[0]


def test_afs_session_verify_records_failed_event_and_returns_command_status(tmp_path: Path) -> None:
    root = tmp_path / "afs-copy"
    scripts_dir = root / "scripts"
    scripts_dir.mkdir(parents=True)

    copied_verify = scripts_dir / "afs-session-verify"
    copied_verify.write_text(AFS_SESSION_VERIFY.read_text(encoding="utf-8"), encoding="utf-8")
    copied_verify.chmod(copied_verify.stat().st_mode | stat.S_IXUSR)
    copied_notify = scripts_dir / "afs-session-notify"
    copied_notify.write_text(AFS_SESSION_NOTIFY.read_text(encoding="utf-8"), encoding="utf-8")
    copied_notify.chmod(copied_notify.stat().st_mode | stat.S_IXUSR)

    bootstrap_json = tmp_path / "bootstrap.json"
    bootstrap_markdown = tmp_path / "bootstrap.md"
    pack_json = tmp_path / "session_pack_codex.json"
    pack_markdown = tmp_path / "session_pack_codex.md"
    skills_json = tmp_path / "session_skills_codex.json"
    prompt_json = tmp_path / "session_system_prompt_codex.json"
    prompt_text = tmp_path / "session_system_prompt_codex.txt"
    payload_json = tmp_path / "session_client_codex.json"
    context_root = tmp_path / "context"
    _write_fake_afs_cli(
        root,
        bootstrap_json,
        bootstrap_markdown,
        pack_json,
        pack_markdown,
        skills_json,
        prompt_json,
        prompt_text,
        payload_json,
        context_root,
    )

    afs_log = tmp_path / "afs-log.txt"
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    env = os.environ.copy()
    env["AFS_ROOT"] = str(root)
    env["AFS_CLI"] = str(root / "scripts" / "afs")
    env["AFS_SESSION_NOTIFY"] = str(copied_notify)
    env["AFS_SESSION_CLIENT"] = "codex"
    env["AFS_SESSION_ID"] = "sess-verify"
    env["AFS_SESSION_CLIENT_PAYLOAD_JSON"] = str(payload_json)
    env["FAKE_AFS_LOG"] = str(afs_log)

    result = subprocess.run(
        ["bash", str(copied_verify), "--", "python3", "-c", "raise SystemExit(7)"],
        cwd=workspace,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 7
    calls = afs_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 1
    assert "--verification-status failed " in calls[0]
    assert "--reason verification_failed" in calls[0]


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


def test_afs_client_session_uses_client_specific_allowed_roots(tmp_path: Path) -> None:
    payload = _run_client_session(
        tmp_path,
        env_overrides={"AFS_GEMINI_MCP_ALLOWED_ROOTS": "/workspaces/company"},
    )

    assert payload["AFS_MCP_ALLOWED_ROOTS"] == "/workspaces/company"
    assert payload["AFS_SESSION_BOOTSTRAP_JSON"].endswith("bootstrap.json")
    assert payload["AFS_SESSION_BOOTSTRAP_MARKDOWN"].endswith("bootstrap.md")
    assert payload["AFS_SESSION_PACK_JSON"].endswith("session_pack_gemini.json")
    assert payload["AFS_SESSION_PACK_MARKDOWN"].endswith("session_pack_gemini.md")
    assert payload["AFS_SESSION_SKILLS_JSON"].endswith("session_skills_gemini.json")
    assert payload["AFS_SESSION_SYSTEM_PROMPT_JSON"].endswith("session_system_prompt_gemini.json")
    assert payload["AFS_SESSION_SYSTEM_PROMPT_TEXT"].endswith("session_system_prompt_gemini.txt")
    assert payload["AFS_SESSION_CLIENT_PAYLOAD_JSON"].endswith("session_client_gemini.json")
    assert payload["AFS_ACTIVE_CONTEXT_ROOT"].endswith("context")
    assert payload["AFS_SESSION_QUERY_HINT"] == f"afs query <text> --path {payload['_workspace']}"
    assert (
        payload["AFS_SESSION_CONTEXT_QUERY_HINT"]
        == f"afs context query <text> --path {payload['_workspace']}"
    )
    assert payload["AFS_SESSION_INDEX_REBUILD_HINT"] == f"afs index rebuild --path {payload['_workspace']}"
    assert payload["GEMINI_SYSTEM_MD"] == payload["AFS_SESSION_SYSTEM_PROMPT_TEXT"]


def test_afs_client_session_preserves_explicit_allowed_roots(tmp_path: Path) -> None:
    payload = _run_client_session(
        tmp_path,
        env_overrides={
            "AFS_MCP_ALLOWED_ROOTS": "/already/set",
            "AFS_GEMINI_MCP_ALLOWED_ROOTS": "/workspaces/company",
            "AFS_CLIENT_MCP_ALLOWED_ROOTS": "/workspaces/shared",
        },
    )

    assert payload["AFS_MCP_ALLOWED_ROOTS"] == "/already/set"


def test_afs_client_session_waits_for_session_agents_by_default(tmp_path: Path) -> None:
    payload = _run_client_session(tmp_path)

    assert payload["AFS_SESSION_ID"]
    assert any(call.startswith("session prepare-client --client gemini ") for call in payload["_afs_calls"])
    assert any(call.startswith("session hook session_start --client gemini ") for call in payload["_afs_calls"])
    assert not any(call.startswith("session event ") for call in payload["_afs_calls"])
    assert any(
        call == f"agents monitor --session-id {payload['AFS_SESSION_ID']}"
        for call in payload["_afs_calls"]
    )
    assert any(
        call.startswith(f"agents wait --all --session-id {payload['AFS_SESSION_ID']} --timeout ")
        for call in payload["_afs_calls"]
    )
    assert any(call.startswith("session hook session_end --client gemini ") for call in payload["_afs_calls"])


def test_afs_client_session_can_disable_agent_drain(tmp_path: Path) -> None:
    payload = _run_client_session(
        tmp_path,
        env_overrides={"AFS_CLIENT_WAIT_FOR_AGENTS": "0"},
    )

    assert any(call.startswith("session hook session_start ") for call in payload["_afs_calls"])
    assert any(call.startswith("agents monitor ") for call in payload["_afs_calls"])
    assert not any(call.startswith("agents wait ") for call in payload["_afs_calls"])


def test_afs_client_session_can_disable_live_monitor(tmp_path: Path) -> None:
    payload = _run_client_session(
        tmp_path,
        env_overrides={"AFS_CLIENT_MONITOR_AGENTS": "0"},
    )

    assert not any(call.startswith("agents monitor ") for call in payload["_afs_calls"])
    assert any(call.startswith("agents wait ") for call in payload["_afs_calls"])


def test_afs_client_session_can_disable_session_pack_and_skill_match(tmp_path: Path) -> None:
    payload = _run_client_session(
        tmp_path,
        env_overrides={
            "AFS_CLIENT_SESSION_PACK": "0",
            "AFS_CLIENT_SKILLS_MATCH": "0",
        },
    )

    prepare_call = next(
        call for call in payload["_afs_calls"] if call.startswith("session prepare-client ")
    )
    assert "--no-session-pack" in prepare_call
    assert "--no-skills-match" in prepare_call


def test_afs_client_session_injects_claude_prompt_file(tmp_path: Path) -> None:
    payload = _run_client_session(
        tmp_path,
        client_label="claude",
    )

    assert payload["args"][:2] == [
        "--append-system-prompt-file",
        payload["AFS_SESSION_SYSTEM_PROMPT_TEXT"],
    ]
    assert payload["args"][2:] == ["ping"]


def test_afs_client_session_respects_explicit_claude_prompt_args(tmp_path: Path) -> None:
    payload = _run_client_session(
        tmp_path,
        client_label="claude",
        client_args=["--system-prompt", "user override", "ping"],
    )

    assert payload["args"] == ["--system-prompt", "user override", "ping"]


def test_afs_client_session_injects_codex_model_instructions_file(tmp_path: Path) -> None:
    payload = _run_client_session(
        tmp_path,
        client_label="codex",
    )

    assert payload["args"][0] == "-c"
    assert payload["args"][1] == f'model_instructions_file="{payload["AFS_SESSION_SYSTEM_PROMPT_TEXT"]}"'
    assert payload["args"][2:] == ["ping"]


def test_afs_client_session_respects_explicit_codex_prompt_config(tmp_path: Path) -> None:
    payload = _run_client_session(
        tmp_path,
        client_label="codex",
        client_args=["-c", 'model_instructions_file="/tmp/custom.md"', "ping"],
    )

    assert payload["args"] == ["-c", 'model_instructions_file="/tmp/custom.md"', "ping"]


def test_afs_client_session_respects_explicit_gemini_system_prompt_env(tmp_path: Path) -> None:
    payload = _run_client_session(
        tmp_path,
        env_overrides={"GEMINI_SYSTEM_MD": "/tmp/custom-system.md"},
    )

    assert payload["GEMINI_SYSTEM_MD"] == "/tmp/custom-system.md"


def test_afs_session_notify_uses_session_env_defaults(tmp_path: Path) -> None:
    root = tmp_path / "afs-copy"
    scripts_dir = root / "scripts"
    scripts_dir.mkdir(parents=True)

    copied_notify = scripts_dir / "afs-session-notify"
    copied_notify.write_text(AFS_SESSION_NOTIFY.read_text(encoding="utf-8"), encoding="utf-8")
    copied_notify.chmod(copied_notify.stat().st_mode | stat.S_IXUSR)

    bootstrap_json = tmp_path / "bootstrap.json"
    bootstrap_markdown = tmp_path / "bootstrap.md"
    pack_json = tmp_path / "session_pack_codex.json"
    pack_markdown = tmp_path / "session_pack_codex.md"
    skills_json = tmp_path / "session_skills_codex.json"
    prompt_json = tmp_path / "session_system_prompt_codex.json"
    prompt_text = tmp_path / "session_system_prompt_codex.txt"
    payload_json = tmp_path / "session_client_codex.json"
    context_root = tmp_path / "context"
    _write_fake_afs_cli(
        root,
        bootstrap_json,
        bootstrap_markdown,
        pack_json,
        pack_markdown,
        skills_json,
        prompt_json,
        prompt_text,
        payload_json,
        context_root,
    )

    afs_log = tmp_path / "afs-log.txt"
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    env = os.environ.copy()
    env["AFS_ROOT"] = str(root)
    env["AFS_CLI"] = str(root / "scripts" / "afs")
    env["AFS_SESSION_CLIENT"] = "codex"
    env["AFS_SESSION_ID"] = "sess-123"
    env["AFS_SESSION_CLIENT_PAYLOAD_JSON"] = str(payload_json)
    env["AFS_SESSION_DEFAULT_TURN_ID"] = "turn-123"
    env["FAKE_AFS_LOG"] = str(afs_log)

    result = subprocess.run(
        [
            "bash",
            str(copied_notify),
            "task_created",
            "--task-id",
            "bg-1",
            "--task-title",
            "Index context",
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert afs_log.read_text(encoding="utf-8").splitlines() == [
        (
            "session event task_created "
            f"--cwd {workspace} "
            "--client codex "
            "--session-id sess-123 "
            f"--payload-file {payload_json} "
            "--turn-id turn-123 "
            "--task-id bg-1 "
            "--task-title Index context"
        )
    ]


def test_afs_client_session_emits_prompt_and_turn_events(tmp_path: Path) -> None:
    prompt = "Investigate live harness session updates."
    payload = _run_client_session(
        tmp_path,
        wrapper_args=["--prompt", prompt],
    )

    turn_id = str(payload["AFS_SESSION_DEFAULT_TURN_ID"])
    assert payload["AFS_SESSION_EVENT_BIN"].endswith("scripts/afs-session-notify")
    assert turn_id.startswith("turn-")

    calls = payload["_afs_calls"]
    prepare_call = next(
        call for call in calls if call.startswith("session prepare-client ")
    )
    session_start_index = next(
        index for index, call in enumerate(calls) if call.startswith("session hook session_start ")
    )
    prompt_index = next(
        index for index, call in enumerate(calls) if call.startswith("session event user_prompt_submit ")
    )
    turn_started_index = next(
        index for index, call in enumerate(calls) if call.startswith("session event turn_started ")
    )
    turn_completed_index = next(
        index for index, call in enumerate(calls) if call.startswith("session event turn_completed ")
    )
    session_end_index = next(
        index for index, call in enumerate(calls) if call.startswith("session hook session_end ")
    )

    assert f"--query {prompt}" in prepare_call
    assert f"--task {prompt}" in prepare_call
    assert f"--skills-prompt {prompt}" in prepare_call
    assert session_start_index < prompt_index < turn_started_index < turn_completed_index < session_end_index
    assert any(
        call.startswith(
            "session event user_prompt_submit "
            f"--cwd {payload['_workspace']} "
            "--client gemini "
            f"--session-id {payload['AFS_SESSION_ID']} "
        )
        and f"--turn-id {turn_id} " in call
        and f"--prompt {prompt}" in call
        for call in calls
    )
    assert any(
        call.startswith(
            "session event turn_started "
            f"--cwd {payload['_workspace']} "
            "--client gemini "
            f"--session-id {payload['AFS_SESSION_ID']} "
        )
        and f"--turn-id {turn_id}" in call
        for call in calls
    )
    assert any(
        call.startswith(
            "session event turn_completed "
            f"--cwd {payload['_workspace']} "
            "--client gemini "
            f"--session-id {payload['AFS_SESSION_ID']} "
        )
        and f"--turn-id {turn_id} " in call
        and "--exit-code 0 " in call
        and "--reason client_exit" in call
        for call in calls
    )


def test_afs_client_session_can_fail_on_strict_verification_gate(tmp_path: Path) -> None:
    payload = _run_client_session(
        tmp_path,
        env_overrides={
            "AFS_CLIENT_SESSION_VERIFICATION_MODE": "error",
            "FAKE_SESSION_END_HOOK_EXIT": "2",
        },
        expected_returncode=2,
    )

    assert payload["_returncode"] == 2
    assert any(
        call.startswith("session hook session_end --client gemini ")
        and "--verification-mode error " in call
        for call in payload["_afs_calls"]
    )
