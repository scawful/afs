"""Generic training dataset and run lifecycle helpers."""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..models import MountType


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower())
    slug = slug.strip("-._")
    return slug or "artifact"


def preview(text: str, limit: int = 180) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text") or item.get("content")
            if isinstance(text, str):
                parts.append(text)
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        text = value.get("text") or value.get("content")
        if isinstance(text, str):
            return text
    return ""


def analyze_row(row: dict[str, Any]) -> dict[str, Any]:
    roles: dict[str, int] = {}
    tool_names: list[str] = []
    user_chars = 0
    assistant_chars = 0
    max_message_chars = 0
    preview_text = ""
    total_chars = 0
    message_count = 0

    if isinstance(row.get("messages"), list):
        for message in row["messages"]:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "unknown")
            content = flatten_text(message.get("content"))
            reasoning = flatten_text(message.get("reasoning_content"))
            text = "\n".join(part for part in [content, reasoning] if part)
            text_len = len(text)
            total_chars += text_len
            max_message_chars = max(max_message_chars, text_len)
            message_count += 1
            roles[role] = roles.get(role, 0) + 1
            if role == "user":
                user_chars += text_len
                if not preview_text and text:
                    preview_text = preview(text)
            elif role == "assistant":
                assistant_chars += text_len
            elif role == "tool":
                assistant_chars += text_len

            for call in message.get("tool_calls", []) or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if isinstance(function, dict) and isinstance(function.get("name"), str):
                    tool_names.append(function["name"])
                elif isinstance(call.get("name"), str):
                    tool_names.append(call["name"])

    elif isinstance(row.get("prompt"), list) and (
        isinstance(row.get("chosen"), list) or isinstance(row.get("rejected"), list)
    ):
        prompt_messages = row.get("prompt") or []
        chosen_messages = row.get("chosen") or []
        rejected_messages = row.get("rejected") or []

        def _sum_messages(messages: list[Any], role_name: str) -> int:
            total = 0
            for message in messages:
                if not isinstance(message, dict):
                    continue
                text = flatten_text(message.get("content"))
                total += len(text)
                roles[role_name] = roles.get(role_name, 0) + 1
            return total

        user_chars = _sum_messages(prompt_messages, "prompt")
        assistant_chars = _sum_messages(chosen_messages, "chosen") + _sum_messages(
            rejected_messages, "rejected"
        )
        total_chars = user_chars + assistant_chars
        max_message_chars = max(
            [0]
            + [len(flatten_text(m.get("content"))) for m in prompt_messages if isinstance(m, dict)]
            + [len(flatten_text(m.get("content"))) for m in chosen_messages if isinstance(m, dict)]
            + [len(flatten_text(m.get("content"))) for m in rejected_messages if isinstance(m, dict)]
        )
        message_count = len(prompt_messages) + len(chosen_messages) + len(rejected_messages)
        for message in prompt_messages:
            if not isinstance(message, dict):
                continue
            text = flatten_text(message.get("content"))
            if text:
                preview_text = preview(text)
                break
    elif isinstance(row.get("prompt"), str) or isinstance(row.get("completion"), str):
        prompt_text = flatten_text(row.get("prompt"))
        completion_text = flatten_text(row.get("completion"))
        total_chars = len(prompt_text) + len(completion_text)
        user_chars = len(prompt_text)
        assistant_chars = len(completion_text)
        max_message_chars = max(user_chars, assistant_chars)
        message_count = 2
        roles = {"user": 1, "assistant": 1}
        preview_text = preview(prompt_text)
    elif isinstance(row.get("text"), str):
        text = row["text"]
        total_chars = len(text)
        max_message_chars = total_chars
        message_count = 1
        roles = {"text": 1}
        preview_text = preview(text)

    return {
        "total_chars": total_chars,
        "user_chars": user_chars,
        "assistant_chars": assistant_chars,
        "max_message_chars": max_message_chars,
        "message_count": message_count,
        "role_counts": roles,
        "tool_call_count": len(tool_names),
        "tool_names": tool_names,
        "preview": preview_text,
    }


def dataset_sources(dataset_path: Path) -> list[tuple[str, Path]]:
    if dataset_path.is_file():
        return [(dataset_path.stem, dataset_path)]

    preferred = ["train.jsonl", "valid.jsonl", "test.jsonl", "dpo_pairs.jsonl"]
    sources: list[tuple[str, Path]] = []
    for name in preferred:
        candidate = dataset_path / name
        if candidate.exists():
            sources.append((candidate.stem, candidate))
    if sources:
        return sources

    return [(path.stem, path) for path in sorted(dataset_path.glob("*.jsonl"))]


def dataset_id_for_path(dataset_path: Path, explicit_name: str | None = None) -> str:
    if explicit_name:
        return slugify(explicit_name)
    if dataset_path.is_file():
        return slugify(dataset_path.stem)
    return slugify(dataset_path.name)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def build_dataset_stats(dataset_path: Path) -> dict[str, Any]:
    splits: list[dict[str, Any]] = []
    overall_role_counts: dict[str, int] = {}
    overall_tool_counts: dict[str, int] = {}
    total_rows = 0
    max_chars = 0

    for split_name, split_path in dataset_sources(dataset_path):
        rows = 0
        invalid_rows = 0
        total_chars = 0
        split_max = 0
        role_counts: dict[str, int] = {}
        tool_counts: dict[str, int] = {}
        with split_path.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    invalid_rows += 1
                    continue
                metrics = analyze_row(row if isinstance(row, dict) else {})
                rows += 1
                total_rows += 1
                total_chars += metrics["total_chars"]
                split_max = max(split_max, metrics["total_chars"])
                max_chars = max(max_chars, metrics["total_chars"])
                for role, count in metrics["role_counts"].items():
                    role_counts[role] = role_counts.get(role, 0) + count
                    overall_role_counts[role] = overall_role_counts.get(role, 0) + count
                for name in metrics["tool_names"]:
                    tool_counts[name] = tool_counts.get(name, 0) + 1
                    overall_tool_counts[name] = overall_tool_counts.get(name, 0) + 1

        splits.append(
            {
                "split": split_name,
                "path": str(split_path),
                "rows": rows,
                "invalid_rows": invalid_rows,
                "avg_chars": round(total_chars / rows, 2) if rows else 0,
                "max_chars": split_max,
                "role_counts": role_counts,
                "tool_counts": tool_counts,
            }
        )

    return {
        "generated_at": now_iso(),
        "dataset_path": str(dataset_path),
        "dataset_id": dataset_id_for_path(dataset_path),
        "total_rows": total_rows,
        "max_chars": max_chars,
        "splits": splits,
        "overall_role_counts": overall_role_counts,
        "overall_tool_counts": overall_tool_counts,
    }


def render_dataset_stats_markdown(stats: dict[str, Any]) -> str:
    lines = ["# Dataset Stats", "", f"- dataset: `{stats['dataset_path']}`", f"- generated_at: `{stats['generated_at']}`", f"- total_rows: `{stats['total_rows']}`", f"- max_chars: `{stats['max_chars']}`", "", "## Splits", ""]
    for split in stats["splits"]:
        lines.extend(
            [
                f"### {split['split']}",
                f"- path: `{split['path']}`",
                f"- rows: `{split['rows']}`",
                f"- invalid_rows: `{split['invalid_rows']}`",
                f"- avg_chars: `{split['avg_chars']}`",
                f"- max_chars: `{split['max_chars']}`",
            ]
        )
        if split["tool_counts"]:
            counts = ", ".join(f"`{name}`={count}" for name, count in sorted(split["tool_counts"].items()))
            lines.append(f"- tool_counts: {counts}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_dataset_outliers(dataset_path: Path, limit: int = 10) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for split_name, split_path in dataset_sources(dataset_path):
        with split_path.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                metrics = analyze_row(row)
                rows.append(
                    {
                        "split": split_name,
                        "path": str(split_path),
                        "line": line_number,
                        **metrics,
                        "metadata": row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
                    }
                )
    rows.sort(key=lambda item: item["total_chars"], reverse=True)
    return {
        "generated_at": now_iso(),
        "dataset_path": str(dataset_path),
        "dataset_id": dataset_id_for_path(dataset_path),
        "limit": limit,
        "rows": rows[:limit],
    }


def render_dataset_outliers_markdown(report: dict[str, Any]) -> str:
    lines = ["# Dataset Outliers", "", f"- dataset: `{report['dataset_path']}`", f"- generated_at: `{report['generated_at']}`", f"- limit: `{report['limit']}`", ""]
    for idx, row in enumerate(report["rows"], 1):
        lines.extend(
            [
                f"## {idx}. {row['split']}:{row['line']}",
                f"- total_chars: `{row['total_chars']}`",
                f"- max_message_chars: `{row['max_message_chars']}`",
                f"- tool_calls: `{row['tool_call_count']}`",
            ]
        )
        if row["tool_names"]:
            lines.append("- tool_names: " + ", ".join(f"`{name}`" for name in row["tool_names"]))
        if row["metadata"]:
            lines.append("- metadata: `" + preview(json.dumps(row["metadata"], ensure_ascii=True), 220) + "`")
        if row["preview"]:
            lines.append(f"- preview: {row['preview']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def dataset_artifact_root(manager: Any, context_path: Path, dataset_id: str) -> Path:
    scratchpad_root = manager.resolve_mount_root(context_path, MountType.SCRATCHPAD)
    return scratchpad_root / "training" / "datasets" / dataset_id


def write_dataset_artifacts(
    artifact_root: Path,
    *,
    manifest: dict[str, Any],
    stats: dict[str, Any] | None = None,
    outliers: dict[str, Any] | None = None,
) -> None:
    artifact_root.mkdir(parents=True, exist_ok=True)
    _write_json(artifact_root / "manifest.json", manifest)
    if stats is not None:
        _write_json(artifact_root / "stats.json", stats)
        (artifact_root / "stats.md").write_text(
            render_dataset_stats_markdown(stats), encoding="utf-8"
        )
    if outliers is not None:
        _write_json(artifact_root / "outliers.json", outliers)
        (artifact_root / "outliers.md").write_text(
            render_dataset_outliers_markdown(outliers), encoding="utf-8"
        )


def load_run_spec(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        payload = json.loads(raw)
    elif path.suffix in {".toml", ".tml"}:
        payload = tomllib.loads(raw)
    else:
        raise ValueError("Run spec must be .json or .toml")
    if not isinstance(payload, dict):
        raise ValueError("Run spec must decode to an object")
    return payload


def normalize_run_spec(path: Path) -> dict[str, Any]:
    spec = load_run_spec(path)
    command = spec.get("command")
    shell_command = spec.get("shell_command")
    if command is None and shell_command is None:
        raise ValueError("Run spec must define `command` or `shell_command`")

    if shell_command is not None:
        if not isinstance(shell_command, str) or not shell_command.strip():
            raise ValueError("`shell_command` must be a non-empty string")
        normalized_command = ["/bin/bash", "-lc", shell_command]
    elif isinstance(command, str):
        normalized_command = shlex.split(command)
    elif isinstance(command, list) and all(isinstance(item, str) for item in command):
        normalized_command = command
    else:
        raise ValueError("`command` must be a string or a list of strings")

    if not normalized_command:
        raise ValueError("Run command cannot be empty")

    workdir = spec.get("workdir")
    if workdir is None:
        resolved_workdir = path.parent
    else:
        resolved_workdir = Path(workdir).expanduser()
        if not resolved_workdir.is_absolute():
            resolved_workdir = (path.parent / resolved_workdir).resolve()
        else:
            resolved_workdir = resolved_workdir.resolve()

    env = spec.get("env") or {}
    if not isinstance(env, dict):
        raise ValueError("`env` must be a mapping if provided")

    name = spec.get("name") or path.stem
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Run spec `name` must be a non-empty string")

    return {
        "name": name.strip(),
        "command": normalized_command,
        "workdir": str(resolved_workdir),
        "env": {str(key): str(value) for key, value in env.items()},
        "backend": spec.get("backend"),
        "dataset": spec.get("dataset"),
        "kind": spec.get("kind", "train"),
        "description": spec.get("description", ""),
        "tags": spec.get("tags", []),
        "spec_path": str(path.resolve()),
    }


def run_root(manager: Any, context_path: Path, run_id: str) -> Path:
    scratchpad_root = manager.resolve_mount_root(context_path, MountType.SCRATCHPAD)
    return scratchpad_root / "training" / "runs" / run_id


def status_path_for_run(run_dir: Path) -> Path:
    return run_dir / "status.json"


def load_run_status(run_dir: Path) -> dict[str, Any]:
    with status_path_for_run(run_dir).open(encoding="utf-8") as handle:
        return json.load(handle)


def process_snapshot(pid: int | None) -> dict[str, Any] | None:
    if not pid:
        return None
    result = subprocess.run(
        [
            "ps",
            "-p",
            str(pid),
            "-o",
            "pid=,ppid=,state=,etime=,%cpu=,%mem=,command=",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    line = result.stdout.strip()
    if not line:
        return None
    parts = line.split(None, 6)
    if len(parts) < 7:
        return None
    return {
        "pid": int(parts[0]),
        "ppid": int(parts[1]),
        "state": parts[2],
        "elapsed": parts[3],
        "cpu_percent": float(parts[4]),
        "mem_percent": float(parts[5]),
        "command": parts[6],
    }


def render_run_status_markdown(status: dict[str, Any]) -> str:
    lines = [
        "# Training Run Status",
        "",
        f"- run_id: `{status['run_id']}`",
        f"- name: `{status.get('name', '')}`",
        f"- status: `{status['status']}`",
        f"- started_at: `{status.get('started_at', '')}`",
        f"- updated_at: `{status.get('updated_at', '')}`",
        f"- workdir: `{status.get('workdir', '')}`",
        f"- log_path: `{status.get('log_path', '')}`",
    ]
    snapshot = status.get("process")
    if isinstance(snapshot, dict):
        lines.extend(
            [
                "",
                "## Process",
                "",
                f"- pid: `{snapshot.get('pid')}`",
                f"- elapsed: `{snapshot.get('elapsed')}`",
                f"- cpu_percent: `{snapshot.get('cpu_percent')}`",
                f"- mem_percent: `{snapshot.get('mem_percent')}`",
                f"- command: `{snapshot.get('command')}`",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_run_status(run_dir: Path, status: dict[str, Any]) -> None:
    _write_json(run_dir / "status.json", status)
    (run_dir / "status.md").write_text(render_run_status_markdown(status), encoding="utf-8")


def append_run_event(run_dir: Path, event_type: str, payload: dict[str, Any]) -> None:
    _append_jsonl(
        run_dir / "events.jsonl",
        {
            "timestamp": now_iso(),
            "event_type": event_type,
            "payload": payload,
        },
    )


def start_run(manager: Any, context_path: Path, spec_path: Path) -> dict[str, Any]:
    spec = normalize_run_spec(spec_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}-{slugify(spec['name'])}"
    run_dir = run_root(manager, context_path, run_id)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "stdout.log"

    env = os.environ.copy()
    env.update(spec["env"])
    env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            spec["command"],
            cwd=spec["workdir"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
            text=True,
        )

    status = {
        "run_id": run_id,
        "name": spec["name"],
        "kind": spec["kind"],
        "backend": spec.get("backend"),
        "dataset": spec.get("dataset"),
        "description": spec.get("description", ""),
        "tags": spec.get("tags", []),
        "status": "running",
        "created_at": now_iso(),
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "context_path": str(context_path),
        "spec_path": spec["spec_path"],
        "workdir": spec["workdir"],
        "command": spec["command"],
        "env_keys": sorted(spec["env"].keys()),
        "pid": proc.pid,
        "pgid": proc.pid,
        "log_path": str(log_path),
        "artifacts_path": str((run_dir / "artifacts.json").resolve()),
        "process": process_snapshot(proc.pid),
    }

    (run_dir / "normalized_spec.json").write_text(
        json.dumps(spec, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_json(run_dir / "artifacts.json", {"log_path": str(log_path)})
    write_run_status(run_dir, status)
    append_run_event(run_dir, "run_started", {"pid": proc.pid, "command": spec["command"]})
    return status


def refresh_run_status(run_dir: Path) -> dict[str, Any]:
    status = load_run_status(run_dir)
    snapshot = process_snapshot(status.get("pid"))
    status["updated_at"] = now_iso()
    status["process"] = snapshot
    if snapshot is None and status.get("status") == "running":
        status["status"] = "exited"
        status.setdefault("finished_at", now_iso())
        append_run_event(run_dir, "run_exited", {"pid": status.get("pid")})
    write_run_status(run_dir, status)
    return status


def stop_run(run_dir: Path, *, force: bool = False, timeout_seconds: float = 3.0) -> dict[str, Any]:
    status = load_run_status(run_dir)
    pgid = status.get("pgid")
    pid = status.get("pid")
    target = int(pgid or pid or 0)
    if not target:
        raise ValueError("Run status does not include a pid or pgid")

    try:
        os.killpg(target, signal.SIGTERM)
        signal_used = "SIGTERM"
    except ProcessLookupError:
        signal_used = "missing"

    deadline = time.time() + timeout_seconds
    snapshot = process_snapshot(status.get("pid"))
    while snapshot is not None and time.time() < deadline:
        time.sleep(0.2)
        snapshot = process_snapshot(status.get("pid"))

    if snapshot is not None and force:
        try:
            os.killpg(target, signal.SIGKILL)
            signal_used = "SIGKILL"
        except ProcessLookupError:
            pass
        time.sleep(0.2)
        snapshot = process_snapshot(status.get("pid"))

    status["updated_at"] = now_iso()
    status["process"] = snapshot
    status["stopped_at"] = now_iso()
    status["status"] = "stopped" if snapshot is None else "stop_timeout"
    status["stop_signal"] = signal_used
    write_run_status(run_dir, status)
    append_run_event(run_dir, "run_stopped", {"signal": signal_used, "force": force})
    return status
