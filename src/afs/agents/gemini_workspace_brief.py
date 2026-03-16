"""Generate a Gemini-powered brief for configured workspaces and contexts."""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from collections.abc import Sequence
from pathlib import Path

from ..agent.models import ModelConfig, ModelProvider, create_backend
from ..models import ContextRoot, MountType
from .base import (
    AgentResult,
    build_base_parser,
    configure_logging,
    emit_result,
    load_agent_config,
    now_iso,
    resolve_contexts,
)

AGENT_NAME = "gemini-workspace-brief"
AGENT_DESCRIPTION = "Use Gemini to summarize configured workspaces and discovered contexts."
DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_SYSTEM_PROMPT = (
    "You are preparing a concise operational brief for Gemini CLI users working in AFS-managed "
    "repositories. Return compact markdown with sections 'Snapshot', 'Risks', and 'Next Actions'. "
    "Prefer concrete path-oriented guidance over general advice."
)


def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser("Generate a Gemini-powered brief for workspaces and contexts.")
    parser.add_argument(
        "--path",
        action="append",
        help="Workspace root to scan for contexts (repeatable). Defaults to configured workspace roots.",
    )
    parser.add_argument(
        "--context-path",
        action="append",
        help="Explicit .context path to include (repeatable).",
    )
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum discovery depth.")
    parser.add_argument("--ignore", action="append", help="Directory names to ignore.")
    parser.add_argument(
        "--model",
        default=os.environ.get("AFS_GEMINI_WORKSPACE_BRIEF_MODEL", DEFAULT_MODEL),
        help=f"Gemini model ID (default: env AFS_GEMINI_WORKSPACE_BRIEF_MODEL or {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--system-prompt",
        default=os.environ.get("AFS_GEMINI_WORKSPACE_BRIEF_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        help="Override system prompt.",
    )
    parser.add_argument(
        "--markdown-output",
        help="Write the markdown brief to this file.",
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=25,
        help="Maximum contexts to include in the prompt (default: 25).",
    )
    parser.add_argument(
        "--max-workspaces",
        type=int,
        default=10,
        help="Maximum workspace roots to include in the prompt (default: 10).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=0,
        help="Seconds between runs (0 = run once).",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Maximum runs when interval > 0 (0 = unlimited).",
    )
    parser.add_argument(
        "--sleep-first",
        action="store_true",
        help="Sleep for the interval before the first run.",
    )
    return parser


def _resolve_workspace_roots(args: argparse.Namespace, config) -> list[Path]:
    candidates: list[Path] = []
    if args.path:
        candidates.extend(Path(path).expanduser().resolve() for path in args.path)
    else:
        candidates.extend(workspace.path for workspace in config.general.workspace_directories)
        candidates.extend(config.general.mcp_allowed_roots)

    unique: list[Path] = []
    seen: set[Path] = set()
    for root in candidates:
        resolved = root.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _context_payload(contexts: list[ContextRoot], *, limit: int) -> tuple[list[dict[str, object]], list[str]]:
    notes: list[str] = []
    ordered = sorted(contexts, key=lambda item: str(item.path))
    if limit > 0 and len(ordered) > limit:
        ordered = ordered[:limit]
        notes.append("context list truncated by --max-contexts")

    payload: list[dict[str, object]] = []
    for context in ordered:
        mount_counts = {
            mount_type.value: len(context.mounts.get(mount_type, []))
            for mount_type in MountType
            if context.mounts.get(mount_type, [])
        }
        payload.append(
            {
                "project": context.project_name,
                "path": str(context.path),
                "valid": context.is_valid,
                "total_mounts": context.total_mounts,
                "mounts": mount_counts,
                "description": context.metadata.description or "",
            }
        )
    return payload, notes


def _build_prompt(
    *,
    workspace_roots: list[Path],
    contexts: list[dict[str, object]],
) -> str:
    lines = [
        "Prepare a brief for Gemini CLI workspace operations.",
        "",
        "Workspace roots:",
    ]
    if workspace_roots:
        for root in workspace_roots:
            lines.append(f"- {root}")
    else:
        lines.append("- (none configured)")

    lines.append("")
    lines.append("Contexts:")
    if contexts:
        for context in contexts:
            mounts = context.get("mounts") or {}
            mount_text = ", ".join(
                f"{name}={count}" for name, count in sorted(mounts.items())
            ) or "no mounts"
            lines.append(
                "- "
                + f"{context['project']} | path={context['path']} | "
                + f"valid={context['valid']} | mounts={context['total_mounts']} | {mount_text}"
            )
    else:
        lines.append("- (no contexts discovered)")

    lines.append("")
    lines.append(
        "Focus on what a Gemini CLI user should inspect or initialize next. "
        "Call out invalid contexts and missing workspace coverage."
    )
    return "\n".join(lines)


def _resolve_gemini_key_name() -> str | None:
    if os.environ.get("GEMINI_API_KEY"):
        return "GEMINI_API_KEY"
    if os.environ.get("GOOGLE_API_KEY"):
        return "GOOGLE_API_KEY"
    return None


async def _generate_brief(prompt: str, model: str, system_prompt: str) -> tuple[str, dict[str, int]]:
    backend = create_backend(
        ModelConfig(
            provider=ModelProvider.GEMINI,
            model_id=model,
            system_prompt=system_prompt,
        )
    )
    try:
        result = await backend.generate([{"role": "user", "content": prompt}])
    finally:
        await backend.close()
    return result.content, dict(result.usage)


def _write_markdown(path: Path | None, content: str) -> Path | None:
    if path is None:
        return None
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _run_once(args: argparse.Namespace) -> tuple[AgentResult, int]:
    config = load_agent_config(args.config)
    workspace_roots = _resolve_workspace_roots(args, config)
    contexts = resolve_contexts(
        config,
        context_paths=args.context_path,
        search_paths=[str(path) for path in workspace_roots],
        max_depth=args.max_depth,
        ignore_names=args.ignore,
    )

    started_at = now_iso()
    start = time.monotonic()
    notes: list[str] = []

    context_payload, context_notes = _context_payload(contexts, limit=args.max_contexts)
    notes.extend(context_notes)

    if args.max_workspaces > 0 and len(workspace_roots) > args.max_workspaces:
        workspace_roots = workspace_roots[: args.max_workspaces]
        notes.append("workspace root list truncated by --max-workspaces")

    key_name = _resolve_gemini_key_name()
    markdown = ""
    usage: dict[str, int] = {}
    status = "ok"
    exit_code = 0

    if key_name is None:
        status = "error"
        exit_code = 1
        notes.append("Set GEMINI_API_KEY or GOOGLE_API_KEY to enable Gemini background briefs.")
    else:
        prompt = _build_prompt(workspace_roots=workspace_roots, contexts=context_payload)
        try:
            markdown, usage = asyncio.run(
                _generate_brief(prompt=prompt, model=args.model, system_prompt=args.system_prompt)
            )
        except Exception as exc:
            status = "error"
            exit_code = 1
            notes.append(str(exc))

    markdown_output = _write_markdown(
        Path(args.markdown_output) if args.markdown_output else None,
        markdown,
    )
    result = AgentResult(
        name=AGENT_NAME,
        status=status if markdown or status != "ok" else "warn",
        started_at=started_at,
        finished_at=now_iso(),
        duration_seconds=time.monotonic() - start,
        metrics={
            "workspace_roots": len(workspace_roots),
            "contexts": len(context_payload),
            "invalid_contexts": sum(1 for context in context_payload if not context["valid"]),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        },
        notes=notes,
        payload={
            "model": args.model,
            "api_key_env": key_name,
            "workspace_roots": [str(path) for path in workspace_roots],
            "contexts": context_payload,
            "brief_markdown": markdown,
            "markdown_output": str(markdown_output) if markdown_output else None,
        },
    )
    return result, exit_code


def run(args: argparse.Namespace) -> int:
    configure_logging(args.quiet)
    runs = 0

    if args.interval > 0 and args.sleep_first:
        time.sleep(args.interval)

    while True:
        runs += 1
        result, exit_code = _run_once(args)
        emit_result(
            result,
            output_path=Path(args.output) if args.output else None,
            force_stdout=args.stdout,
            pretty=args.pretty,
        )
        if args.interval <= 0:
            return exit_code
        if args.max_runs > 0 and runs >= args.max_runs:
            return exit_code
        time.sleep(args.interval)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
