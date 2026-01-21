"""Generate a draft response using the Scribe persona."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from ..gateway.backends import BackendManager
from ..gateway.server import PERSONAS
from .base import (
    AgentResult,
    build_base_parser,
    configure_logging,
    emit_result,
    now_iso,
)

AGENT_NAME = "scribe-draft"
AGENT_DESCRIPTION = "Draft responses using the Scribe persona."

MODEL_ID_MAP = {
    "din": "din-v2:latest",
    "nayru": "nayru-v5:latest",
    "farore": "farore-v1:latest",
    "veran": "qwen2.5-coder:7b",
    "scribe": "qwen2.5-coder:7b",
}


def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser("Draft responses using the Scribe persona.")
    parser.add_argument("--prompt", help="Prompt to send.")
    parser.add_argument("--model", default="scribe", help="Persona to use.")
    parser.add_argument("--model-id", help="Override backend model ID.")
    parser.add_argument("--output-text", help="Write response to this file.")
    return parser


async def _chat(prompt: str, model: str, model_id: str | None) -> dict:
    async with BackendManager() as manager:
        if not manager.active:
            return {"error": "No backend available"}

        persona = PERSONAS.get(model, PERSONAS.get("scribe"))
        if not persona:
            return {"error": f"Unknown persona: {model}"}

        resolved_model = model_id or MODEL_ID_MAP.get(model, model)
        messages = [
            {"role": "system", "content": persona["system_prompt"]},
            {"role": "user", "content": prompt},
        ]
        response = await manager.chat(model=resolved_model, messages=messages)
        content = response.get("message", {}).get("content", "")
        return {
            "backend": manager.active.name if manager.active else None,
            "model_id": resolved_model,
            "response": content,
        }


def run(args: argparse.Namespace) -> int:
    configure_logging(args.quiet)
    prompt = args.prompt
    if not prompt:
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("No prompt provided")
        return 1

    started_at = now_iso()
    start = time.monotonic()
    payload = asyncio.run(_chat(prompt, args.model, args.model_id))
    duration = time.monotonic() - start

    output_text_path = None
    if args.output_text:
        output_text_path = Path(args.output_text).expanduser().resolve()
        output_text_path.parent.mkdir(parents=True, exist_ok=True)
        output_text_path.write_text(payload.get("response", ""), encoding="utf-8")

    result = AgentResult(
        name=AGENT_NAME,
        status="ok" if payload.get("response") else "error",
        started_at=started_at,
        finished_at=now_iso(),
        duration_seconds=duration,
        metrics={},
        notes=[payload["error"]] if payload.get("error") else [],
        payload={
            "model": args.model,
            "model_id": payload.get("model_id"),
            "backend": payload.get("backend"),
            "prompt": prompt,
            "response": payload.get("response"),
            "output_text": str(output_text_path) if output_text_path else None,
        },
    )

    emit_result(
        result,
        output_path=Path(args.output) if args.output else None,
        force_stdout=args.stdout,
        pretty=args.pretty,
    )
    return 0 if result.status == "ok" else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
