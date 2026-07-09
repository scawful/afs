"""MCP JSON-RPC message transport over stdio (JSONL and content-length)."""

from __future__ import annotations

import json
from typing import Any

from afs.version import __version__

SERVER_NAME = "afs"
SERVER_VERSION = __version__
PROTOCOL_VERSION = "2025-06-18"
LEGACY_PROTOCOL_VERSION = "2024-11-05"
SUPPORTED_PROTOCOL_VERSIONS = (PROTOCOL_VERSION, LEGACY_PROTOCOL_VERSION)


def read_message(stream) -> tuple[dict[str, Any] | None, str | None]:
    """Read a JSON-RPC message from stream in JSONL or content-length mode."""
    first = stream.read(1)
    while first in (b"\r", b"\n"):
        first = stream.read(1)
    if first == b"":
        return None, None

    if first in (b"{", b"["):
        line = first + stream.readline()
        return json.loads(line.decode("utf-8")), "jsonl"

    headers: dict[str, str] = {}
    header_bytes = bytearray(first)
    while True:
        chunk = stream.read(1)
        if chunk == b"":
            return None, None
        header_bytes.extend(chunk)
        if header_bytes.endswith(b"\r\n\r\n") or header_bytes.endswith(b"\n\n") or header_bytes.endswith(b"\r\r"):
            break

    header_text = (
        header_bytes.decode("utf-8", errors="replace")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )
    for line in header_text.split("\n"):
        if not line.strip() or ":" not in line:
            continue
        key, value = line.split(":", 1)
        headers[key.strip().lower()] = value.strip()

    length_raw = headers.get("content-length")
    if not length_raw:
        return None, None
    try:
        length = int(length_raw)
    except ValueError:
        return None, None
    body = bytearray()
    while len(body) < length:
        chunk = stream.read(length - len(body))
        if chunk == b"":
            return None, None
        body.extend(chunk)
    return json.loads(body.decode("utf-8")), "content-length"


def write_message(stream, payload: dict[str, Any], mode: str = "content-length") -> None:
    """Write a JSON-RPC message to stream in specified mode."""
    raw = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    if mode == "jsonl":
        stream.write(raw + b"\n")
    else:
        header = f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii")
        stream.write(header)
        stream.write(raw)
    stream.flush()


def error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    """Construct a JSON-RPC 2.0 error response."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def success_response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    """Construct a JSON-RPC 2.0 success response."""
    return {"jsonrpc": "2.0", "id": request_id, "result": result}
