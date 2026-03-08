"""Lightweight MCP server exposing AFS context operations over stdio."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import load_config_model
from .discovery import discover_contexts
from .manager import AFSManager
from .models import MountType
from .plugins import load_enabled_extensions

SERVER_NAME = "afs"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2024-11-05"

ToolHandler = Callable[[dict[str, Any], AFSManager], dict[str, Any]]


@dataclass(frozen=True)
class MCPToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    source: str = "core"

    def to_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass
class ExtensionMCPStatus:
    extension: str
    module: str
    factory: str
    loaded_tools: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "extension": self.extension,
            "module": self.module,
            "factory": self.factory,
            "loaded_tools": list(self.loaded_tools),
            "error": self.error,
        }


@dataclass
class MCPToolRegistry:
    tools: dict[str, MCPToolDefinition] = field(default_factory=dict)
    extension_status: list[ExtensionMCPStatus] = field(default_factory=list)
    load_errors: dict[str, str] = field(default_factory=dict)

    def add_tool(self, tool: MCPToolDefinition) -> None:
        if tool.name in self.tools:
            existing = self.tools[tool.name]
            raise ValueError(
                f"Tool '{tool.name}' already registered by {existing.source}; "
                f"cannot override from {tool.source}"
            )
        self.tools[tool.name] = tool

    def specs(self) -> list[dict[str, Any]]:
        return [self.tools[name].to_spec() for name in sorted(self.tools)]

    def call(
        self,
        name: str,
        arguments: dict[str, Any],
        manager: AFSManager,
    ) -> dict[str, Any]:
        tool = self.tools.get(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name}")
        return tool.handler(arguments, manager)


def _read_message(stream) -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = stream.readline()
        if line == b"":
            return None
        if line in (b"\r\n", b"\n"):
            break
        if b":" not in line:
            continue
        key, value = line.decode("utf-8", errors="replace").split(":", 1)
        headers[key.strip().lower()] = value.strip()

    length_raw = headers.get("content-length")
    if not length_raw:
        return None
    try:
        length = int(length_raw)
    except ValueError:
        return None
    body = stream.read(length)
    if not body:
        return None
    return json.loads(body.decode("utf-8"))


def _write_message(stream, payload: dict[str, Any]) -> None:
    raw = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    header = f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii")
    stream.write(header)
    stream.write(raw)
    stream.flush()


def _error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _success_response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _allowed_roots(manager: AFSManager) -> list[Path]:
    roots: list[Path] = []

    home_context = (Path.home() / ".context").resolve()
    roots.append(home_context)

    config_root = manager.config.general.context_root.resolve()
    if config_root not in roots:
        roots.append(config_root)

    local_context = (Path.cwd() / ".context").resolve()
    if local_context.exists() and local_context not in roots:
        roots.append(local_context)

    return roots


def _assert_allowed(path: Path, manager: AFSManager) -> Path:
    resolved = path.expanduser().resolve()
    for root in _allowed_roots(manager):
        if resolved == root or resolved.is_relative_to(root):
            return resolved
    raise PermissionError(f"Path outside allowed roots: {resolved}")


def _resolve_context_path(arguments: dict[str, Any], manager: AFSManager) -> Path:
    raw = arguments.get("context_path")
    if isinstance(raw, str) and raw.strip():
        return _assert_allowed(Path(raw), manager)
    default = Path.cwd() / ".context"
    return _assert_allowed(default, manager)


def _as_text_result(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=True)}],
        "structuredContent": payload,
    }


def _tool_fs_read(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    path_value = arguments.get("path")
    if not isinstance(path_value, str):
        raise ValueError("path must be a string")
    path = _assert_allowed(Path(path_value), manager)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Path is a directory: {path}")
    return {
        "path": str(path),
        "content": path.read_text(encoding="utf-8", errors="replace"),
    }


def _tool_fs_write(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    path_value = arguments.get("path")
    content = arguments.get("content")
    append = bool(arguments.get("append", False))
    mkdirs = bool(arguments.get("mkdirs", False))
    if not isinstance(path_value, str) or not isinstance(content, str):
        raise ValueError("path and content must be strings")

    path = _assert_allowed(Path(path_value), manager)
    if not path.parent.exists():
        if not mkdirs:
            raise FileNotFoundError(f"Parent directory missing: {path.parent}")
        path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(content)
    return {"path": str(path), "bytes": len(content.encode("utf-8")), "append": append}


def _tool_fs_list(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    path_value = arguments.get("path")
    max_depth = arguments.get("max_depth", 1)
    if not isinstance(path_value, str):
        raise ValueError("path must be a string")
    if not isinstance(max_depth, int):
        max_depth = 1

    root = _assert_allowed(Path(path_value), manager)
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")

    entries: list[dict[str, Any]] = []
    if root.is_file():
        entries.append({"path": str(root), "is_dir": False})
    else:
        for candidate in root.rglob("*"):
            try:
                depth = len(candidate.relative_to(root).parts)
            except Exception:
                continue
            if max_depth >= 0 and depth > max_depth:
                continue
            entries.append({"path": str(candidate), "is_dir": candidate.is_dir()})
    return {"path": str(root), "entries": entries}


def _tool_context_discover(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    search_paths_value = arguments.get("search_paths", [])
    max_depth = arguments.get("max_depth", 3)

    search_paths: list[Path] | None = None
    if isinstance(search_paths_value, list):
        values: list[Path] = []
        for item in search_paths_value:
            if isinstance(item, str):
                values.append(Path(item).expanduser())
        if values:
            search_paths = values

    if not isinstance(max_depth, int):
        max_depth = 3

    contexts = discover_contexts(search_paths=search_paths, max_depth=max_depth, config=manager.config)
    return {
        "contexts": [
            {
                "project": context.project_name,
                "path": str(context.path),
                "valid": context.is_valid,
                "mounts": context.total_mounts,
            }
            for context in contexts
        ]
    }


def _tool_context_mount(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    context_path = _resolve_context_path(arguments, manager)
    source_value = arguments.get("source")
    mount_type_value = arguments.get("mount_type")
    alias_value = arguments.get("alias")

    if not isinstance(source_value, str):
        raise ValueError("source must be a string")
    if not isinstance(mount_type_value, str):
        raise ValueError("mount_type must be a string")

    source = Path(source_value).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    mount_type = MountType(mount_type_value)
    alias = alias_value if isinstance(alias_value, str) else None
    mount = manager.mount(source, mount_type, alias=alias, context_path=context_path)
    return {
        "context_path": str(context_path),
        "mount": {
            "name": mount.name,
            "mount_type": mount.mount_type.value,
            "source": str(mount.source),
            "is_symlink": mount.is_symlink,
        },
    }


def _builtin_tool_definitions() -> list[MCPToolDefinition]:
    return [
        MCPToolDefinition(
            name="fs.read",
            description="Read UTF-8 text from a context-scoped file.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path."},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=_tool_fs_read,
        ),
        MCPToolDefinition(
            name="fs.write",
            description="Write UTF-8 text to a context-scoped file.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "append": {"type": "boolean", "default": False},
                    "mkdirs": {"type": "boolean", "default": False},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            handler=_tool_fs_write,
        ),
        MCPToolDefinition(
            name="fs.list",
            description="List files under a context-scoped path.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_depth": {"type": "integer", "default": 1},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=_tool_fs_list,
        ),
        MCPToolDefinition(
            name="context.discover",
            description="Discover project .context roots.",
            input_schema={
                "type": "object",
                "properties": {
                    "search_paths": {"type": "array", "items": {"type": "string"}},
                    "max_depth": {"type": "integer", "default": 3},
                },
                "additionalProperties": False,
            },
            handler=_tool_context_discover,
        ),
        MCPToolDefinition(
            name="context.mount",
            description="Mount a source path into a context mount type.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "source": {"type": "string"},
                    "mount_type": {"type": "string", "enum": [mount.value for mount in MountType]},
                    "alias": {"type": "string"},
                },
                "required": ["source", "mount_type"],
                "additionalProperties": False,
            },
            handler=_tool_context_mount,
        ),
    ]


@contextmanager
def _prepend_paths(paths: Iterable[Path]):
    values = [str(path) for path in paths if path and path.exists()]
    if not values:
        yield
        return
    original = list(sys.path)
    sys.path = values + original
    try:
        yield
    finally:
        sys.path = original


def _invoke_factory(factory: Callable[..., Any], manager: AFSManager) -> Any:
    signature = inspect.signature(factory)
    params = list(signature.parameters.values())
    if not params:
        return factory()
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params):
        return factory(manager=manager)
    first = params[0]
    if first.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        return factory(manager)
    if first.kind == inspect.Parameter.KEYWORD_ONLY and first.name == "manager":
        return factory(manager=manager)
    return factory()


def _invoke_tool_handler(
    handler: Callable[..., dict[str, Any]],
    arguments: dict[str, Any],
    manager: AFSManager,
) -> dict[str, Any]:
    signature = inspect.signature(handler)
    params = list(signature.parameters.values())
    if not params:
        return handler()
    if len(params) == 1:
        return handler(arguments)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params):
        return handler(arguments=arguments, manager=manager)
    return handler(arguments, manager)


def _normalize_extension_tools(
    extension_name: str,
    definitions: Any,
) -> list[MCPToolDefinition]:
    if definitions is None:
        return []
    if isinstance(definitions, dict):
        payloads = [definitions]
    elif isinstance(definitions, (list, tuple)):
        payloads = list(definitions)
    else:
        raise TypeError(
            f"Extension {extension_name} must return list[dict] from mcp tool factory"
        )

    tools: list[MCPToolDefinition] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            raise TypeError(
                f"Extension {extension_name} returned non-dict MCP tool payload"
            )
        name = payload.get("name")
        description = payload.get("description")
        input_schema = payload.get("inputSchema", payload.get("input_schema"))
        handler = payload.get("handler")

        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Extension {extension_name} returned tool without valid name")
        if not isinstance(description, str):
            description = ""
        if not isinstance(input_schema, dict):
            input_schema = {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            }
        if not callable(handler):
            raise ValueError(
                f"Extension {extension_name} tool '{name}' missing callable handler"
            )

        def _wrapped(
            arguments: dict[str, Any],
            manager: AFSManager,
            _handler=handler,
        ) -> dict[str, Any]:
            return _invoke_tool_handler(_handler, arguments, manager)

        tools.append(
            MCPToolDefinition(
                name=name.strip(),
                description=description.strip(),
                input_schema=input_schema,
                handler=_wrapped,
                source=f"extension:{extension_name}",
            )
        )
    return tools


def _load_extension_tool_definitions(
    manager: AFSManager,
) -> tuple[list[MCPToolDefinition], list[ExtensionMCPStatus]]:
    loaded_tools: list[MCPToolDefinition] = []
    statuses: list[ExtensionMCPStatus] = []

    extensions = load_enabled_extensions(config=manager.config)
    for extension_name, manifest in sorted(extensions.items()):
        module_name = manifest.mcp_tools_module.strip()
        factory_name = manifest.mcp_tools_factory.strip() or "register_mcp_tools"
        if not module_name:
            continue

        status = ExtensionMCPStatus(
            extension=extension_name,
            module=module_name,
            factory=factory_name,
        )
        statuses.append(status)

        with _prepend_paths([manifest.root, manifest.root.parent]):
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:
                status.error = f"import failed: {exc}"
                continue

            factory = getattr(module, factory_name, None)
            if not callable(factory):
                status.error = f"factory not callable: {factory_name}"
                continue

            try:
                definitions = _invoke_factory(factory, manager)
                extension_tools = _normalize_extension_tools(extension_name, definitions)
            except Exception as exc:
                status.error = str(exc)
                continue

            for tool in extension_tools:
                loaded_tools.append(tool)
                status.loaded_tools.append(tool.name)

    return loaded_tools, statuses


def build_mcp_registry(manager: AFSManager) -> MCPToolRegistry:
    """Build MCP tool registry including extension tools."""
    registry = MCPToolRegistry()

    for tool in _builtin_tool_definitions():
        registry.add_tool(tool)

    extension_tools, statuses = _load_extension_tool_definitions(manager)
    registry.extension_status = statuses
    for status in statuses:
        if status.error:
            registry.load_errors[status.extension] = status.error

    for tool in extension_tools:
        try:
            registry.add_tool(tool)
        except ValueError as exc:
            registry.load_errors[tool.source] = str(exc)

    return registry


def _tool_specs(registry: MCPToolRegistry | None = None) -> list[dict[str, Any]]:
    if registry is None:
        return [tool.to_spec() for tool in _builtin_tool_definitions()]
    return registry.specs()


def get_mcp_status(config_path: Path | None = None) -> dict[str, Any]:
    """Return MCP registry and extension status for diagnostics."""
    config = load_config_model(config_path=config_path, merge_user=True)
    manager = AFSManager(config=config)
    registry = build_mcp_registry(manager)
    return {
        "tools": sorted(registry.tools.keys()),
        "extension_status": [status.to_dict() for status in registry.extension_status],
        "load_errors": dict(registry.load_errors),
    }


def _handle_request(
    request: dict[str, Any],
    manager: AFSManager,
    registry: MCPToolRegistry | None = None,
) -> dict[str, Any] | None:
    method = request.get("method")
    request_id = request.get("id")
    active_registry = registry or build_mcp_registry(manager)

    if method == "initialize":
        return _success_response(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        )

    if method == "notifications/initialized":
        return None

    if method == "ping":
        return _success_response(request_id, {})

    if method == "tools/list":
        return _success_response(request_id, {"tools": _tool_specs(active_registry)})

    if method == "tools/call":
        params = request.get("params", {})
        if not isinstance(params, dict):
            return _error_response(request_id, -32602, "Invalid params")

        name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(name, str):
            return _error_response(request_id, -32602, "Missing tool name")
        if not isinstance(arguments, dict):
            return _error_response(request_id, -32602, "arguments must be object")

        try:
            payload = active_registry.call(name, arguments, manager)
        except Exception as exc:
            return _error_response(request_id, -32000, str(exc))

        return _success_response(request_id, _as_text_result(payload))

    if request_id is not None:
        return _error_response(request_id, -32601, f"Method not found: {method}")
    return None


def serve(config_path: Path | None = None) -> int:
    config = load_config_model(config_path=config_path, merge_user=True)
    manager = AFSManager(config=config)
    registry = build_mcp_registry(manager)

    while True:
        message = _read_message(sys.stdin.buffer)
        if message is None:
            break
        response = _handle_request(message, manager, registry=registry)
        if response is not None:
            _write_message(sys.stdout.buffer, response)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AFS MCP server")
    parser.add_argument("--config", help="Config path override.")
    args = parser.parse_args(argv)
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    return serve(config_path=config_path)


if __name__ == "__main__":
    raise SystemExit(main())
