"""MCP tool, resource, and prompt registry with deferred loading support."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..response_schemas import SCHEMA_URI_PREFIX

ToolHandler = Callable[[dict[str, Any], Any], dict[str, Any]]
ResourceHandler = Callable[..., dict[str, Any]]
PromptHandler = Callable[..., list[dict[str, Any]]]

# Core names that extensions cannot override.
CORE_RESOURCE_URIS = {"afs://contexts", "afs://claude/bootstrap"}
CORE_RESOURCE_PREFIXES = ("afs://context/", SCHEMA_URI_PREFIX)
CORE_PROMPT_NAMES = {
    "afs.context.overview",
    "afs.query.search",
    "afs.session.bootstrap",
    "afs.session.pack",
    "afs.scratchpad.review",
    "afs.workflow.structured",
}

MCP_TOOL_CATALOG_VALUES = frozenset({"", "slim", "full"})


@dataclass(frozen=True)
class MCPToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    source: str = "core"
    deferred: bool = False
    concurrent_safe: bool = False
    pre_hook: Callable[[dict[str, Any], Any], dict[str, Any]] | None = None
    post_hook: Callable[[dict[str, Any], dict[str, Any], Any], dict[str, Any]] | None = None
    # Empty means "inherit" while an extension contribution is normalized;
    # outside an extension-level default it remains full-catalog-only.
    catalog: str = ""

    def __post_init__(self) -> None:
        if self.catalog not in MCP_TOOL_CATALOG_VALUES:
            raise ValueError(
                "MCP tool catalog must be one of: '', 'full', 'slim'"
            )

    def to_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass(frozen=True)
class MCPResourceDefinition:
    uri: str
    name: str
    description: str
    mime_type: str
    handler: ResourceHandler
    source: str = "core"

    def to_spec(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }


@dataclass(frozen=True)
class MCPPromptDefinition:
    name: str
    description: str
    arguments: list[dict[str, Any]]
    handler: PromptHandler
    source: str = "core"

    def to_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": list(self.arguments),
        }


@dataclass(frozen=True)
class MCPExtensionContribution:
    tools: list[MCPToolDefinition] = field(default_factory=list)
    resources: list[MCPResourceDefinition] = field(default_factory=list)
    prompts: list[MCPPromptDefinition] = field(default_factory=list)


@dataclass
class ExtensionMCPStatus:
    extension: str
    surface: str
    module: str
    factory: str
    loaded_tools: list[str] = field(default_factory=list)
    loaded_resources: list[str] = field(default_factory=list)
    loaded_prompts: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "extension": self.extension,
            "surface": self.surface,
            "module": self.module,
            "factory": self.factory,
            "loaded_tools": list(self.loaded_tools),
            "loaded_resources": list(self.loaded_resources),
            "loaded_prompts": list(self.loaded_prompts),
            "error": self.error,
        }


@dataclass
class MCPToolRegistry:
    tools: dict[str, MCPToolDefinition] = field(default_factory=dict)
    resources: dict[str, MCPResourceDefinition] = field(default_factory=dict)
    prompts: dict[str, MCPPromptDefinition] = field(default_factory=dict)
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

    def add_resource(self, resource: MCPResourceDefinition) -> None:
        if resource.uri in CORE_RESOURCE_URIS or any(
            resource.uri.startswith(prefix) for prefix in CORE_RESOURCE_PREFIXES
        ):
            raise ValueError(
                f"Resource '{resource.uri}' already registered by core; "
                f"cannot override from {resource.source}"
            )
        if resource.uri in self.resources:
            existing = self.resources[resource.uri]
            raise ValueError(
                f"Resource '{resource.uri}' already registered by {existing.source}; "
                f"cannot override from {resource.source}"
            )
        self.resources[resource.uri] = resource

    def add_prompt(self, prompt: MCPPromptDefinition) -> None:
        if prompt.name in CORE_PROMPT_NAMES:
            raise ValueError(
                f"Prompt '{prompt.name}' already registered by core; "
                f"cannot override from {prompt.source}"
            )
        if prompt.name in self.prompts:
            existing = self.prompts[prompt.name]
            raise ValueError(
                f"Prompt '{prompt.name}' already registered by {existing.source}; "
                f"cannot override from {prompt.source}"
            )
        self.prompts[prompt.name] = prompt

    def specs(self) -> list[dict[str, Any]]:
        """Return sorted tool specs.  Built-in (core) tools come first as a
        contiguous prefix for prompt cache stability, followed by extension
        tools sorted by name."""
        core = []
        ext = []
        for name in sorted(self.tools):
            tool = self.tools[name]
            if tool.source == "core":
                core.append(tool.to_spec())
            else:
                ext.append(tool.to_spec())
        return core + ext

    def resource_specs(self) -> list[dict[str, Any]]:
        return [self.resources[uri].to_spec() for uri in sorted(self.resources)]

    def prompt_specs(self) -> list[dict[str, Any]]:
        return [self.prompts[name].to_spec() for name in sorted(self.prompts)]

    def call(
        self,
        name: str,
        arguments: dict[str, Any],
        manager: Any,
    ) -> dict[str, Any]:
        import time

        from ..agent_scope import assert_tool_allowed

        tool = self.tools.get(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name}")
        assert_tool_allowed(name)

        # Run pre-hook if defined (can modify arguments or block)
        if tool.pre_hook is not None:
            arguments = tool.pre_hook(arguments, manager)

        start = time.monotonic()
        try:
            result = tool.handler(arguments, manager)
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            try:
                from ..history import log_mcp_tool_call

                log_mcp_tool_call(
                    name,
                    arguments,
                    {"error": str(exc)},
                    duration_ms=elapsed_ms,
                    context_root=manager.config.general.context_root,
                )
            except Exception:
                pass
            raise
        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Run post-hook if defined (can transform result)
        if tool.post_hook is not None:
            result = tool.post_hook(arguments, result, manager)

        try:
            from ..history import log_mcp_tool_call
            log_mcp_tool_call(
                name,
                arguments,
                result,
                duration_ms=elapsed_ms,
                context_root=manager.config.general.context_root,
            )
        except Exception:
            pass
        return result

    def read_resource(self, uri: str, manager: Any) -> dict[str, Any]:
        resource = self.resources.get(uri)
        if not resource:
            raise ValueError(f"Unknown resource URI: {uri}")
        return resource.handler(uri, manager)

    def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any],
        manager: Any,
    ) -> list[dict[str, Any]]:
        prompt = self.prompts.get(name)
        if not prompt:
            raise ValueError(f"Unknown prompt: {name}")
        return prompt.handler(arguments, manager)
