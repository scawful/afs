"""Built-in AFS agents plus extension-provided agent registrations."""

from __future__ import annotations

import importlib
import logging
import sys
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from ..plugins import load_enabled_extensions

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentCapability:
    tools: list[str]
    topics: list[str]
    mount_types: list[str]
    description: str = ""


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str
    entrypoint: Callable[[Sequence[str] | None], int]
    capabilities: AgentCapability | None = None


CORE_AGENT_MODULES = (
    "afs.agents.claude_orchestrator",
    "afs.agents.context_audit",
    "afs.agents.context_inventory",
    "afs.agents.context_warm",
    "afs.agents.dashboard_export",
    "afs.agents.gemini_workspace_brief",
    "afs.agents.history_memory",
    "afs.agents.journal_agent",
    "afs.agents.mission_runner",
    "afs.agents.scribe_draft",
    "afs.agents.supervisor",
    "afs.agents.tether_bridge",
    "afs.agents.workspace_analyst",
)


@contextmanager
def _extension_import_path(extension_root: Path):
    candidates = [str(extension_root), str(extension_root.parent)]
    original = list(sys.path)
    sys.path = [entry for entry in candidates if Path(entry).exists()] + original
    try:
        yield
    finally:
        sys.path = original


def _load_agent_module(module_name: str) -> AgentSpec | None:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        logger.warning("Failed to import agent module %s: %s", module_name, exc)
        return None

    name = getattr(module, "AGENT_NAME", "")
    description = getattr(module, "AGENT_DESCRIPTION", "")
    entrypoint = getattr(module, "main", None)
    if not isinstance(name, str) or not name.strip() or not callable(entrypoint):
        logger.warning("Agent module %s is missing AGENT_NAME or main()", module_name)
        return None

    capabilities_raw = getattr(module, "AGENT_CAPABILITIES", None)
    capabilities = None
    if isinstance(capabilities_raw, dict):
        capabilities = AgentCapability(
            tools=list(capabilities_raw.get("tools", [])),
            topics=list(capabilities_raw.get("topics", [])),
            mount_types=list(capabilities_raw.get("mount_types", [])),
            description=str(capabilities_raw.get("description", "")),
        )
    elif isinstance(capabilities_raw, AgentCapability):
        capabilities = capabilities_raw

    return AgentSpec(
        name=name.strip(),
        description=str(description or "").strip(),
        entrypoint=entrypoint,
        capabilities=capabilities,
    )


def _normalize_agent_specs(payload: object) -> list[AgentSpec]:
    if payload is None:
        return []
    if isinstance(payload, AgentSpec):
        return [payload]
    if isinstance(payload, dict):
        payloads = [payload]
    elif isinstance(payload, (list, tuple)):
        payloads = list(payload)
    else:
        raise TypeError("register_agents() must return AgentSpec, dict, or list")

    specs: list[AgentSpec] = []
    for item in payloads:
        if isinstance(item, AgentSpec):
            specs.append(item)
            continue
        if not isinstance(item, dict):
            raise TypeError("agent payload must be dict or AgentSpec")
        name = item.get("name")
        entrypoint = item.get("entrypoint")
        description = item.get("description", "")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("agent payload missing name")
        if not callable(entrypoint):
            raise ValueError(f"agent '{name}' missing callable entrypoint")
        specs.append(
            AgentSpec(
                name=name.strip(),
                description=str(description or "").strip(),
                entrypoint=entrypoint,
            )
        )
    return specs


def _load_core_agents() -> dict[str, AgentSpec]:
    loaded: dict[str, AgentSpec] = {}
    for module_name in CORE_AGENT_MODULES:
        spec = _load_agent_module(module_name)
        if spec is not None:
            loaded[spec.name] = spec
    return loaded


def _load_extension_agents(config: object = None) -> dict[str, AgentSpec]:
    loaded: dict[str, AgentSpec] = {}
    for extension in load_enabled_extensions(config=config).values():
        for module_name in extension.agent_modules:
            try:
                with _extension_import_path(extension.root):
                    module = importlib.import_module(module_name)
            except Exception as exc:
                logger.warning("Failed to import agent module %s: %s", module_name, exc)
                continue
            register = getattr(module, "register_agents", None)
            if not callable(register):
                continue
            try:
                specs = _normalize_agent_specs(register())
            except Exception as exc:
                logger.warning("Failed to register agents from %s: %s", module_name, exc)
                continue
            for spec in specs:
                loaded[spec.name] = spec
    return loaded


def get_agent_registry(config: object = None) -> dict[str, AgentSpec]:
    registry = _load_core_agents()
    registry.update(_load_extension_agents(config=config))
    return registry


def list_agents(config: object = None) -> list[AgentSpec]:
    return sorted(get_agent_registry(config=config).values(), key=lambda spec: spec.name)


def get_agent(name: str, config: object = None) -> AgentSpec | None:
    return get_agent_registry(config=config).get(name)


__all__ = ["AgentCapability", "AgentSpec", "CORE_AGENT_MODULES", "get_agent_registry", "list_agents", "get_agent"]
