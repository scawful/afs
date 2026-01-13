"""Built-in AFS background agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .context_audit import (
    AGENT_DESCRIPTION as CONTEXT_AUDIT_DESCRIPTION,
    AGENT_NAME as CONTEXT_AUDIT_NAME,
    main as context_audit_main,
)
from .context_warm import (
    AGENT_DESCRIPTION as CONTEXT_WARM_DESCRIPTION,
    AGENT_NAME as CONTEXT_WARM_NAME,
    main as context_warm_main,
)
from .context_inventory import (
    AGENT_DESCRIPTION as CONTEXT_INVENTORY_DESCRIPTION,
    AGENT_NAME as CONTEXT_INVENTORY_NAME,
    main as context_inventory_main,
)
from .memory_export import (
    AGENT_DESCRIPTION as MEMORY_EXPORT_DESCRIPTION,
    AGENT_NAME as MEMORY_EXPORT_NAME,
    main as memory_export_main,
)
from .scribe_draft import (
    AGENT_DESCRIPTION as SCRIBE_DRAFT_DESCRIPTION,
    AGENT_NAME as SCRIBE_DRAFT_NAME,
    main as scribe_draft_main,
)
from .claude_orchestrator import (
    AGENT_DESCRIPTION as CLAUDE_ORCHESTRATOR_DESCRIPTION,
    AGENT_NAME as CLAUDE_ORCHESTRATOR_NAME,
    main as claude_orchestrator_main,
)


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str
    entrypoint: Callable[[Sequence[str] | None], int]


AGENTS: dict[str, AgentSpec] = {
    CONTEXT_AUDIT_NAME: AgentSpec(
        name=CONTEXT_AUDIT_NAME,
        description=CONTEXT_AUDIT_DESCRIPTION,
        entrypoint=context_audit_main,
    ),
    CONTEXT_INVENTORY_NAME: AgentSpec(
        name=CONTEXT_INVENTORY_NAME,
        description=CONTEXT_INVENTORY_DESCRIPTION,
        entrypoint=context_inventory_main,
    ),
    MEMORY_EXPORT_NAME: AgentSpec(
        name=MEMORY_EXPORT_NAME,
        description=MEMORY_EXPORT_DESCRIPTION,
        entrypoint=memory_export_main,
    ),
    SCRIBE_DRAFT_NAME: AgentSpec(
        name=SCRIBE_DRAFT_NAME,
        description=SCRIBE_DRAFT_DESCRIPTION,
        entrypoint=scribe_draft_main,
    ),
    CONTEXT_WARM_NAME: AgentSpec(
        name=CONTEXT_WARM_NAME,
        description=CONTEXT_WARM_DESCRIPTION,
        entrypoint=context_warm_main,
    ),
    CLAUDE_ORCHESTRATOR_NAME: AgentSpec(
        name=CLAUDE_ORCHESTRATOR_NAME,
        description=CLAUDE_ORCHESTRATOR_DESCRIPTION,
        entrypoint=claude_orchestrator_main,
    ),
}


def list_agents() -> list[AgentSpec]:
    return sorted(AGENTS.values(), key=lambda spec: spec.name)


def get_agent(name: str) -> AgentSpec | None:
    return AGENTS.get(name)


__all__ = ["AgentSpec", "AGENTS", "list_agents", "get_agent"]
