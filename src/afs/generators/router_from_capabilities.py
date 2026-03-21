"""Generate router training data from the agent capability registry.

Instead of hand-crafting expert_router_v1.jsonl, this module derives
routing examples directly from declared AgentCapability metadata.
Each agent's tools, topics, and mount_types inform the generation of
realistic user prompts that should route to that agent.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..agents import AgentSpec, get_agent_registry

logger = logging.getLogger(__name__)

# Maps capability keywords to realistic user prompt templates.
# Each template will be formatted with agent-specific details.
_TOOL_PROMPT_TEMPLATES = {
    "context.index.rebuild": [
        "Rebuild the context index for the current workspace",
        "The index seems stale, refresh it",
        "Re-index all mounted context directories",
    ],
    "context.query": [
        "Search the context for information about {topic}",
        "Find files related to {topic} in the knowledge base",
        "Query the context index for {topic}",
    ],
    "memory.search": [
        "Search memory for past decisions about {topic}",
        "What do we know from previous sessions about {topic}",
        "Look up memory entries related to {topic}",
    ],
    "memory.status": [
        "Check the current memory consolidation status",
        "How many memory entries do we have?",
        "Is the memory pipeline up to date?",
    ],
    "context.freshness": [
        "Check which context mounts are stale",
        "Show freshness scores for all mounts",
        "Are there any outdated files in the index?",
    ],
    "hivemind.cleanup": [
        "Clean up old hivemind messages",
        "Remove stale agent messages from the bus",
        "Run the hivemind reaper",
    ],
    "session.replay": [
        "Show me the timeline from the last session",
        "Replay recent session events",
        "What happened in yesterday's session?",
    ],
}

_TOPIC_PROMPT_TEMPLATES = {
    "context:repair": [
        "The context seems broken, can you fix it?",
        "Repair the context directory structure",
        "Some mount paths are missing, fix them",
    ],
    "context:refresh": [
        "Refresh the workspace context",
        "Update all context mounts with latest files",
        "Sync the context with the current workspace state",
    ],
    "agent:lifecycle": [
        "Check on the background agents",
        "What agents have run recently?",
        "Consolidate recent agent activity into memory",
    ],
}

_MOUNT_PROMPT_TEMPLATES = {
    "knowledge": [
        "Find documentation about {topic}",
        "What knowledge files do we have about {topic}?",
        "Look up the reference material for {topic}",
    ],
    "tools": [
        "List available tools in the context",
        "What tool configurations are mounted?",
    ],
    "scratchpad": [
        "Check the scratchpad for recent notes",
        "Save this to the scratchpad: {topic}",
        "Read the current scratchpad state",
    ],
    "history": [
        "Show recent history events",
        "What operations were performed today?",
        "Review the event log for errors",
    ],
    "memory": [
        "What has been consolidated into long-term memory?",
        "Search memory for {topic}",
        "Check memory status",
    ],
}

# Sample topics for template filling
_SAMPLE_TOPICS = [
    "authentication", "database schema", "API endpoints",
    "deployment pipeline", "test coverage", "error handling",
    "configuration", "agent coordination", "index structure",
    "file organization", "mount types", "context paths",
    "build system", "CI/CD", "monitoring",
]

ROUTER_SYSTEM_PROMPT = (
    "Classify the prompt to the appropriate expert. "
    "Respond with exactly one word: the name of the agent best suited to handle this request."
)


@dataclass
class RouterDatasetConfig:
    """Configuration for router dataset generation."""

    system_prompt: str = ROUTER_SYSTEM_PROMPT
    samples_per_agent: int = 10
    include_agents_without_capabilities: bool = False


@dataclass
class RouterGenerationResult:
    """Result of router dataset generation."""

    agents_processed: int = 0
    agents_with_capabilities: int = 0
    samples_generated: int = 0
    output_path: Path | None = None
    agent_sample_counts: dict[str, int] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def generate_router_dataset(
    output_path: Path | None = None,
    *,
    config: RouterDatasetConfig | None = None,
    afs_config: Any = None,
) -> RouterGenerationResult:
    """Generate router training data from the agent capability registry.

    For each agent with declared capabilities, generates realistic user
    prompts that should route to that agent, based on its declared tools,
    topics, and mount_types.
    """
    cfg = config or RouterDatasetConfig()
    result = RouterGenerationResult(output_path=output_path)

    registry = get_agent_registry(config=afs_config)
    result.agents_processed = len(registry)

    samples: list[dict[str, Any]] = []

    for agent_name, spec in sorted(registry.items()):
        if spec.capabilities is None:
            if not cfg.include_agents_without_capabilities:
                continue
            # Generate generic samples from description
            agent_samples = _generate_from_description(agent_name, spec, cfg)
        else:
            result.agents_with_capabilities += 1
            agent_samples = _generate_from_capabilities(agent_name, spec, cfg)

        samples.extend(agent_samples)
        result.agent_sample_counts[agent_name] = len(agent_samples)

    result.samples_generated = len(samples)

    if output_path and samples:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        logger.info(
            "Wrote %d router training samples to %s",
            len(samples),
            output_path,
        )

    return result


def _generate_from_capabilities(
    agent_name: str,
    spec: AgentSpec,
    config: RouterDatasetConfig,
) -> list[dict[str, Any]]:
    """Generate routing examples from declared capabilities."""
    caps = spec.capabilities
    if caps is None:
        return []

    prompts: list[str] = []

    # Generate prompts from tool capabilities
    for tool in caps.tools:
        templates = _TOOL_PROMPT_TEMPLATES.get(tool, [])
        for template in templates:
            if "{topic}" in template:
                for topic in _SAMPLE_TOPICS[:3]:
                    prompts.append(template.format(topic=topic))
            else:
                prompts.append(template)

    # Generate prompts from topic subscriptions
    for topic in caps.topics:
        templates = _TOPIC_PROMPT_TEMPLATES.get(topic, [])
        prompts.extend(templates)

    # Generate prompts from mount type access
    for mount_type in caps.mount_types:
        templates = _MOUNT_PROMPT_TEMPLATES.get(mount_type, [])
        for template in templates:
            if "{topic}" in template:
                for topic in _SAMPLE_TOPICS[:2]:
                    prompts.append(template.format(topic=topic))
            else:
                prompts.append(template)

    # Add description-based prompt if available
    if caps.description:
        prompts.append(f"I need help with: {caps.description}")

    # Limit to configured samples_per_agent
    prompts = _dedupe_prompts(prompts)[: config.samples_per_agent]

    return [
        {
            "messages": [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": agent_name},
            ],
            "metadata": {
                "source": "capability_registry",
                "agent": agent_name,
                "has_capabilities": True,
            },
        }
        for prompt in prompts
    ]


def _generate_from_description(
    agent_name: str,
    spec: AgentSpec,
    config: RouterDatasetConfig,
) -> list[dict[str, Any]]:
    """Generate basic routing examples from agent description."""
    if not spec.description:
        return []

    prompts = [
        f"I need help with: {spec.description}",
        f"Can you {spec.description.lower().rstrip('.')}?",
    ]

    return [
        {
            "messages": [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": agent_name},
            ],
            "metadata": {
                "source": "agent_description",
                "agent": agent_name,
                "has_capabilities": False,
            },
        }
        for prompt in prompts[: config.samples_per_agent]
    ]


def _dedupe_prompts(prompts: list[str]) -> list[str]:
    """Preserve order while removing duplicate routing prompts."""
    seen: set[str] = set()
    deduped: list[str] = []
    for prompt in prompts:
        normalized = prompt.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped
