"""Conservative default supervised agent set shipped with AFS.

The supervisor reconciles ``profile.agent_configs``. Profiles without any
configured agents receive this set so a fresh supervisor has useful bounded
work. Existing profile agent lists are never augmented implicitly. Disable
with ``[agents] default_set = false`` or ``AFS_DEFAULT_AGENTS=off``.
"""

from __future__ import annotations

import logging
import os

from .context_paths import resolve_mount_root
from .models import MountType
from .schema import AFSConfig, AgentConfig

DEFAULT_AGENTS_ENV = "AFS_DEFAULT_AGENTS"
DEFAULT_AGENT_TAG = "afs-default"

_ENV_FALSE = {"0", "off", "false", "no"}
_ENV_TRUE = {"1", "on", "true", "yes"}

logger = logging.getLogger(__name__)


def default_agents_enabled(config: AFSConfig | None = None) -> bool:
    """Return whether the shipped default agent set should be merged."""
    raw_env = os.environ.get(DEFAULT_AGENTS_ENV, "").strip()
    env = raw_env.lower()
    if env in _ENV_FALSE:
        return False
    if env in _ENV_TRUE:
        return True
    if raw_env:
        logger.warning(
            "Ignoring invalid %s=%r and disabling default agents; expected one of %s",
            DEFAULT_AGENTS_ENV,
            raw_env,
            sorted(_ENV_FALSE | _ENV_TRUE),
        )
        return False
    if config is not None:
        return config.agents.default_set
    return True


def default_agent_configs(config: AFSConfig | None = None) -> list[AgentConfig]:
    """Return the shipped default agent set for the global context root."""
    active_config = config or AFSConfig()
    context_root = active_config.general.context_root.expanduser().resolve()
    knowledge_root = resolve_mount_root(
        context_root,
        MountType.KNOWLEDGE,
        config=active_config,
    )
    memory_root = resolve_mount_root(
        context_root,
        MountType.MEMORY,
        config=active_config,
    )
    return [
        AgentConfig(
            name="context-warm",
            role="maintenance",
            description="Audit workspace contexts once per day without repair or network calls.",
            tags=[DEFAULT_AGENT_TAG],
            schedule="daily",
            module="afs.agents.default_context_warm",
        ),
        AgentConfig(
            name="index-rebuild",
            role="maintenance",
            description="Rebuild the context SQLite index when knowledge or memory mounts change.",
            tags=[DEFAULT_AGENT_TAG],
            watch_paths=[knowledge_root, memory_root],
            # afs watch announces change batches on this hivemind topic; the
            # reactor makes it the first consumed signal (watch_paths still
            # cover mutations that bypass afs watch).
            on_event=["hivemind:context:repair"],
            module="afs.agents.index_rebuild",
        ),
        AgentConfig(
            name="skills-mine",
            role="learning",
            description="Mine repeated successful session traces into reviewable skill candidates.",
            tags=[DEFAULT_AGENT_TAG],
            schedule="weekly",
            module="afs.agents.skills_mine",
        ),
        AgentConfig(
            name="morning-briefing",
            role="reporting",
            description="Write a daily briefing digest to the scratchpad briefings directory.",
            tags=[DEFAULT_AGENT_TAG],
            schedule="daily",
            module="afs.agents.briefing_agent",
        ),
    ]


def merge_default_agent_configs(
    existing: list[AgentConfig],
    *,
    config: AFSConfig | None = None,
) -> list[AgentConfig]:
    """Return defaults only for an otherwise empty profile agent list."""
    if existing or not default_agents_enabled(config):
        return list(existing)
    return default_agent_configs(config)
