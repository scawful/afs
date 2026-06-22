from __future__ import annotations

import pytest

from afs.grounding_hooks import run_grounding_hooks
from afs.schema import AFSConfig, ProfileConfig, ProfilesConfig


def _work_config() -> AFSConfig:
    return AFSConfig(
        profiles=ProfilesConfig(
            active_profile="work",
            auto_apply=True,
            profiles={
                "work": ProfileConfig(policies=["deny_keywords:classified,internal-only"]),
            },
        )
    )


def test_deny_keywords_policy_blocks_agent_dispatch() -> None:
    with pytest.raises(PermissionError):
        run_grounding_hooks(
            event="before_agent_dispatch",
            payload={"summary": "Review classified launch issue"},
            config=_work_config(),
        )


def test_deny_keywords_policy_allows_unrelated_dispatch() -> None:
    run_grounding_hooks(
        event="before_agent_dispatch",
        payload={"summary": "Review Antigravity/Gemini MCP integration"},
        config=_work_config(),
    )
