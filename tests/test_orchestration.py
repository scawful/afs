from __future__ import annotations

from afs.orchestration import Orchestrator, TaskRequest
from afs.schema import AgentConfig, OrchestratorConfig


def test_orchestrator_disabled_returns_note() -> None:
    config = OrchestratorConfig(enabled=False)
    orchestrator = Orchestrator(config=config)
    plan = orchestrator.plan(TaskRequest(summary="Test"))
    assert not plan.agents
    assert "orchestrator disabled" in plan.notes


def test_orchestrator_matches_tags() -> None:
    config = OrchestratorConfig(
        enabled=True,
        max_agents=2,
        default_agents=[
            AgentConfig(name="planner", role="planner", tags=["plan"]),
            AgentConfig(name="builder", role="coder", tags=["build"]),
        ],
    )
    orchestrator = Orchestrator(config=config)
    plan = orchestrator.plan(TaskRequest(summary="Build", tags=["build"]))
    assert len(plan.agents) == 1
    assert plan.agents[0].name == "builder"
