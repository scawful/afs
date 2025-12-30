"""Minimal orchestration helpers for routing tasks to agents."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Iterable

from .config import load_config_model
from .schema import AgentConfig, OrchestratorConfig


@dataclass
class TaskRequest:
    summary: str
    tags: list[str] = field(default_factory=list)
    role: str | None = None


@dataclass
class OrchestrationPlan:
    summary: str
    agents: list[AgentConfig]
    notes: list[str] = field(default_factory=list)


class Orchestrator:
    def __init__(self, config: OrchestratorConfig | None = None) -> None:
        self.config = config or load_config_model().orchestrator

    def list_agents(self) -> list[AgentConfig]:
        return list(self.config.default_agents)

    def plan(self, request: TaskRequest) -> OrchestrationPlan:
        if not self.config.enabled:
            return OrchestrationPlan(
                summary=request.summary,
                agents=[],
                notes=["orchestrator disabled"],
            )

        candidates = list(self.config.default_agents)
        if request.role:
            candidates = [a for a in candidates if a.role == request.role]

        if request.tags:
            tagged = [
                agent
                for agent in candidates
                if set(request.tags) & set(agent.tags)
            ]
            if tagged:
                candidates = tagged

        if not candidates:
            return OrchestrationPlan(
                summary=request.summary,
                agents=[],
                notes=["no matching agents"],
            )

        selected = candidates[: self.config.max_agents]
        notes = []
        if len(candidates) > len(selected):
            notes.append("truncated agent list to max_agents")

        return OrchestrationPlan(
            summary=request.summary,
            agents=selected,
            notes=notes,
        )


def _list_command(args: argparse.Namespace) -> int:
    orchestrator = Orchestrator()
    agents = orchestrator.list_agents()
    for agent in agents:
        tags = ",".join(agent.tags) if agent.tags else "-"
        print(f"{agent.name}\t{agent.role}\t{agent.backend}\t{tags}")
    return 0


def _plan_command(args: argparse.Namespace) -> int:
    orchestrator = Orchestrator()
    request = TaskRequest(
        summary=args.summary,
        tags=args.tag or [],
        role=args.role,
    )
    plan = orchestrator.plan(request)
    if plan.notes:
        for note in plan.notes:
            print(f"note: {note}")
    for agent in plan.agents:
        tags = ",".join(agent.tags) if agent.tags else "-"
        print(f"{agent.name}\t{agent.role}\t{agent.backend}\t{tags}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="afs-orchestrator")
    sub = parser.add_subparsers(dest="command")

    list_cmd = sub.add_parser("list", help="List configured agents.")
    list_cmd.set_defaults(func=_list_command)

    plan_cmd = sub.add_parser("plan", help="Plan a routing decision.")
    plan_cmd.add_argument("summary", help="Task summary.")
    plan_cmd.add_argument("--tag", action="append", help="Tag to match.")
    plan_cmd.add_argument("--role", help="Role to match.")
    plan_cmd.set_defaults(func=_plan_command)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
