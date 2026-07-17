from __future__ import annotations

import pytest

from afs.cli import build_parser


@pytest.mark.parametrize(
    ("argv", "destination"),
    [
        (["files", "list", "scratchpad"], "fs_command"),
        (["jobs", "list"], "agent_jobs_command"),
        (["missions", "list"], "mission_command"),
        (["repair"], "func"),
        (["check"], "func"),
    ],
)
def test_friendly_aliases_route_to_existing_commands(
    argv: list[str], destination: str
) -> None:
    args = build_parser(argv).parse_args(argv)

    assert hasattr(args, destination)
