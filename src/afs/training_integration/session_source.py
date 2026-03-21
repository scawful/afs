"""Extract training samples from AFS session replay timelines.

Converts real agent interaction sequences into training data by walking
session timelines and extracting MCP tool call patterns, CLI invocations,
and agent coordination events as instruction/response pairs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..event_log import (
    build_session_replay,
    build_session_timeline,
    list_recorded_sessions,
    list_sessions,
)
from ..generators.base import TrainingSample, write_jsonl

logger = logging.getLogger(__name__)


@dataclass
class SessionExtractionConfig:
    """Configuration for extracting training data from sessions."""

    min_events_per_session: int = 3
    include_event_types: set[str] = field(
        default_factory=lambda: {"mcp_tool", "cli", "context", "hivemind"}
    )
    format_type: str = "chatml"  # chatml or alpaca
    system_prompt: str = (
        "You are an AFS (Agent File System) assistant. "
        "Given a user request, select and execute the appropriate "
        "AFS tool or operation."
    )
    quality_floor: float = 0.6


@dataclass
class ExtractionResult:
    """Result of extracting training data from sessions."""

    sessions_scanned: int = 0
    sessions_with_data: int = 0
    samples_extracted: int = 0
    samples_filtered: int = 0
    output_path: Path | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def extract_samples_from_timeline(
    timeline: dict[str, Any],
    *,
    config: SessionExtractionConfig | None = None,
) -> list[TrainingSample]:
    """Extract training samples from a single session timeline.

    Walks the timeline looking for meaningful event sequences:
    - MCP tool calls → instruction is the tool + args, output is the result summary
    - CLI invocations → instruction is the command, output is the operation description
    - Context operations → instruction is the operation, output is the result
    - Hivemind sequences → agent coordination patterns
    """
    cfg = config or SessionExtractionConfig()
    events = timeline.get("timeline", [])
    if len(events) < cfg.min_events_per_session:
        return []

    samples: list[TrainingSample] = []
    session_id = timeline.get("session_id", "unknown")

    # Group consecutive events into interaction windows
    windows = _build_interaction_windows(events, cfg.include_event_types)

    for window in windows:
        sample = _window_to_sample(window, session_id=session_id, config=cfg)
        if sample is not None:
            samples.append(sample)

    return samples


def extract_from_sessions(
    context_path: Path,
    *,
    output_path: Path | None = None,
    config: SessionExtractionConfig | None = None,
    session_limit: int = 20,
    afs_config: Any = None,
) -> ExtractionResult:
    """Extract training samples from all recent sessions.

    Scans session history, extracts interaction patterns, and writes
    training samples to JSONL.
    """
    cfg = config or SessionExtractionConfig()
    result = ExtractionResult(output_path=output_path)

    sessions = list_recorded_sessions(
        context_path,
        config=afs_config,
        limit=session_limit,
    )
    replay_mode = True
    if not sessions:
        sessions = list_sessions(context_path, config=afs_config, limit=session_limit)
        replay_mode = False
    result.sessions_scanned = len(sessions)

    all_samples: list[TrainingSample] = []

    for session in sessions:
        sid = session.get("session_id", "")
        if not sid:
            continue

        if replay_mode:
            replay = build_session_replay(
                context_path,
                session_id=sid,
                limit=500,
                config=afs_config,
            )
            timeline = _replay_to_timeline(replay)
        else:
            timeline = build_session_timeline(
                context_path,
                session_id=sid,
                limit=500,
                config=afs_config,
            )

        samples = extract_samples_from_timeline(timeline, config=cfg)
        if samples:
            result.sessions_with_data += 1
            all_samples.extend(samples)

    result.samples_extracted = len(all_samples)

    # Filter low-quality samples
    filtered = [s for s in all_samples if s.quality_score >= cfg.quality_floor]
    result.samples_filtered = len(all_samples) - len(filtered)

    if output_path and filtered:
        write_jsonl(filtered, output_path)
        logger.info("Wrote %d session-replay samples to %s", len(filtered), output_path)

    return result


def _replay_to_timeline(replay: dict[str, Any]) -> dict[str, Any]:
    """Convert ``build_session_replay`` output into timeline format."""
    events: list[dict[str, Any]] = []
    for event in replay.get("events", []):
        metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
        summary = str(metadata.get("tool_name", "")).strip()
        if str(event.get("type", "")).strip() == "mcp_tool" and summary:
            summary = f"MCP tool: {summary}"
        if not summary:
            summary = str(event.get("op", "")).strip()
        events.append(
            {
                "timestamp": str(event.get("timestamp", "")),
                "type": str(event.get("type", "")),
                "op": str(event.get("op", "")),
                "source": str(event.get("source", "")),
                "id": str(event.get("id", "")),
                "summary": summary,
            }
        )
    return {
        "session_id": replay.get("session_id"),
        "since": None,
        "event_count": len(events),
        "timeline": events,
    }


def _build_interaction_windows(
    events: list[dict[str, Any]],
    include_types: set[str],
) -> list[list[dict[str, Any]]]:
    """Group events into interaction windows.

    A window is a sequence of related events that form a logical interaction.
    Windows break on:
    - Session bootstrap events (new interaction boundary)
    - Gaps > 5 minutes between events
    - Event type transitions (mcp_tool → cli)
    """
    windows: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for event in events:
        event_type = event.get("type", "")
        if event_type not in include_types:
            continue

        # Break on session boundaries
        if event_type == "session" and event.get("op") == "bootstrap":
            if current:
                windows.append(current)
                current = []
            continue

        # Break on type transitions if current window is non-empty
        if current and current[-1].get("type") != event_type:
            windows.append(current)
            current = []

        current.append(event)

    if current:
        windows.append(current)

    return windows


def _window_to_sample(
    window: list[dict[str, Any]],
    *,
    session_id: str,
    config: SessionExtractionConfig,
) -> TrainingSample | None:
    """Convert an interaction window into a training sample."""
    if not window:
        return None

    first = window[0]
    event_type = first.get("type", "")

    if event_type == "mcp_tool":
        return _mcp_tool_window_to_sample(window, session_id=session_id, config=config)
    elif event_type == "cli":
        return _cli_window_to_sample(window, session_id=session_id, config=config)
    elif event_type == "context":
        return _context_window_to_sample(window, session_id=session_id, config=config)
    elif event_type == "hivemind":
        return _hivemind_window_to_sample(window, session_id=session_id, config=config)

    return None


def _mcp_tool_window_to_sample(
    window: list[dict[str, Any]],
    *,
    session_id: str,
    config: SessionExtractionConfig,
) -> TrainingSample | None:
    """Extract training sample from MCP tool call sequence."""
    tool_names = []
    summaries = []

    for event in window:
        summary = event.get("summary", "")
        tool_name = summary.replace("MCP tool: ", "") if summary.startswith("MCP tool:") else ""
        if tool_name:
            tool_names.append(tool_name)
        summaries.append(summary)

    if not tool_names:
        return None

    # Build instruction from the tool sequence
    if len(tool_names) == 1:
        instruction = f"Use the {tool_names[0]} tool to complete the requested operation."
    else:
        tool_list = ", ".join(tool_names)
        instruction = (
            f"Execute the following AFS tool sequence: {tool_list}. "
            f"Coordinate the tools to complete the operation."
        )

    # Build output from the event summaries
    output_lines = [f"Step {i+1}: {s}" for i, s in enumerate(summaries)]
    output = "\n".join(output_lines)

    # Quality heuristic: more tools in sequence = more complex = higher value
    quality = min(0.5 + 0.1 * len(tool_names), 1.0)

    return TrainingSample(
        instruction=instruction,
        output=output,
        domain="afs_tools",
        source=f"session_replay:{session_id}",
        quality_score=quality,
    )


def _cli_window_to_sample(
    window: list[dict[str, Any]],
    *,
    session_id: str,
    config: SessionExtractionConfig,
) -> TrainingSample | None:
    """Extract training sample from CLI invocation sequence."""
    commands = []
    for event in window:
        summary = event.get("summary", "")
        if summary.startswith("CLI:"):
            commands.append(summary[4:].strip())
        elif summary.startswith("CLI invocation"):
            commands.append(event.get("op", "unknown"))

    if not commands:
        return None

    instruction = f"Execute the AFS CLI command: {commands[0]}"
    if len(commands) > 1:
        instruction = "Execute the following AFS CLI sequence:\n" + "\n".join(
            f"  {i+1}. {cmd}" for i, cmd in enumerate(commands)
        )

    output = "\n".join(
        f"Executed: {cmd}" for cmd in commands
    )

    quality = min(0.5 + 0.05 * len(commands), 0.9)

    return TrainingSample(
        instruction=instruction,
        output=output,
        domain="afs_cli",
        source=f"session_replay:{session_id}",
        quality_score=quality,
    )


def _context_window_to_sample(
    window: list[dict[str, Any]],
    *,
    session_id: str,
    config: SessionExtractionConfig,
) -> TrainingSample | None:
    """Extract training sample from context operations."""
    ops = []
    for event in window:
        summary = event.get("summary", "")
        if summary:
            ops.append(summary)

    if not ops:
        return None

    instruction = f"Perform context operation: {ops[0]}"
    output = "\n".join(ops)

    return TrainingSample(
        instruction=instruction,
        output=output,
        domain="afs_context",
        source=f"session_replay:{session_id}",
        quality_score=0.6,
    )


def _hivemind_window_to_sample(
    window: list[dict[str, Any]],
    *,
    session_id: str,
    config: SessionExtractionConfig,
) -> TrainingSample | None:
    """Extract training sample from hivemind coordination."""
    agents = set()
    ops = []
    for event in window:
        summary = event.get("summary", "")
        source = event.get("source", "")
        if source:
            agents.add(source)
        ops.append(summary)

    if not ops:
        return None

    agent_list = ", ".join(sorted(agents)) if agents else "agents"
    instruction = f"Coordinate {agent_list} via hivemind for: {ops[0]}"
    output = "\n".join(ops)

    # Multi-agent coordination is high-value training data
    quality = min(0.6 + 0.1 * len(agents), 1.0)

    return TrainingSample(
        instruction=instruction,
        output=output,
        domain="afs_hivemind",
        source=f"session_replay:{session_id}",
        quality_score=quality,
    )
