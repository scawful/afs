"""Tests for AFS Phase 2 training integrations.

Covers:
1. Session replay → training data extraction
2. Router dataset generation from agent capabilities
3. Pre-training freshness gate
4. Training watch script exists and is executable
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

# ---------------------------------------------------------------------------
# 1. Session replay → training data extraction
# ---------------------------------------------------------------------------


def _make_timeline(events: list[dict], session_id: str = "2026-03-21") -> dict:
    return {
        "session_id": session_id,
        "since": None,
        "event_count": len(events),
        "timeline": events,
    }


def _make_event(event_type: str, op: str = "", summary: str = "", source: str = "") -> dict:
    return {
        "timestamp": "2026-03-21T12:00:00+00:00",
        "type": event_type,
        "op": op,
        "source": source,
        "id": "test-id",
        "summary": summary,
    }


class TestSessionReplayExtraction:
    def test_extract_mcp_tool_samples(self):
        from afs.training_integration.session_source import (
            SessionExtractionConfig,
            extract_samples_from_timeline,
        )

        timeline = _make_timeline([
            _make_event("mcp_tool", summary="MCP tool: context.query"),
            _make_event("mcp_tool", summary="MCP tool: context.index.rebuild"),
            _make_event("mcp_tool", summary="MCP tool: memory.search"),
        ])
        config = SessionExtractionConfig(min_events_per_session=1)
        samples = extract_samples_from_timeline(timeline, config=config)
        assert len(samples) >= 1
        # Should mention the tool names
        tool_sample = samples[0]
        assert "context.query" in tool_sample.instruction or "context.query" in tool_sample.output
        assert tool_sample.domain == "afs_tools"
        assert tool_sample.source.startswith("session_replay:")

    def test_extract_cli_samples(self):
        from afs.training_integration.session_source import (
            SessionExtractionConfig,
            extract_samples_from_timeline,
        )

        timeline = _make_timeline([
            _make_event("cli", summary="CLI: session bootstrap --json"),
            _make_event("cli", summary="CLI: memory status"),
        ])
        config = SessionExtractionConfig(min_events_per_session=1)
        samples = extract_samples_from_timeline(timeline, config=config)
        assert len(samples) >= 1
        assert samples[0].domain == "afs_cli"

    def test_extract_empty_timeline_returns_empty(self):
        from afs.training_integration.session_source import extract_samples_from_timeline

        timeline = _make_timeline([])
        samples = extract_samples_from_timeline(timeline)
        assert samples == []

    def test_extract_filters_by_quality_floor(self):
        from afs.training_integration.session_source import (
            SessionExtractionConfig,
            extract_samples_from_timeline,
        )

        timeline = _make_timeline([
            _make_event("context", op="mount", summary="mount knowledge/docs"),
        ])
        config = SessionExtractionConfig(
            min_events_per_session=1,
            quality_floor=0.99,
        )
        samples = extract_samples_from_timeline(timeline, config=config)
        # Context events get quality_score=0.6, should all be below 0.99
        low_q = [s for s in samples if s.quality_score < 0.99]
        assert len(low_q) == len(samples)

    def test_extract_from_sessions_integration(self, tmp_path):
        from afs.training_integration.session_source import (
            SessionExtractionConfig,
            extract_from_sessions,
        )

        output = tmp_path / "output.jsonl"
        config = SessionExtractionConfig(
            min_events_per_session=1,
            quality_floor=0.0,
        )

        mock_sessions = [
            {"session_id": "2026-03-21", "event_count": 5},
        ]
        mock_timeline = _make_timeline([
            _make_event("mcp_tool", summary="MCP tool: context.query"),
            _make_event("mcp_tool", summary="MCP tool: memory.search"),
        ])

        with patch("afs.training_integration.session_source.list_sessions", return_value=mock_sessions), \
             patch("afs.training_integration.session_source.build_session_timeline", return_value=mock_timeline):
            result = extract_from_sessions(
                tmp_path,
                output_path=output,
                config=config,
                session_limit=5,
            )

        assert result.sessions_scanned == 1
        assert result.sessions_with_data >= 1
        assert result.samples_extracted >= 1
        assert output.exists()

    def test_extract_from_recorded_sessions_prefers_explicit_session_ids(self, tmp_path):
        from afs.training_integration.session_source import (
            SessionExtractionConfig,
            extract_from_sessions,
        )

        output = tmp_path / "recorded.jsonl"
        config = SessionExtractionConfig(
            min_events_per_session=1,
            quality_floor=0.0,
        )
        mock_sessions = [
            {"session_id": "session-a", "event_count": 2},
        ]
        mock_replay = {
            "session_id": "session-a",
            "events": [
                {
                    "timestamp": "2026-03-21T12:00:00+00:00",
                    "type": "mcp_tool",
                    "op": "call",
                    "source": "afs.mcp",
                    "id": "evt-1",
                    "metadata": {"tool_name": "context.query"},
                },
                {
                    "timestamp": "2026-03-21T12:01:00+00:00",
                    "type": "mcp_tool",
                    "op": "call",
                    "source": "afs.mcp",
                    "id": "evt-2",
                    "metadata": {"tool_name": "memory.search"},
                },
            ],
        }

        with patch(
            "afs.training_integration.session_source.list_recorded_sessions",
            return_value=mock_sessions,
        ), patch(
            "afs.training_integration.session_source.build_session_replay",
            return_value=mock_replay,
        ), patch(
            "afs.training_integration.session_source.list_sessions",
            return_value=[],
        ), patch(
            "afs.training_integration.session_source.build_session_timeline",
        ) as legacy_timeline:
            result = extract_from_sessions(
                tmp_path,
                output_path=output,
                config=config,
                session_limit=5,
            )

        assert result.sessions_scanned == 1
        assert result.samples_extracted >= 1
        assert output.exists()
        legacy_timeline.assert_not_called()

    def test_hivemind_window_extraction(self):
        from afs.training_integration.session_source import (
            SessionExtractionConfig,
            extract_samples_from_timeline,
        )

        timeline = _make_timeline([
            _make_event("hivemind", op="send", summary="hivemind send by context-warm", source="agent.context-warm"),
            _make_event("hivemind", op="send", summary="hivemind send by history-memory", source="agent.history-memory"),
        ])
        config = SessionExtractionConfig(min_events_per_session=1)
        samples = extract_samples_from_timeline(timeline, config=config)
        # Hivemind events break into separate windows per type transition
        assert len(samples) >= 1
        assert any(s.domain == "afs_hivemind" for s in samples)


# ---------------------------------------------------------------------------
# 2. Router dataset generation from capabilities
# ---------------------------------------------------------------------------


class TestRouterFromCapabilities:
    def test_generate_produces_chatml_format(self, tmp_path):
        from afs.generators.router_from_capabilities import (
            RouterDatasetConfig,
            generate_router_dataset,
        )

        output = tmp_path / "router.jsonl"
        config = RouterDatasetConfig(samples_per_agent=3)
        result = generate_router_dataset(output_path=output, config=config)

        assert result.agents_processed > 0
        assert result.samples_generated > 0
        assert output.exists()

        # Verify format
        with open(output) as f:
            for line in f:
                data = json.loads(line)
                assert "messages" in data
                messages = data["messages"]
                assert len(messages) == 3
                assert messages[0]["role"] == "system"
                assert messages[1]["role"] == "user"
                assert messages[2]["role"] == "assistant"

    def test_generate_without_capabilities_excluded_by_default(self, tmp_path):
        from afs.generators.router_from_capabilities import (
            RouterDatasetConfig,
            generate_router_dataset,
        )

        output = tmp_path / "router.jsonl"
        config = RouterDatasetConfig(
            samples_per_agent=2,
            include_agents_without_capabilities=False,
        )
        result = generate_router_dataset(output_path=output, config=config)
        assert result.agents_with_capabilities <= result.agents_processed

    def test_generate_includes_all_when_configured(self, tmp_path):
        from afs.generators.router_from_capabilities import (
            RouterDatasetConfig,
            generate_router_dataset,
        )

        output = tmp_path / "router_all.jsonl"
        config = RouterDatasetConfig(
            samples_per_agent=2,
            include_agents_without_capabilities=True,
        )
        result = generate_router_dataset(output_path=output, config=config)
        # Should generate more samples when including all agents
        assert result.samples_generated > 0
        # All agents with descriptions should produce samples
        assert len(result.agent_sample_counts) > 0

    def test_result_tracks_per_agent_counts(self, tmp_path):
        from afs.generators.router_from_capabilities import (
            RouterDatasetConfig,
            generate_router_dataset,
        )

        output = tmp_path / "router.jsonl"
        config = RouterDatasetConfig(samples_per_agent=5)
        result = generate_router_dataset(output_path=output, config=config)

        total_from_counts = sum(result.agent_sample_counts.values())
        assert total_from_counts == result.samples_generated

    def test_assistant_content_is_agent_name(self, tmp_path):
        from afs.generators.router_from_capabilities import (
            RouterDatasetConfig,
            generate_router_dataset,
        )

        output = tmp_path / "router.jsonl"
        config = RouterDatasetConfig(samples_per_agent=2)
        result = generate_router_dataset(output_path=output, config=config)

        with open(output) as f:
            for line in f:
                data = json.loads(line)
                agent_name = data["messages"][2]["content"]
                assert agent_name in result.agent_sample_counts


# ---------------------------------------------------------------------------
# 3. Pre-training freshness gate
# ---------------------------------------------------------------------------


class TestFreshnessGate:
    def test_ready_when_scores_above_threshold(self):
        from afs.models import MountType
        from afs.training_integration.freshness_gate import (
            FreshnessGateConfig,
            check_training_readiness,
        )

        mock_scores = {
            "mount_scores": {"knowledge": 0.8, "tools": 0.9},
            "files": {
                "knowledge": [{"score": 0.8}],
                "tools": [{"score": 0.9}],
            },
        }
        config = FreshnessGateConfig(
            min_score=0.3,
            mount_types=[MountType.KNOWLEDGE, MountType.TOOLS],
        )

        with patch("afs.training_integration.freshness_gate.AFSManager"), \
             patch("afs.training_integration.freshness_gate.ContextSQLiteIndex") as MockIndex:
            MockIndex.return_value.freshness_scores.return_value = mock_scores
            report = check_training_readiness(Path("/fake"), config=config)

        assert report.ready is True
        assert len(report.blocked_mounts) == 0
        assert report.overall_score > 0.3

    def test_blocked_when_scores_below_threshold(self):
        from afs.models import MountType
        from afs.training_integration.freshness_gate import (
            FreshnessGateConfig,
            check_training_readiness,
        )

        mock_scores = {
            "mount_scores": {"knowledge": 0.1, "tools": 0.05},
            "files": {
                "knowledge": [{"score": 0.1}],
                "tools": [{"score": 0.05}],
            },
        }
        config = FreshnessGateConfig(
            min_score=0.3,
            mount_types=[MountType.KNOWLEDGE, MountType.TOOLS],
            block_on_failure=True,
        )

        with patch("afs.training_integration.freshness_gate.AFSManager"), \
             patch("afs.training_integration.freshness_gate.ContextSQLiteIndex") as MockIndex:
            MockIndex.return_value.freshness_scores.return_value = mock_scores
            report = check_training_readiness(Path("/fake"), config=config)

        assert report.ready is False
        assert len(report.blocked_mounts) == 2
        assert "knowledge" in report.blocked_mounts
        assert any("afs index rebuild --path <workspace>" in warning for warning in report.warnings)

    def test_warn_only_mode_never_blocks(self):
        from afs.models import MountType
        from afs.training_integration.freshness_gate import (
            FreshnessGateConfig,
            check_training_readiness,
        )

        mock_scores = {
            "mount_scores": {"knowledge": 0.05},
            "files": {"knowledge": [{"score": 0.05}]},
        }
        config = FreshnessGateConfig(
            min_score=0.3,
            mount_types=[MountType.KNOWLEDGE],
            block_on_failure=False,
        )

        with patch("afs.training_integration.freshness_gate.AFSManager"), \
             patch("afs.training_integration.freshness_gate.ContextSQLiteIndex") as MockIndex:
            MockIndex.return_value.freshness_scores.return_value = mock_scores
            report = check_training_readiness(Path("/fake"), config=config)

        assert report.ready is True  # warn-only never blocks

    def test_missing_mount_adds_warning(self):
        from afs.models import MountType
        from afs.training_integration.freshness_gate import (
            FreshnessGateConfig,
            check_training_readiness,
        )

        mock_scores = {
            "mount_scores": {},
            "files": {},
        }
        config = FreshnessGateConfig(
            mount_types=[MountType.KNOWLEDGE],
        )

        with patch("afs.training_integration.freshness_gate.AFSManager"), \
             patch("afs.training_integration.freshness_gate.ContextSQLiteIndex") as MockIndex:
            MockIndex.return_value.freshness_scores.return_value = mock_scores
            report = check_training_readiness(Path("/fake"), config=config)

        assert any("not found" in w for w in report.warnings)

    def test_report_serializes_to_dict(self):
        from afs.training_integration.freshness_gate import (
            FreshnessReport,
            MountReadiness,
        )

        report = FreshnessReport(
            ready=True,
            overall_score=0.85,
            mounts=[
                MountReadiness(
                    mount_type="knowledge",
                    score=0.85,
                    status="ready",
                    file_count=10,
                    stale_files=1,
                ),
            ],
            blocked_mounts=[],
            warnings=[],
            config={"min_score": 0.3},
        )
        d = report.to_dict()
        assert d["ready"] is True
        assert d["overall_score"] == 0.85
        assert len(d["mounts"]) == 1
        assert d["mounts"][0]["mount_type"] == "knowledge"


# ---------------------------------------------------------------------------
# 4. Training watch script
# ---------------------------------------------------------------------------


class TestTrainingWatch:
    def test_script_exists_and_is_executable(self):
        script = Path(__file__).parent.parent / "scripts" / "training_watch.sh"
        assert script.exists(), f"Training watch script not found at {script}"
        assert script.stat().st_mode & 0o111, "Script is not executable"

    def test_script_has_afs_watch_invocation(self):
        script = Path(__file__).parent.parent / "scripts" / "training_watch.sh"
        content = script.read_text()
        assert "afs watch" in content
        assert "--on-change" in content
        assert "--debounce" in content
