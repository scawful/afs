"""Claude log analysis helpers.

This package extracts actionable summaries from Claude Code transcripts and
subagent artifacts, suitable for writing into AFS project contexts.
"""

from .session_report import (  # noqa: F401
    ClaudeSessionPaths,
    ClaudeSessionReport,
    SubagentSummary,
    build_session_report,
    discover_session_paths,
    render_session_report_markdown,
)

