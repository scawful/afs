"""Oracle of Secrets integration module for AFS.

Provides specialized tools and orchestration for ROM hacking tasks,
including emulator-based testing via yaze-mcp integration.
"""

from .tools import OracleTools, ORACLE_TOOLS, execute_tool
from .orchestrator import TriforceOrchestrator, Expert, TaskType
from .testing import (
    AgenticTestLoop,
    OracleTodo,
    PatchTestResult,
    TestStatus,
    YazeMCPClient,
    load_oracle_todos,
)
from .embeddings import (
    OracleEmbeddingGenerator,
    EmbeddingChunk,
    OracleEmbeddingStats,
)

__all__ = [
    # Tools
    "OracleTools",
    "ORACLE_TOOLS",
    "execute_tool",
    # Orchestration
    "TriforceOrchestrator",
    "Expert",
    "TaskType",
    # Testing
    "AgenticTestLoop",
    "OracleTodo",
    "PatchTestResult",
    "TestStatus",
    "YazeMCPClient",
    "load_oracle_todos",
    # Embeddings
    "OracleEmbeddingGenerator",
    "EmbeddingChunk",
    "OracleEmbeddingStats",
]
