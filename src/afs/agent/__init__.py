"""Unified agent module for local and cloud model access with tool support.

This module provides a consistent interface for running models (Ollama, Gemini,
Anthropic, OpenAI) with tool access and AFS integration, regardless of whether
the code is running through Claude Code, Gemini CLI, or standalone.

Key components:
    - AgentHarness: Core runtime with tool loop
    - ModelBackend: Abstraction over model providers
    - Tool: Model-agnostic tool definition
    - TrainingExportHook: Auto-export quality interactions
"""

from .models import (
    ModelBackend,
    OllamaBackend,
    GeminiBackend,
    ModelConfig,
    GenerateResult,
)
from .tools import (
    Tool,
    ToolResult,
    AFS_TOOLS,
    TRIFORCE_TOOLS,
)
from .harness import (
    AgentHarness,
    AgentResult,
    HarnessConfig,
    run_agent,
)
from .hooks import (
    TrainingExportHook,
    HookConfig,
    create_training_hook,
)

__all__ = [
    # Models
    "ModelBackend",
    "OllamaBackend",
    "GeminiBackend",
    "ModelConfig",
    "GenerateResult",
    # Tools
    "Tool",
    "ToolResult",
    "AFS_TOOLS",
    "TRIFORCE_TOOLS",
    # Harness
    "AgentHarness",
    "AgentResult",
    "HarnessConfig",
    "run_agent",
    # Hooks
    "TrainingExportHook",
    "HookConfig",
    "create_training_hook",
]
