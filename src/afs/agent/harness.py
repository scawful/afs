"""Core agent runtime with tool loop.

Provides a unified execution environment for any model (local or cloud)
with AFS tool access and automatic training data export.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import ModelBackend, ModelConfig, ToolCall, create_backend
from .tools import AFS_TOOLS, Tool, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class HarnessConfig:
    """Configuration for agent harness."""

    max_iterations: int = 10
    max_history_messages: int = 20
    verbose: bool = False
    context_root: Path = field(default_factory=lambda: Path.home() / ".context")
    log_to_history: bool = True
    auto_export: bool = True
    export_min_quality: float = 0.6


@dataclass
class ToolExecution:
    """Record of a single tool execution."""

    name: str
    arguments: dict[str, Any]
    result: ToolResult
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentResult:
    """Result from agent execution."""

    response: str
    history: list[dict[str, Any]]
    tool_executions: list[ToolExecution]
    iterations: int
    success: bool
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "response": self.response,
            "history": self.history,
            "tool_executions": [
                {
                    "name": te.name,
                    "arguments": te.arguments,
                    "result": te.result.to_dict(),
                    "timestamp": te.timestamp.isoformat(),
                }
                for te in self.tool_executions
            ],
            "iterations": self.iterations,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }


class AgentHarness:
    """Unified runtime for local and cloud models with tool access.

    Provides a consistent agentic interface regardless of model provider.
    Models can call tools iteratively until task completion.

    Example:
        ```python
        harness = AgentHarness("nayru-v5:latest", tools=TRIFORCE_TOOLS)

        async with harness:
            result = await harness.run("Write a DMA transfer routine")
            print(result.response)
        ```
    """

    def __init__(
        self,
        model: str | ModelConfig,
        tools: list[Tool] | None = None,
        config: HarnessConfig | None = None,
    ):
        """Initialize agent harness.

        Args:
            model: Model identifier or config
            tools: Tools available to the model (default: AFS_TOOLS)
            config: Harness configuration
        """
        self.config = config or HarnessConfig()
        self.tools = {t.name: t for t in (tools or AFS_TOOLS)}
        self._model_config = (
            model if isinstance(model, ModelConfig) else ModelConfig.from_string(model)
        )
        self._backend: ModelBackend | None = None
        self._hooks: list[Any] = []

    async def __aenter__(self) -> AgentHarness:
        """Initialize backend."""
        self._backend = create_backend(self._model_config)
        return self

    async def __aexit__(self, *args) -> None:
        """Clean up backend."""
        if self._backend:
            await self._backend.close()
            self._backend = None

    def add_hook(self, hook: Any) -> None:
        """Add a hook to be called on agent events."""
        self._hooks.append(hook)

    async def run(
        self,
        prompt: str,
        context: str = "",
        system_prompt: str | None = None,
    ) -> AgentResult:
        """Execute agent loop until completion or max iterations.

        Args:
            prompt: User's request
            context: Additional context to prepend
            system_prompt: Override the model's system prompt

        Returns:
            AgentResult with response, history, and tool executions
        """
        if not self._backend:
            raise RuntimeError(
                "Harness not initialized. Use async context manager."
            )

        # Build initial messages
        messages: list[dict[str, Any]] = []

        # System prompt (override if provided)
        effective_system = system_prompt or self._model_config.system_prompt
        if effective_system:
            messages.append({"role": "system", "content": effective_system})

        # User message with optional context
        user_content = f"{context}\n\n{prompt}" if context else prompt
        messages.append({"role": "user", "content": user_content})

        # Convert tools to OpenAI format for the backend
        tool_defs = [t.to_openai() for t in self.tools.values()]

        # Track executions
        tool_executions: list[ToolExecution] = []
        final_response = ""

        for iteration in range(self.config.max_iterations):
            if self.config.verbose:
                logger.info(f"Iteration {iteration + 1}/{self.config.max_iterations}")

            # Bounded Context enforcement
            # If history exceeds limit, we truncate to simulate "Bounded Context"
            # and force reliance on external state (scratchpad/files).
            if len(messages) > self.config.max_history_messages:
                logger.info("Context limit exceeded. Truncating history.")

                # Preserve system prompt if it exists (usually index 0)
                new_messages = []
                if messages and messages[0]["role"] == "system":
                    new_messages.append(messages[0])

                # Keep the last N messages
                                # We subtract 2 to leave room for system prompt + warning
                                retain_count = self.config.max_history_messages - 2
                                if retain_count < 1:
                                    retain_count = 1
                
                                new_messages.append({                    "role": "system",
                    "content": "Context truncated due to length limits. Please read 'scratchpad/state.md' or query files to restore your understanding of the task state."
                })
                new_messages.extend(messages[-retain_count:])

                messages = new_messages

            # Generate
            try:
                result = await self._backend.generate(
                    messages=messages,
                    tools=tool_defs if self.tools else None,
                )
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                return AgentResult(
                    response="",
                    history=messages,
                    tool_executions=tool_executions,
                    iterations=iteration + 1,
                    success=False,
                    error=str(e),
                )

            # Check for tool calls
            if result.has_tool_calls:
                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": result.content,
                    "tool_calls": [
                        {"name": tc.name, "arguments": tc.arguments}
                        for tc in result.tool_calls
                    ],
                })

                # Execute tools
                tool_results = await self._execute_tools(result.tool_calls)
                tool_executions.extend(tool_results)

                # Add tool results to messages
                messages.append({
                    "role": "tool",
                    "results": [
                        {
                            "name": te.name,
                            "content": te.result.content if te.result.success else te.result.error,
                        }
                        for te in tool_results
                    ],
                })

                # Continue loop
                continue

            # No tool calls = final response
            final_response = result.content
            break

        else:
            # Max iterations reached
            logger.warning(f"Max iterations ({self.config.max_iterations}) reached")
            return AgentResult(
                response=final_response or "Max iterations reached without completion",
                history=messages,
                tool_executions=tool_executions,
                iterations=self.config.max_iterations,
                success=False,
                error="max_iterations",
            )

        # Build successful result
        agent_result = AgentResult(
            response=final_response,
            history=messages,
            tool_executions=tool_executions,
            iterations=iteration + 1,
            success=True,
            metadata={
                "model": self._model_config.model_id,
                "provider": self._model_config.provider.value,
            },
        )

        # Call hooks
        for hook in self._hooks:
            if hasattr(hook, "on_agent_complete"):
                try:
                    await hook.on_agent_complete(agent_result)
                except Exception as e:
                    logger.error(f"Hook failed: {e}")

        return agent_result

    async def _execute_tools(
        self,
        tool_calls: list[ToolCall],
    ) -> list[ToolExecution]:
        """Execute a list of tool calls."""
        executions = []

        for tc in tool_calls:
            if tc.name not in self.tools:
                result = ToolResult(
                    success=False,
                    content="",
                    error=f"Unknown tool: {tc.name}",
                )
            else:
                tool = self.tools[tc.name]
                if self.config.verbose:
                    logger.info(f"Executing tool: {tc.name}")
                result = await tool.execute(tc.arguments)

            executions.append(
                ToolExecution(
                    name=tc.name,
                    arguments=tc.arguments,
                    result=result,
                )
            )

        return executions


async def run_agent(
    model: str,
    prompt: str,
    tools: list[Tool] | None = None,
    context: str = "",
    verbose: bool = False,
) -> AgentResult:
    """Convenience function to run a single agent query.

    Args:
        model: Model identifier (e.g., "nayru-v5:latest", "gemini-3-flash-preview")
        prompt: User's request
        tools: Tools available (default: AFS_TOOLS)
        context: Additional context
        verbose: Enable verbose logging

    Returns:
        AgentResult
    """
    config = HarnessConfig(verbose=verbose)
    harness = AgentHarness(model, tools=tools, config=config)

    async with harness:
        return await harness.run(prompt, context)


# CLI interface
async def main():
    """CLI for testing agent harness."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run agent with tool access")
    parser.add_argument("prompt", help="Prompt for the agent")
    parser.add_argument(
        "--model",
        default="nayru-v5:latest",
        help="Model to use (default: nayru-v5:latest)",
    )
    parser.add_argument(
        "--tools",
        choices=["afs", "triforce", "none"],
        default="afs",
        help="Tool set to use (default: afs)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Max tool loop iterations (default: 10)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Select tools
    if args.tools == "none":
        tools = []
    elif args.tools == "triforce":
        from .tools import TRIFORCE_TOOLS
        tools = TRIFORCE_TOOLS
    else:
        from .tools import AFS_TOOLS
        tools = AFS_TOOLS

    # Run agent
    config = HarnessConfig(
        max_iterations=args.max_iterations,
        verbose=args.verbose,
    )

    harness = AgentHarness(args.model, tools=tools, config=config)

    print(f"Model: {args.model}")
    print(f"Tools: {args.tools} ({len(tools)} available)")
    print(f"Prompt: {args.prompt}")
    print("-" * 60)

    async with harness:
        result = await harness.run(args.prompt)

    print("\n" + "=" * 60)
    print(f"Success: {result.success}")
    print(f"Iterations: {result.iterations}")
    print(f"Tool executions: {len(result.tool_executions)}")

    if result.tool_executions:
        print("\nTool calls:")
        for te in result.tool_executions:
            status = "✓" if te.result.success else "✗"
            print(f"  {status} {te.name}({list(te.arguments.keys())})")

    print("\nResponse:")
    print(result.response)

    if result.error:
        print(f"\nError: {result.error}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
