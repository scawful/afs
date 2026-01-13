"""Oracle of Secrets Testing Integration with yaze-mcp.

Provides agentic testing loops that connect Triforce experts
to the emulator testing infrastructure for automated patch validation.
"""

import asyncio
import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# Configuration paths
YAZE_MCP_DIR = Path.home() / "src" / "tools" / "yaze-mcp"
STATE_LIBRARY = Path.home() / ".context" / "knowledge" / "oracle-of-secrets" / "states" / "state_library.json"


class TestStatus(Enum):
    """Status of a test execution."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


@dataclass
class AssertionResult:
    """Result of a single assertion."""
    expression: str
    passed: bool
    actual: Any = None
    expected: Any = None
    error: str | None = None


@dataclass
class PatchTestResult:
    """Complete result of patch testing."""
    status: TestStatus
    patch_valid: bool = False
    build_success: bool = False
    state_loaded: str | None = None
    frames_executed: int = 0
    assertions: list[AssertionResult] = field(default_factory=list)
    crashed: bool = False
    error: str | None = None
    execution_time_ms: int = 0
    screenshot: str | None = None
    final_state: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Overall success of the test."""
        return (
            self.status == TestStatus.PASSED and
            self.patch_valid and
            self.build_success and
            not self.crashed and
            all(a.passed for a in self.assertions)
        )


@dataclass
class OracleTodo:
    """A TODO item from Oracle of Secrets for testing.

    Represents a task that needs implementation and testing,
    derived from oracle.org or manual annotation.
    """
    id: str
    title: str
    description: str
    category: str  # sprite, item, menu, feature, fix
    priority: int = 0
    suggested_state: str = "new_game_start"
    assertions: list[str] = field(default_factory=list)
    context_files: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "OracleTodo":
        return cls(
            id=data.get("id", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            category=data.get("category", "feature"),
            priority=data.get("priority", 0),
            suggested_state=data.get("suggested_state", "new_game_start"),
            assertions=data.get("assertions", []),
            context_files=data.get("context_files", []),
        )

    def to_prompt(self) -> str:
        """Convert to a prompt for expert models."""
        return f"""# Task: {self.title}

## Description
{self.description}

## Category
{self.category}

## Testing
- Test State: {self.suggested_state}
- Assertions: {', '.join(self.assertions) if self.assertions else 'None specified'}

Please implement this feature following Oracle of Secrets coding conventions.
Return valid ASAR-compatible 65816 assembly code."""


@dataclass
class TestIteration:
    """Single iteration of the test-debug loop."""
    iteration: int
    expert: str
    code: str
    test_result: PatchTestResult
    debug_analysis: str | None = None


@dataclass
class AgenticTestResult:
    """Complete result of agentic test loop."""
    todo: OracleTodo
    iterations: list[TestIteration]
    final_code: str | None
    success: bool
    total_time_ms: int


class YazeMCPClient:
    """Client for invoking yaze-mcp tools programmatically.

    Uses subprocess to invoke the MCP server's Python functions directly,
    or can be configured to use gRPC/HTTP if available.
    """

    def __init__(self, venv_path: Path | None = None):
        self.venv_path = venv_path or YAZE_MCP_DIR / "venv"
        self.python = self.venv_path / "bin" / "python"

    def _run_tool(self, tool_name: str, **kwargs) -> dict:
        """Run a yaze-mcp tool and return JSON result.

        Args:
            tool_name: Name of the MCP tool function
            **kwargs: Arguments to pass to the tool

        Returns:
            Parsed JSON result
        """
        # Build inline Python to invoke the tool
        args_json = json.dumps(kwargs)
        code = f'''
import sys
sys.path.insert(0, "{YAZE_MCP_DIR}")
from server import {tool_name}
import json
args = json.loads({repr(args_json)})
result = {tool_name}(**args)
print(result)
'''

        try:
            result = subprocess.run(
                [str(self.python), "-c", code],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(YAZE_MCP_DIR),
            )

            if result.returncode != 0:
                return {"error": result.stderr}

            # Try to parse as JSON
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return {"result": result.stdout, "raw": True}

        except subprocess.TimeoutExpired:
            return {"error": "Timeout"}
        except Exception as e:
            return {"error": str(e)}

    def test_oracle_patch(
        self,
        patch_content: str,
        test_state: str = "new_game_start",
        assertions: list[str] = None,
        frames: int = 120,
        breakpoints: list[str] = None,
        capture_screenshot: bool = True,
        org_address: str = "$008000",
        rom_type: str = "lorom",
    ) -> PatchTestResult:
        """Run the test_oracle_patch tool.

        Args:
            patch_content: 65816 assembly code to test
            test_state: State ID from test library
            assertions: Memory assertions to check
            frames: Frames to run
            breakpoints: Addresses to break on
            capture_screenshot: Include screenshot
            org_address: Origin address for patch
            rom_type: ROM type (lorom/hirom)

        Returns:
            PatchTestResult with test results
        """
        result = self._run_tool(
            "test_oracle_patch",
            patch_content=patch_content,
            test_state=test_state,
            assertions=assertions or [],
            frames=frames,
            breakpoints=breakpoints or [],
            capture_screenshot=capture_screenshot,
            org_address=org_address,
            rom_type=rom_type,
        )

        if "error" in result and not result.get("success", True):
            return PatchTestResult(
                status=TestStatus.ERROR,
                error=result.get("error"),
            )

        # Parse result
        assertion_results = []
        for a in result.get("assertions", []):
            assertion_results.append(AssertionResult(
                expression=a.get("expr", ""),
                passed=a.get("passed", False),
                actual=a.get("actual"),
                expected=a.get("expected"),
                error=a.get("error"),
            ))

        status = TestStatus.PASSED if result.get("success") else TestStatus.FAILED

        return PatchTestResult(
            status=status,
            patch_valid=result.get("patch_valid", False),
            build_success=result.get("build_success", False),
            state_loaded=result.get("state_loaded"),
            frames_executed=result.get("frames_executed", 0),
            assertions=assertion_results,
            crashed=result.get("final_state", {}).get("crashed", False),
            error=result.get("error"),
            execution_time_ms=result.get("execution_time_ms", 0),
            screenshot=result.get("screenshot"),
            final_state=result.get("final_state", {}),
        )

    def validate_asm(self, code: str, rom_type: str = "lorom") -> dict:
        """Validate assembly code syntax."""
        return self._run_tool("validate_asm", code=code, rom_type=rom_type)

    def load_test_state(self, state_id: str) -> dict:
        """Load a test state from the library."""
        return self._run_tool("load_test_state", state_id=state_id)

    def list_test_states(self, category: str = "") -> dict:
        """List available test states."""
        return self._run_tool("list_test_states", category=category)


class AgenticTestLoop:
    """Implements the agentic test-debug loop for patch development.

    Flow:
    1. Expert generates code from TODO
    2. test_oracle_patch validates and tests
    3. If fail, Farore analyzes and suggests fixes
    4. Loop until success or max iterations
    """

    # Tool routing by expert
    EXPERT_TOOLS = {
        "nayru": ["validate_asm", "test_oracle_patch", "lookup_oracle_docs"],
        "din": ["validate_asm", "test_oracle_patch", "lookup_oracle_docs"],
        "farore": ["read_memory", "add_breakpoint", "step_emulator", "get_disassembly"],
        "veran": ["read_memory", "lookup_snes_register", "get_disassembly"],
        "onox": ["lookup_oracle_docs", "search_oracle_code"],
        "twinrova": ["read_memory", "lookup_alttp_ram"],
        "agahnim": ["validate_asm", "validate_namespace"],
    }

    def __init__(
        self,
        max_iterations: int = 5,
        verbose: bool = False,
    ):
        """Initialize the test loop.

        Args:
            max_iterations: Maximum test-debug iterations
            verbose: Enable verbose logging
        """
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.mcp_client = YazeMCPClient()

    def extract_code(self, response: str) -> str | None:
        """Extract assembly code from expert response.

        Args:
            response: Expert model response text

        Returns:
            Extracted assembly code or None
        """
        # Look for code blocks
        code_block_pattern = r'```(?:asm|assembly|65816)?\s*\n(.*?)```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        if matches:
            # Return the longest code block
            return max(matches, key=len).strip()

        # Look for org directives as marker
        lines = response.split("\n")
        code_lines = []
        in_code = False

        for line in lines:
            if re.match(r'\s*(org|pushpc|namespace)', line, re.IGNORECASE):
                in_code = True
            if in_code:
                code_lines.append(line)
            if in_code and re.match(r'\s*(pullpc|RTS|RTL)\s*$', line, re.IGNORECASE):
                break

        if code_lines:
            return "\n".join(code_lines)

        return None

    async def invoke_expert(
        self,
        expert: str,
        prompt: str,
        context: str = "",
    ) -> str:
        """Invoke an expert model.

        This is a placeholder that should be connected to the actual
        model invocation infrastructure (Ollama, vLLM, etc.)

        Args:
            expert: Expert name (nayru, din, farore, etc.)
            prompt: Task prompt
            context: Optional context

        Returns:
            Expert response
        """
        # Import the orchestrator for actual model invocation
        try:
            from .orchestrator import TriforceOrchestrator, Expert

            expert_map = {
                "nayru": Expert.NAYRU,
                "din": Expert.DIN,
                "farore": Expert.FARORE,
                "veran": Expert.VERAN,
                "onox": Expert.ONOX,
                "twinrova": Expert.TWINROVA,
                "agahnim": Expert.AGAHNIM,
            }

            if expert not in expert_map:
                return f"Unknown expert: {expert}"

            orchestrator = TriforceOrchestrator(verbose=self.verbose)
            result = await orchestrator.invoke_expert(
                expert_map[expert],
                prompt,
                context,
            )
            return result.response

        except Exception as e:
            logger.error(f"Expert invocation failed: {e}")
            return f"Error invoking {expert}: {e}"

    async def debug_failure(
        self,
        code: str,
        test_result: PatchTestResult,
    ) -> str:
        """Use Farore to analyze a test failure.

        Args:
            code: The failing code
            test_result: Test result with failure info

        Returns:
            Debug analysis and suggested fixes
        """
        prompt = f"""# Debug Analysis Request

## Failing Code
```asm
{code}
```

## Test Result
- Status: {test_result.status.value}
- ASAR Valid: {test_result.patch_valid}
- Build Success: {test_result.build_success}
- Crashed: {test_result.crashed}
- Error: {test_result.error}

## Failed Assertions
"""
        for a in test_result.assertions:
            if not a.passed:
                prompt += f"- {a.expression}: expected {a.expected}, got {a.actual}\n"
                if a.error:
                    prompt += f"  Error: {a.error}\n"

        prompt += f"""
## Final CPU State
{json.dumps(test_result.final_state, indent=2)}

Please analyze this failure and suggest specific fixes.
Focus on:
1. Why the assertion failed
2. What the code should do differently
3. Specific line changes needed
"""

        return await self.invoke_expert("farore", prompt)

    async def run_todo(
        self,
        todo: OracleTodo,
        initial_expert: str = "nayru",
    ) -> AgenticTestResult:
        """Run the agentic test loop for a TODO.

        Args:
            todo: The TODO item to implement
            initial_expert: Expert to generate initial code

        Returns:
            Complete test result with all iterations
        """
        import time
        start_time = time.time()

        iterations: list[TestIteration] = []
        current_code = None
        success = False

        logger.info(f"Starting agentic test for TODO: {todo.id}")

        # Generate initial code
        prompt = todo.to_prompt()
        response = await self.invoke_expert(initial_expert, prompt)
        current_code = self.extract_code(response)

        if not current_code:
            return AgenticTestResult(
                todo=todo,
                iterations=[],
                final_code=None,
                success=False,
                total_time_ms=int((time.time() - start_time) * 1000),
            )

        # Test-debug loop
        for i in range(self.max_iterations):
            logger.info(f"Iteration {i+1}/{self.max_iterations}")

            # Test the code
            test_result = self.mcp_client.test_oracle_patch(
                patch_content=current_code,
                test_state=todo.suggested_state,
                assertions=todo.assertions,
            )

            iteration = TestIteration(
                iteration=i + 1,
                expert=initial_expert if i == 0 else "farore",
                code=current_code,
                test_result=test_result,
            )

            if test_result.success:
                logger.info("Test passed!")
                success = True
                iterations.append(iteration)
                break

            # Debug the failure
            logger.info(f"Test failed: {test_result.error or 'assertion failure'}")
            debug_analysis = await self.debug_failure(current_code, test_result)
            iteration.debug_analysis = debug_analysis
            iterations.append(iteration)

            # Extract fix from debug analysis
            fix_code = self.extract_code(debug_analysis)
            if fix_code:
                current_code = fix_code
            else:
                # Ask Nayru to regenerate with debug info
                retry_prompt = f"""# Retry: {todo.title}

The previous attempt failed. Here's the analysis:

{debug_analysis}

Please generate corrected code that addresses these issues.

Original task:
{todo.description}
"""
                response = await self.invoke_expert("nayru", retry_prompt)
                current_code = self.extract_code(response)

                if not current_code:
                    logger.warning("Could not extract code from retry response")
                    break

        return AgenticTestResult(
            todo=todo,
            iterations=iterations,
            final_code=current_code if success else None,
            success=success,
            total_time_ms=int((time.time() - start_time) * 1000),
        )

    async def run_batch(
        self,
        todos: list[OracleTodo],
        parallel: int = 1,
    ) -> list[AgenticTestResult]:
        """Run batch testing on multiple TODOs.

        Args:
            todos: List of TODOs to test
            parallel: Number of parallel tests (default 1)

        Returns:
            List of test results
        """
        if parallel <= 1:
            results = []
            for todo in todos:
                result = await self.run_todo(todo)
                results.append(result)
            return results

        # Run in parallel batches
        results = []
        for i in range(0, len(todos), parallel):
            batch = todos[i:i + parallel]
            batch_results = await asyncio.gather(
                *[self.run_todo(todo) for todo in batch]
            )
            results.extend(batch_results)

        return results


def load_oracle_todos(path: Path | str | None = None) -> list[OracleTodo]:
    """Load Oracle TODOs from JSON file.

    Args:
        path: Path to todos JSON file

    Returns:
        List of OracleTodo items
    """
    if path is None:
        path = Path.home() / ".context" / "knowledge" / "oracle-of-secrets" / "todos.json"

    path = Path(path)
    if not path.exists():
        return []

    with open(path) as f:
        data = json.load(f)

    return [OracleTodo.from_dict(item) for item in data.get("todos", [])]


async def main():
    """Test the agentic testing infrastructure."""
    import argparse

    parser = argparse.ArgumentParser(description="Oracle Agentic Testing")
    parser.add_argument("--todo", help="TODO ID to test")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--max-iter", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    # Load todos
    todos = load_oracle_todos()

    if args.todo:
        # Find specific TODO
        todo = next((t for t in todos if t.id == args.todo), None)
        if not todo:
            print(f"TODO not found: {args.todo}")
            return

        loop = AgenticTestLoop(max_iterations=args.max_iter, verbose=args.verbose)
        result = await loop.run_todo(todo)

        print(f"\n{'='*60}")
        print(f"TODO: {result.todo.title}")
        print(f"Success: {result.success}")
        print(f"Iterations: {len(result.iterations)}")
        print(f"Time: {result.total_time_ms}ms")
        print(f"{'='*60}")

        if result.final_code:
            print("\nFinal Code:")
            print(result.final_code)
    else:
        # List available todos
        print("Available TODOs:")
        for todo in todos[:20]:
            print(f"  {todo.id}: {todo.title} [{todo.category}]")


if __name__ == "__main__":
    asyncio.run(main())
