"""Orchestration layer for Triforce MoE with Gemini thinking.

Architecture:
    User Query
        ↓
    [Gemini 3 Flash Preview - Planner]
        - Analyzes task with thinking
        - Creates execution plan
        - Identifies experts/tools needed
        ↓
    [Orchestrator]
        - Executes plan steps
        - Dispatches to din/nayru/farore
        - Calls file/debugger tools
        ↓
    [Response Synthesis]
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from google import genai
from google.genai import types

from .router import MoERouter, RouterConfig
from .classifier import QueryIntent

logger = logging.getLogger(__name__)


class ToolType(Enum):
    """Available tools for orchestration."""

    # Triforce experts
    DIN = "din"           # Optimization expert
    NAYRU = "nayru"       # Generation expert
    FARORE = "farore"     # Debugging expert

    # File operations
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"

    # Assembly tools
    ASSEMBLE = "assemble"
    DISASSEMBLE = "disassemble"

    # Emulator/debugger
    YAZE_DEBUG = "yaze_debug"


@dataclass
class PlanStep:
    """A single step in the execution plan."""

    tool: ToolType
    description: str
    input_data: dict[str, Any] = field(default_factory=dict)
    depends_on: list[int] = field(default_factory=list)  # Step indices


@dataclass
class ExecutionPlan:
    """Full execution plan from the planner."""

    goal: str
    reasoning: str
    steps: list[PlanStep]

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "reasoning": self.reasoning,
            "steps": [
                {
                    "tool": s.tool.value,
                    "description": s.description,
                    "input_data": s.input_data,
                    "depends_on": s.depends_on,
                }
                for s in self.steps
            ],
        }


@dataclass
class StepResult:
    """Result from executing a plan step."""

    step_index: int
    tool: ToolType
    success: bool
    output: str
    error: str | None = None


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""

    planner_model: str = "gemini-3-flash-preview"  # Latest Gemini 3 with thinking
    thinking_budget: int = 8192  # More thinking budget for complex plans
    max_steps: int = 10
    enable_file_tools: bool = True
    enable_debug_tools: bool = True
    moe_config: RouterConfig | None = None

    # Tool paths
    asar_path: str = "/Users/scawful/src/third_party/asar-repo/build/asar/bin/asar"
    temp_dir: str = "/tmp/afs_asm"


PLANNER_SYSTEM_PROMPT = """You are a 65816 assembly and ROM hacking expert planner. Create simple, focused execution plans.

## Available Tools

### Triforce Experts (specialized 65816 models)
- **din**: Optimization expert - use for making code faster/smaller
- **nayru**: Generation expert - use for writing NEW assembly code
- **farore**: Debugging expert - use for finding and fixing bugs

### Assembly Tools
- **assemble**: Compile assembly to machine code (validates syntax)
- **disassemble**: Convert machine code to assembly

## Your Task

Create a JSON execution plan. IMPORTANT: Use "depends_on" to chain steps - the output from step N is passed to steps that depend on it.

```json
{
    "goal": "Brief description",
    "reasoning": "Your thinking",
    "steps": [
        {"tool": "nayru", "description": "Generate code", "input_data": {"query": "task"}, "depends_on": []},
        {"tool": "din", "description": "Optimize", "input_data": {"query": "optimize"}, "depends_on": [0]},
        {"tool": "assemble", "description": "Compile", "input_data": {}, "depends_on": [1]}
    ]
}
```

## CRITICAL: Dependencies

- Step 0 has no dependencies: `"depends_on": []`
- Step 1 needs Step 0's output: `"depends_on": [0]`
- Step 2 needs Step 1's output: `"depends_on": [1]`

## Examples

### Simple generation:
Query: "write code to read controller"
```json
{"goal": "Generate controller code", "reasoning": "Code generation task", "steps": [{"tool": "nayru", "description": "Generate code", "input_data": {"query": "Read SNES controller joypad registers"}, "depends_on": []}]}
```

### Generate + Optimize + Assemble:
Query: "write controller code, optimize it, assemble"
```json
{"goal": "Generate, optimize, assemble controller code", "reasoning": "Multi-step with dependencies", "steps": [{"tool": "nayru", "description": "Generate", "input_data": {"query": "Read controller"}, "depends_on": []}, {"tool": "din", "description": "Optimize", "input_data": {"query": "Optimize for size"}, "depends_on": [0]}, {"tool": "assemble", "description": "Compile", "input_data": {}, "depends_on": [1]}]}
```
"""


class Planner:
    """Uses Gemini with thinking to plan task execution."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.client = genai.Client()

    async def create_plan(self, query: str, context: str = "") -> ExecutionPlan:
        """Create an execution plan for the given query."""

        prompt = f"""## Context
{context if context else "No additional context."}

## User Request
{query}

Create an execution plan to accomplish this task. Respond with JSON only."""

        try:
            response = self.client.models.generate_content(
                model=self.config.planner_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=PLANNER_SYSTEM_PROMPT,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=self.config.thinking_budget
                    ),
                    response_mime_type="application/json",
                ),
            )

            # Parse the plan
            plan_data = json.loads(response.text)

            steps = []
            for step_data in plan_data.get("steps", []):
                try:
                    tool = ToolType(step_data["tool"])
                except ValueError:
                    logger.warning(f"Unknown tool: {step_data['tool']}")
                    continue

                # Parse depends_on - ensure integers
                depends_on_raw = step_data.get("depends_on", [])
                depends_on = []
                for dep in depends_on_raw:
                    if isinstance(dep, int):
                        depends_on.append(dep)
                    elif isinstance(dep, str) and dep.isdigit():
                        depends_on.append(int(dep))

                steps.append(PlanStep(
                    tool=tool,
                    description=step_data.get("description", ""),
                    input_data=step_data.get("input_data", {}),
                    depends_on=depends_on,
                ))

            return ExecutionPlan(
                goal=plan_data.get("goal", query),
                reasoning=plan_data.get("reasoning", ""),
                steps=steps,
            )

        except Exception as e:
            logger.error(f"Planning failed: {e}")
            # Fallback: simple single-step plan
            return ExecutionPlan(
                goal=query,
                reasoning=f"Planning failed ({e}), falling back to direct expert query",
                steps=[PlanStep(
                    tool=ToolType.NAYRU,
                    description="Direct query to generation expert",
                    input_data={"query": query},
                )],
            )


class Orchestrator:
    """Executes plans using Triforce experts and tools."""

    def __init__(self, config: OrchestratorConfig | None = None):
        self.config = config or OrchestratorConfig()
        self.planner = Planner(self.config)
        self._router: MoERouter | None = None
        self._tools: dict[ToolType, Callable] = {}

        # Register default tools
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register built-in tools."""

        # File tools
        if self.config.enable_file_tools:
            self._tools[ToolType.READ_FILE] = self._tool_read_file
            self._tools[ToolType.WRITE_FILE] = self._tool_write_file

        # Expert tools are handled specially via router
        self._tools[ToolType.DIN] = self._tool_expert
        self._tools[ToolType.NAYRU] = self._tool_expert
        self._tools[ToolType.FARORE] = self._tool_expert

        # Assembly tools
        self._tools[ToolType.ASSEMBLE] = self._tool_assemble
        self._tools[ToolType.DISASSEMBLE] = self._tool_disassemble

        # Debugger tools
        if self.config.enable_debug_tools:
            self._tools[ToolType.YAZE_DEBUG] = self._tool_yaze_debug

    async def __aenter__(self) -> "Orchestrator":
        """Async context manager entry."""
        moe_config = self.config.moe_config or RouterConfig.default()
        self._router = MoERouter(moe_config)
        await self._router.__aenter__()
        return self

    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        if self._router:
            await self._router.__aexit__(*args)
            self._router = None

    async def run(
        self,
        query: str,
        context: str = "",
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Run the full orchestration pipeline.

        Args:
            query: User's request
            context: Additional context (file contents, etc.)
            verbose: Whether to include detailed step results

        Returns:
            Dict with goal, plan, results, and final_response
        """
        if not self._router:
            raise RuntimeError("Orchestrator not initialized. Use async context manager.")

        # Phase 1: Planning
        if verbose:
            logger.info("Phase 1: Creating execution plan...")

        plan = await self.planner.create_plan(query, context)

        if verbose:
            logger.info(f"Plan: {plan.goal}")
            logger.info(f"Reasoning: {plan.reasoning[:200]}...")
            logger.info(f"Steps: {len(plan.steps)}")

        # Phase 2: Execution
        if verbose:
            logger.info("Phase 2: Executing plan...")

        results: list[StepResult] = []
        step_outputs: dict[int, str] = {}

        for i, step in enumerate(plan.steps):
            if i >= self.config.max_steps:
                logger.warning(f"Max steps ({self.config.max_steps}) reached")
                break

            # Check dependencies
            for dep_idx in step.depends_on:
                if dep_idx >= len(results) or not results[dep_idx].success:
                    results.append(StepResult(
                        step_index=i,
                        tool=step.tool,
                        success=False,
                        output="",
                        error=f"Dependency on step {dep_idx} not satisfied",
                    ))
                    continue

            # Execute step
            if verbose:
                logger.info(f"Step {i}: {step.tool.value} - {step.description}")

            result = await self._execute_step(step, step_outputs)
            result.step_index = i  # Set correct step index
            results.append(result)

            if result.success:
                step_outputs[i] = result.output

        # Phase 3: Synthesis
        if verbose:
            logger.info("Phase 3: Synthesizing response...")

        final_response = self._synthesize_response(plan, results)

        return {
            "goal": plan.goal,
            "reasoning": plan.reasoning,
            "plan": plan.to_dict(),
            "results": [
                {
                    "step": r.step_index,
                    "tool": r.tool.value,
                    "success": r.success,
                    "output": r.output[:500] if verbose else r.output[:100],
                    "error": r.error,
                }
                for r in results
            ],
            "final_response": final_response,
        }

    async def _execute_step(
        self,
        step: PlanStep,
        previous_outputs: dict[int, str],
    ) -> StepResult:
        """Execute a single plan step."""

        try:
            # Inject previous outputs into input_data if referenced
            input_data = dict(step.input_data)
            for dep_idx in step.depends_on:
                if dep_idx in previous_outputs:
                    input_data[f"step_{dep_idx}_output"] = previous_outputs[dep_idx]

            # Get tool handler
            if step.tool in (ToolType.DIN, ToolType.NAYRU, ToolType.FARORE):
                output = await self._tool_expert(step.tool, input_data)
            elif step.tool in self._tools:
                output = await self._tools[step.tool](input_data)
            else:
                return StepResult(
                    step_index=0,
                    tool=step.tool,
                    success=False,
                    output="",
                    error=f"Tool {step.tool.value} not implemented",
                )

            return StepResult(
                step_index=-1,  # Set by caller
                tool=step.tool,
                success=True,
                output=output,
            )

        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            return StepResult(
                step_index=-1,  # Set by caller
                tool=step.tool,
                success=False,
                output="",
                error=str(e),
            )

    async def _tool_expert(
        self,
        tool: ToolType,
        input_data: dict[str, Any],
    ) -> str:
        """Call a Triforce expert."""

        # Handle various input key names from planner
        query = input_data.get("query", "") or input_data.get("prompt", "")
        code = (
            input_data.get("code", "")
            or input_data.get("assembly_code", "")
            or input_data.get("asm", "")
        )

        # Build prompt with intent-specific framing
        intent_prefixes = {
            ToolType.DIN: "Optimize this 65816 assembly code. Output ONLY the optimized code in a ```asm block, no explanations:",
            ToolType.NAYRU: "Write complete, working 65816 assembly code. Do not ask clarifying questions. Generate the code now:",
            ToolType.FARORE: "Debug and fix this 65816 assembly code. Output the fixed code in a ```asm block:",
        }

        prefix = intent_prefixes.get(tool, "")

        # Include previous step outputs if available (find any step_N_output)
        prev_output = ""
        for key, value in input_data.items():
            if key.startswith("step_") and key.endswith("_output") and value:
                prev_output = value
                logger.debug(f"Found prev_output from {key}: {len(value)} chars")
                break
        if not prev_output:
            prev_output = input_data.get("previous_output", "")

        # Prioritize: prev_output (from dependencies) > explicit code > nothing
        # Only use explicit code if it looks like actual assembly (not a placeholder)
        if prev_output and len(prev_output) > 50:
            effective_code = prev_output
        elif code and len(code) > 50:
            effective_code = code
        else:
            effective_code = prev_output or code
        logger.debug(f"Expert {tool.value}: code={len(code)}, prev_output={len(prev_output)}, effective_code={len(effective_code)}")

        if effective_code and query:
            prompt = f"{prefix} {query}\n\n```asm\n{effective_code}\n```"
        elif effective_code:
            prompt = f"{prefix}\n\n```asm\n{effective_code}\n```"
        else:
            prompt = f"{prefix} {query}" if prefix else query

        # Map tool to expert config for direct model selection
        expert_map = {
            ToolType.DIN: "din-v3-fewshot:latest",
            ToolType.NAYRU: "nayru-v5:latest",
            ToolType.FARORE: "farore-v1:latest",
        }

        model_id = expert_map.get(tool)
        if not model_id:
            # Fallback to router decision
            result = await self._router.generate(prompt, use_rag=True)
            return result.content

        # Try local Ollama first, fallback to Gemini
        import httpx
        host = "http://localhost:11435"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{host}/api/generate",
                    json={
                        "model": model_id,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.5, "top_p": 0.85},
                    },
                )
                response.raise_for_status()
                data = response.json()
            output = data.get("response", "")
        except Exception as e:
            # Fallback to Gemini when Ollama unavailable
            logger.warning(f"Ollama unavailable ({e}), using Gemini fallback for {tool.value}")
            output = await self._gemini_expert_fallback(tool, prompt)

        # Post-process din output to extract clean code
        if tool == ToolType.DIN:
            output = self._extract_din_code(output)

        return output

    async def _gemini_expert_fallback(self, tool: ToolType, prompt: str) -> str:
        """Use Gemini as fallback when local experts unavailable."""
        from google.genai import types

        # ASAR syntax rules to inject into all prompts
        asar_rules = """
CRITICAL: Use ASAR assembler syntax (NOT ca65):
- Address operators: label&$FFFF (NOT .LOWORD), label>>16 (NOT .BANKBYTE, NOT ^label)
- Data: db/dw/dl (NOT .BYTE/.WORD)
- Structure: lorom, org $address (NOT .SEGMENT)
- Local labels: .label or -/+ for anonymous
- Defines: !MyConst = $1234

Example DMA source address:
  REP #$20
  LDA #SpriteData&$FFFF   ; Low word (16-bit mode)
  STA $4302
  SEP #$20                ; Switch to 8-bit for bank
  LDA #SpriteData>>16     ; Bank byte (use >>16, NOT ^)
  STA $4304
"""

        # Expert-specific system prompts with ASAR guidance
        expert_prompts = {
            ToolType.DIN: f"""You are Din, a 65816 assembly optimization expert for SNES/ALTTP.
Your task is to optimize assembly code for fewer cycles and smaller size.
{asar_rules}
Output ONLY the optimized assembly code in a ```asm block. No explanations.""",

            ToolType.NAYRU: f"""You are Nayru, a 65816 assembly code generation expert for SNES/ALTTP.
Your task is to write complete, working assembly code for the requested functionality.
{asar_rules}
Output complete, compilable 65816 assembly in a ```asm block. Start with lorom and org $008000.""",

            ToolType.FARORE: f"""You are Farore, a 65816 assembly debugging expert for SNES/ALTTP.
Your task is to identify bugs and provide fixes for assembly code.
{asar_rules}
Output the diagnosis and fixed code in a ```asm block.""",
        }

        system_prompt = expert_prompts.get(tool, "You are a 65816 assembly expert.")

        response = self.planner.client.models.generate_content(
            model=self.config.planner_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.5,
            ),
        )

        return response.text

    def _extract_din_code(self, output: str) -> str:
        """Extract clean assembly code from din's explanatory output.

        din often outputs text like:
            This can be optimized using MVN:

            Original:
            LDA $4218 ...

            Optimized:
            MVN $4218-$4219, $10-$11
        """
        import re

        # Common 65816 opcodes for validation
        opcodes = {
            "LDA", "LDX", "LDY", "STA", "STX", "STY", "STZ",
            "ADC", "SBC", "AND", "ORA", "EOR", "CMP", "CPX", "CPY",
            "INC", "INX", "INY", "DEC", "DEX", "DEY",
            "ASL", "LSR", "ROL", "ROR",
            "BCC", "BCS", "BEQ", "BNE", "BMI", "BPL", "BVC", "BVS", "BRA", "BRL",
            "JMP", "JSR", "JSL", "RTS", "RTL", "RTI",
            "PHA", "PHX", "PHY", "PHP", "PHB", "PHD", "PHK",
            "PLA", "PLX", "PLY", "PLP", "PLB", "PLD",
            "SEC", "CLC", "SEI", "CLI", "SED", "CLD", "CLV",
            "REP", "SEP", "XBA", "XCE", "TCD", "TDC", "TCS", "TSC",
            "TAX", "TAY", "TXA", "TYA", "TXS", "TSX", "TXY", "TYX",
            "NOP", "WDM", "BRK", "COP", "STP", "WAI",
            "MVN", "MVP", "PEA", "PEI", "PER",
        }

        def looks_like_asm(line: str) -> bool:
            """Check if a line looks like assembly code."""
            line = line.strip()
            if not line or line.startswith(";"):
                return True  # Comments are OK
            # Check if first word (before any space/operand) is an opcode
            first_word = line.split()[0].split(":")[0].upper() if line.split() else ""
            # Remove any label prefix (e.g., "Loop:")
            if ":" in line and first_word.endswith(":"):
                # It's a label, that's OK
                return True
            return first_word in opcodes

        # Try to extract from "Optimized:" section first
        if "Optimized:" in output or "optimized:" in output:
            parts = re.split(r"[Oo]ptimized:", output)
            if len(parts) > 1:
                optimized_section = parts[-1]

                # Try to extract code block from optimized section
                code_match = re.search(
                    r"```(?:asm|assembly)?\n(.*?)```",
                    optimized_section,
                    re.DOTALL
                )
                if code_match:
                    return code_match.group(1).strip()

                # Extract only lines that look like assembly
                lines = []
                for line in optimized_section.strip().split("\n"):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    # Stop at explanatory text
                    if stripped.lower().startswith(("the ", "this ", "note:", "total:")):
                        break
                    # Skip "or" alternatives
                    if stripped.lower() == "or":
                        continue
                    # Only include if it looks like assembly
                    if looks_like_asm(stripped):
                        lines.append(stripped)
                    else:
                        # Stop at first non-assembly line (probably explanation)
                        if lines:  # Only if we already have some code
                            break

                if lines:
                    return "\n".join(lines)

        # Try to extract any code block
        code_match = re.search(
            r"```(?:asm|assembly)?\n(.*?)```",
            output,
            re.DOTALL
        )
        if code_match:
            return code_match.group(1).strip()

        # Last resort: return as-is (may fail assembly)
        logger.warning("Could not extract clean code from din output")
        return output

    async def _tool_read_file(self, input_data: dict[str, Any]) -> str:
        """Read a file."""
        path = input_data.get("path", "")
        if not path:
            raise ValueError("No path provided")

        with open(path) as f:
            return f.read()

    async def _tool_write_file(self, input_data: dict[str, Any]) -> str:
        """Write to a file."""
        path = input_data.get("path", "")
        content = input_data.get("content", "")
        if not path:
            raise ValueError("No path provided")

        with open(path, "w") as f:
            f.write(content)

        return f"Wrote {len(content)} bytes to {path}"

    async def _tool_assemble(self, input_data: dict[str, Any]) -> str:
        """Assemble 65816 code using asar."""
        import os
        import re
        import subprocess
        import tempfile

        code = (
            input_data.get("code", "")
            or input_data.get("assembly_code", "")
        )

        # Check for step outputs
        if not code or len(code) < 50:
            for key, value in input_data.items():
                if key.startswith("step_") and key.endswith("_output") and value:
                    code = value
                    break

        if not code:
            return "Error: No assembly code provided"

        # Extract code from markdown blocks if present
        code_match = re.search(r"```(?:asm|assembly)?\n(.*?)```", code, re.DOTALL)
        if code_match:
            code = code_match.group(1)

        # Ensure temp directory exists
        os.makedirs(self.config.temp_dir, exist_ok=True)

        # Write code to temp file
        asm_path = os.path.join(self.config.temp_dir, "input.asm")
        rom_path = os.path.join(self.config.temp_dir, "output.sfc")

        # Add lorom header if not present
        if "lorom" not in code.lower() and "hirom" not in code.lower():
            code = "lorom\n\n" + code

        # Add org if not present
        if "org " not in code.lower():
            code = code.replace("lorom\n", "lorom\n\norg $008000\n")

        with open(asm_path, "w") as f:
            f.write(code)

        # Create empty ROM to patch
        with open(rom_path, "wb") as f:
            f.write(b"\x00" * 0x80000)  # 512KB ROM

        # Run asar
        try:
            result = subprocess.run(
                [self.config.asar_path, "--no-title-check", asm_path, rom_path],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                # Get ROM size
                rom_size = os.path.getsize(rom_path)

                # Count non-zero bytes (actual code)
                with open(rom_path, "rb") as f:
                    rom_data = f.read()
                code_bytes = sum(1 for b in rom_data if b != 0)

                return f"✓ Assembly successful!\n  ROM: {rom_path}\n  Size: {rom_size} bytes\n  Code: {code_bytes} bytes\n\nSource:\n```asm\n{code}\n```"
            else:
                errors = result.stderr or result.stdout
                return f"✗ Assembly failed:\n{errors}\n\nSource:\n```asm\n{code}\n```"

        except subprocess.TimeoutExpired:
            return "Error: Assembly timed out"
        except FileNotFoundError:
            return f"Error: asar not found at {self.config.asar_path}"

    async def _tool_disassemble(self, input_data: dict[str, Any]) -> str:
        """Disassemble machine code using yaze-debugger MCP."""
        address = input_data.get("address", "$008000")
        length = input_data.get("length", 32)

        # Try to call yaze-debugger MCP
        try:
            result = await self._call_mcp_tool(
                "yaze-debugger",
                "disassemble",
                {"address": address, "length": length}
            )
            return result
        except Exception as e:
            return f"Disassembly at {address} ({length} bytes): MCP not available ({e})"

    async def _tool_yaze_debug(self, input_data: dict[str, Any]) -> str:
        """Query yaze emulator state via MCP."""
        command = input_data.get("command", "status")
        args = input_data.get("args", {})

        try:
            result = await self._call_mcp_tool("yaze-debugger", command, args)
            return result
        except Exception as e:
            return f"Yaze debugger error: {e}"

    async def _call_mcp_tool(
        self,
        server: str,
        tool: str,
        args: dict[str, Any],
    ) -> str:
        """Call an MCP tool via subprocess.

        Uses `claude mcp call` to invoke MCP tools.
        """
        import json
        import subprocess

        # Build command
        cmd = [
            "claude", "mcp", "call",
            "--server", server,
            "--tool", tool,
            "--args", json.dumps(args),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                return result.stdout
            else:
                raise RuntimeError(result.stderr or "MCP call failed")

        except FileNotFoundError:
            raise RuntimeError("claude CLI not found - MCP tools unavailable")
        except subprocess.TimeoutExpired:
            raise RuntimeError("MCP call timed out")

    def _synthesize_response(
        self,
        plan: ExecutionPlan,
        results: list[StepResult],
    ) -> str:
        """Combine step results into a final response."""

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        parts = [f"## {plan.goal}\n"]

        if successful:
            # Combine successful outputs
            for result in successful:
                if result.output.strip():
                    parts.append(f"### {result.tool.value}\n{result.output}\n")

        if failed:
            parts.append("\n### Issues\n")
            for result in failed:
                parts.append(f"- {result.tool.value}: {result.error}\n")

        return "\n".join(parts)


async def orchestrate(
    query: str,
    context: str = "",
    verbose: bool = False,
) -> dict[str, Any]:
    """Convenience function to run orchestration."""

    async with Orchestrator() as orch:
        return await orch.run(query, context, verbose)


# CLI interface
async def main():
    """CLI for testing orchestration."""
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python -m afs.moe.orchestrator <query>")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    print(f"Query: {query}\n")
    print("=" * 60)

    result = await orchestrate(query, verbose=True)

    print("\n" + "=" * 60)
    print("\nFinal Response:")
    print(result["final_response"])


if __name__ == "__main__":
    asyncio.run(main())
