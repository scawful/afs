"""Triforce Orchestrator for routing ROM hacking tasks to experts.

Routes tasks to the appropriate Triforce expert models based on
task analysis and manages multi-expert pipelines.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..agent import AgentHarness, HarnessConfig
from .tools import ORACLE_TOOLS, OracleTools, execute_tool

logger = logging.getLogger(__name__)


class Expert(Enum):
    """Available Triforce expert models."""
    NAYRU = "nayru-v7:latest"      # Code generation
    DIN = "din-v4:latest"          # Optimization
    FARORE = "farore-v2:latest"    # Debugging
    VERAN = "veran-v2:latest"      # Hardware knowledge
    ONOX = "onox-v1:latest"        # Data architecture
    TWINROVA = "twinrova-v1:latest"  # State/memory
    AGAHNIM = "agahnim-v1:latest"  # Build/integration


# Expert system prompts for fallback models
EXPERT_SYSTEM_PROMPTS = {
    Expert.NAYRU: """You are Nayru, a 65816 assembly code generation expert for SNES ROM hacking.
You specialize in writing clean, efficient assembly code for the Super Nintendo.
Always output complete, working ASAR-compatible assembly code.
Follow Oracle of Secrets coding conventions when provided.""",

    Expert.DIN: """You are Din, a 65816 assembly optimization expert.
You specialize in making code faster and smaller for the SNES.
Focus on reducing cycles, bytes, and improving efficiency.
Output only optimized code with brief explanations of changes.""",

    Expert.FARORE: """You are Farore, a 65816 assembly debugging expert.
You specialize in finding and fixing bugs in SNES ROM hacks.
Analyze crash logs, memory dumps, and code to identify issues.
Explain the root cause and provide corrected code.""",

    Expert.VERAN: """You are Veran, a SNES hardware expert.
You have deep knowledge of PPU, DMA, HDMA, and all SNES registers.
Provide accurate technical information about hardware operations.
Reference specific register addresses ($2100-$21FF, $4200-$43FF).""",

    Expert.ONOX: """You are Onox, a data architecture expert for SNES ROM hacking.
You specialize in designing efficient data tables, structures, and formats.
Focus on memory-efficient layouts and fast lookup patterns.""",

    Expert.TWINROVA: """You are Twinrova, a state machine and memory expert.
You specialize in WRAM/SRAM layouts, save systems, and game state management.
Design robust state machines and memory-efficient flag systems.""",

    Expert.AGAHNIM: """You are Agahnim, a build and integration expert.
You specialize in ASAR syntax, namespace organization, and patch integration.
Ensure code follows proper org directives and namespace conventions.""",
}


class TaskType(Enum):
    """Categories of ROM hacking tasks."""
    CODE_GENERATION = "code_generation"
    OPTIMIZATION = "optimization"
    DEBUGGING = "debugging"
    HARDWARE = "hardware"
    DATA_STRUCTURE = "data_structure"
    STATE_MACHINE = "state_machine"
    BUILD_INTEGRATION = "build_integration"
    SPRITE = "sprite"
    ITEM = "item"
    MENU = "menu"


@dataclass
class TaskAnalysis:
    """Analysis of a task for routing."""
    task_type: TaskType
    primary_expert: Expert
    pipeline: list[Expert]
    keywords_matched: list[str]
    context_needed: list[str]
    confidence: float


@dataclass
class ExpertResult:
    """Result from an expert invocation."""
    expert: Expert
    response: str
    latency_ms: float
    tools_used: list[str] = field(default_factory=list)


@dataclass
class OrchestratorResult:
    """Complete result from task orchestration."""
    task: str
    analysis: TaskAnalysis
    results: list[ExpertResult]
    final_output: str
    total_latency_ms: float


class TriforceOrchestrator:
    """Orchestrates ROM hacking tasks across Triforce expert models."""

    OPTIMIZATION_KEYWORDS = [
        "optimize", "optimization", "faster", "smaller", "speed", "size",
        "cycle", "cycles", "byte", "bytes",
    ]

    # Keyword mappings for task classification
    EXPERT_KEYWORDS = {
        Expert.DIN: OPTIMIZATION_KEYWORDS,
        Expert.FARORE: ["bug", "crash", "BRK", "debug", "fix", "wrong", "error", "corrupt"],
        Expert.VERAN: ["DMA", "HDMA", "register", "PPU", "VRAM", "OAM", "$21", "$42", "$43"],
        Expert.ONOX: ["table", "struct", "data", "format", "layout", "bitfield", "palette"],
        Expert.TWINROVA: ["state", "WRAM", "SRAM", "flag", "machine", "transition", "save"],
        Expert.AGAHNIM: ["build", "org", "namespace", "hook", "incsrc", "asar", "patch"],
        Expert.NAYRU: ["write", "create", "implement", "code", "routine", "function"],
    }

    # Task type to expert pipeline mappings
    TASK_PIPELINES = {
        TaskType.SPRITE: [Expert.TWINROVA, Expert.ONOX, Expert.NAYRU],
        TaskType.ITEM: [Expert.ONOX, Expert.TWINROVA, Expert.NAYRU],
        TaskType.MENU: [Expert.ONOX, Expert.NAYRU, Expert.VERAN],
        TaskType.OPTIMIZATION: [Expert.DIN],
        TaskType.DEBUGGING: [Expert.FARORE, Expert.VERAN],
        TaskType.HARDWARE: [Expert.VERAN],
        TaskType.DATA_STRUCTURE: [Expert.ONOX],
        TaskType.STATE_MACHINE: [Expert.TWINROVA, Expert.NAYRU],
        TaskType.BUILD_INTEGRATION: [Expert.AGAHNIM],
        TaskType.CODE_GENERATION: [Expert.NAYRU],
    }

    # Context injection paths
    CONTEXT_PATHS = {
        "sprites": ["Docs/SpriteCreationGuide.md", "Docs/General/DevelopmentGuidelines.md"],
        "items": ["Docs/Items.md", "Docs/General/DevelopmentGuidelines.md"],
        "menu": ["Docs/Menu.md"],
        "masks": ["Docs/Masks.md"],
        "memory": ["Docs/Core/MemoryMap.md"],
        "overworld": ["Docs/Overworld.md", "Docs/World/Overworld/ZSCustomOverworldAdvanced.md"],
        "dungeons": ["Docs/Dungeons.md"],
        "build": ["Docs/General/DevelopmentGuidelines.md", "Docs/General/AsarUsage.md"],
    }

    # Fallback model when Triforce experts aren't available
    FALLBACK_MODEL = "gemini-2.0-flash"

    def __init__(
        self,
        oracle_project_path: Path | None = None,
        verbose: bool = False,
        inject_context: bool = True,
        fallback_model: str | None = None,
        use_fallback: bool = True,
    ):
        """Initialize the orchestrator.

        Args:
            oracle_project_path: Path to Oracle of Secrets project
            verbose: Enable verbose logging
            inject_context: Whether to inject Oracle docs as context
            fallback_model: Model to use when Triforce experts unavailable
            use_fallback: Whether to use fallback when experts unavailable
        """
        self.oracle_path = oracle_project_path or Path.home() / "src" / "hobby" / "oracle-of-secrets"
        self.verbose = verbose
        self.inject_context = inject_context
        self.fallback_model = fallback_model or self.FALLBACK_MODEL
        self.use_fallback = use_fallback
        self.tools = OracleTools()
        self._available_models: set[str] | None = None

    async def _check_model_available(self, model_id: str) -> bool:
        """Check if a model is available in Ollama."""
        if self._available_models is None:
            self._available_models = await self._get_ollama_models()

        # Check exact match or base name match
        base_name = model_id.split(":")[0]
        return model_id in self._available_models or base_name in self._available_models

    async def _get_ollama_models(self) -> set[str]:
        """Get list of available Ollama models."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = set()
                    for model in data.get("models", []):
                        name = model.get("name", "")
                        models.add(name)
                        # Also add base name without tag
                        if ":" in name:
                            models.add(name.split(":")[0])
                    return models
        except Exception as e:
            logger.debug(f"Could not query Ollama: {e}")
        return set()

    def analyze_task(self, task: str) -> TaskAnalysis:
        """Analyze a task to determine routing.

        Args:
            task: The task description

        Returns:
            TaskAnalysis with routing decisions
        """
        task_lower = task.lower()
        matched_keywords: dict[Expert, list[str]] = {e: [] for e in Expert}

        # Match keywords to experts
        for expert, keywords in self.EXPERT_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in task_lower:
                    matched_keywords[expert].append(kw)

        # Determine primary expert by match count
        expert_scores = {
            expert: len(matches)
            for expert, matches in matched_keywords.items()
        }

        # Default to Nayru for general code tasks
        primary_expert = max(expert_scores.items(), key=lambda x: x[1])[0]
        if expert_scores[primary_expert] == 0:
            primary_expert = Expert.NAYRU

        # Determine task type
        task_type = self._classify_task_type(task_lower)
        if task_type != TaskType.OPTIMIZATION and primary_expert == Expert.DIN:
            primary_expert = Expert.NAYRU

        # Get pipeline for task type
        pipeline = self.TASK_PIPELINES.get(task_type, [primary_expert])

        # Determine what context to inject
        context_needed = self._determine_context(task_lower)

        # Calculate confidence
        total_matches = sum(len(m) for m in matched_keywords.values())
        confidence = min(1.0, total_matches / 3)  # 3+ matches = high confidence

        all_matched = []
        for matches in matched_keywords.values():
            all_matched.extend(matches)

        return TaskAnalysis(
            task_type=task_type,
            primary_expert=primary_expert,
            pipeline=pipeline,
            keywords_matched=all_matched,
            context_needed=context_needed,
            confidence=confidence,
        )

    def _classify_task_type(self, task: str) -> TaskType:
        """Classify the task type from description."""
        if any(kw in task for kw in ["sprite", "enemy", "npc", "boss"]):
            return TaskType.SPRITE
        if any(kw in task for kw in ["item", "inventory", "equipment"]):
            return TaskType.ITEM
        if any(kw in task for kw in ["menu", "hud", "ui", "interface"]):
            return TaskType.MENU
        if any(kw in task for kw in self.OPTIMIZATION_KEYWORDS):
            return TaskType.OPTIMIZATION
        if any(kw in task for kw in ["bug", "crash", "debug", "fix"]):
            return TaskType.DEBUGGING
        if any(kw in task for kw in ["dma", "hdma", "vram", "register"]):
            return TaskType.HARDWARE
        if any(kw in task for kw in ["table", "data", "struct", "format"]):
            return TaskType.DATA_STRUCTURE
        if any(kw in task for kw in ["state", "machine", "wram", "sram"]):
            return TaskType.STATE_MACHINE
        if any(kw in task for kw in ["build", "hook", "namespace", "org"]):
            return TaskType.BUILD_INTEGRATION
        return TaskType.CODE_GENERATION

    def _determine_context(self, task: str) -> list[str]:
        """Determine what context documents to inject."""
        needed = []
        for category, paths in self.CONTEXT_PATHS.items():
            if category in task:
                needed.extend(paths)
        # Always include development guidelines for code tasks
        if not needed:
            needed.append("Docs/General/DevelopmentGuidelines.md")
        return list(set(needed))  # Dedupe

    def _load_context(self, paths: list[str]) -> str:
        """Load context documents."""
        context_parts = []
        for rel_path in paths:
            full_path = self.oracle_path / rel_path
            if full_path.exists():
                content = full_path.read_text()
                # Truncate to reasonable size
                if len(content) > 4000:
                    content = content[:4000] + "\n... (truncated)"
                context_parts.append(f"## {rel_path}\n\n{content}")
        return "\n\n---\n\n".join(context_parts)

    async def invoke_expert(
        self,
        expert: Expert,
        prompt: str,
        context: str = "",
    ) -> ExpertResult:
        """Invoke a single expert model.

        Args:
            expert: Expert to invoke
            prompt: Task prompt
            context: Optional context to prepend

        Returns:
            ExpertResult with response
        """
        import time
        from ..agent.models import ModelConfig, ModelProvider

        # Build full prompt with context
        full_prompt = prompt
        if context:
            full_prompt = f"## Reference Documentation\n\n{context}\n\n## Task\n\n{prompt}"

        # Check if Triforce model is available
        model_id = expert.value
        use_triforce = await self._check_model_available(model_id)

        if use_triforce:
            logger.info(f"Using Triforce model: {model_id}")
            model_config = ModelConfig.from_string(model_id)
        elif self.use_fallback:
            logger.info(f"Triforce model {model_id} not available, using fallback: {self.fallback_model}")
            model_config = ModelConfig.from_string(self.fallback_model)
            # Apply expert's system prompt to fallback model
            model_config.system_prompt = EXPERT_SYSTEM_PROMPTS.get(expert, "")
        else:
            return ExpertResult(
                expert=expert,
                response=f"Error: Model {model_id} not available and fallback disabled",
                latency_ms=0,
            )

        config = HarnessConfig(max_iterations=3, verbose=self.verbose)
        harness = AgentHarness(model_config, tools=None, config=config)
        harness.tools = {}  # No tools for basic invocation

        start = time.time()
        try:
            async with harness:
                result = await harness.run(full_prompt)
                latency = (time.time() - start) * 1000

                return ExpertResult(
                    expert=expert,
                    response=result.response,
                    latency_ms=latency,
                )
        except Exception as e:
            return ExpertResult(
                expert=expert,
                response=f"Error: {e}",
                latency_ms=(time.time() - start) * 1000,
            )

    async def run_pipeline(
        self,
        task: str,
        analysis: TaskAnalysis,
    ) -> OrchestratorResult:
        """Run a multi-expert pipeline.

        Args:
            task: Original task
            analysis: Task analysis with pipeline

        Returns:
            Complete orchestration result
        """
        import time
        start = time.time()

        results: list[ExpertResult] = []

        # Load context once
        context = ""
        if self.inject_context and analysis.context_needed:
            context = self._load_context(analysis.context_needed)

        # Run pipeline
        current_input = task
        for i, expert in enumerate(analysis.pipeline):
            logger.info(f"[{i+1}/{len(analysis.pipeline)}] Invoking {expert.name}...")

            # For subsequent experts, include previous results
            if i > 0 and results:
                current_input = f"{task}\n\n## Previous Expert ({results[-1].expert.name}) Output:\n\n{results[-1].response}"

            result = await self.invoke_expert(expert, current_input, context if i == 0 else "")
            results.append(result)

            if self.verbose:
                logger.info(f"  Response length: {len(result.response)} chars")

        total_latency = (time.time() - start) * 1000

        # Combine final output
        if len(results) == 1:
            final_output = results[0].response
        else:
            # For multi-expert, format combined output
            parts = [f"## {r.expert.name} Output\n\n{r.response}" for r in results]
            final_output = "\n\n---\n\n".join(parts)

        return OrchestratorResult(
            task=task,
            analysis=analysis,
            results=results,
            final_output=final_output,
            total_latency_ms=total_latency,
        )

    async def run(self, task: str) -> OrchestratorResult:
        """Run a task through the orchestrator.

        Args:
            task: Task description

        Returns:
            Complete orchestration result
        """
        # Analyze task
        analysis = self.analyze_task(task)

        if self.verbose:
            logger.info(f"Task analysis:")
            logger.info(f"  Type: {analysis.task_type.value}")
            logger.info(f"  Primary: {analysis.primary_expert.name}")
            logger.info(f"  Pipeline: {[e.name for e in analysis.pipeline]}")
            logger.info(f"  Keywords: {analysis.keywords_matched}")
            logger.info(f"  Confidence: {analysis.confidence:.2f}")

        # Run pipeline
        return await self.run_pipeline(task, analysis)

    async def quick_query(self, task: str) -> str:
        """Quick single-expert query.

        Args:
            task: Task description

        Returns:
            Expert response string
        """
        analysis = self.analyze_task(task)
        result = await self.invoke_expert(
            analysis.primary_expert,
            task,
            self._load_context(analysis.context_needed) if self.inject_context else "",
        )
        return result.response


async def main():
    """Test the orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(description="Triforce Orchestrator")
    parser.add_argument("task", nargs="?", help="Task to run")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--no-context", action="store_true", help="Disable context injection")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    orchestrator = TriforceOrchestrator(
        verbose=args.verbose,
        inject_context=not args.no_context,
    )

    if args.task:
        result = await orchestrator.run(args.task)
        print(f"\n{'='*60}")
        print(f"Task: {result.task}")
        print(f"Type: {result.analysis.task_type.value}")
        print(f"Pipeline: {[e.name for e in result.analysis.pipeline]}")
        print(f"Latency: {result.total_latency_ms:.0f}ms")
        print(f"{'='*60}")
        print(result.final_output)
    else:
        # Interactive mode
        print("Triforce Orchestrator - Interactive Mode")
        print("Type 'quit' to exit\n")

        while True:
            try:
                task = input("Task> ").strip()
                if task.lower() in ["quit", "exit", "q"]:
                    break
                if not task:
                    continue

                result = await orchestrator.run(task)
                print(f"\n[{result.analysis.task_type.value}] via {[e.name for e in result.analysis.pipeline]}")
                print(f"({result.total_latency_ms:.0f}ms)\n")
                print(result.final_output)
                print()

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
