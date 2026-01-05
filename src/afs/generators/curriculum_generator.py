"""Curriculum-based generator for scaled training data.

Implements difficulty-stratified data generation for expert models,
with support for multi-provider parallelization, checkpointing,
and progress tracking.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, TYPE_CHECKING

from .base import BaseGenerator, GenerationResult, TrainingSample, write_jsonl

if TYPE_CHECKING:
    from afs.generators.cot.client import LLMClient

logger = logging.getLogger(__name__)


class Difficulty(str, Enum):
    """Difficulty levels for curriculum learning."""

    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ExpertDomain(str, Enum):
    """Expert model domains."""

    DIN = "din"  # Optimization
    NAYRU = "nayru"  # Generation
    FARORE = "farore"  # Debugging
    VERAN = "veran"  # Explanation


@dataclass
class CurriculumTemplate:
    """Template for generating training instructions."""

    domain: ExpertDomain
    difficulty: Difficulty
    instruction_templates: list[str]
    context_hints: list[str] = field(default_factory=list)
    required_entities: list[str] = field(default_factory=list)
    min_output_length: int = 50
    max_output_length: int = 2000
    quality_threshold: float = 0.6
    # Code-rich generation fields
    code_snippets: list[str] = field(default_factory=list)  # Concrete code examples
    code_context: str = ""  # Hardware documentation context
    expected_patterns: list[str] = field(default_factory=list)  # Keywords for validation
    use_pattern_library: bool = False  # Whether to pull from template_libraries


# Template library organized by domain and difficulty
CURRICULUM_TEMPLATES: dict[ExpertDomain, dict[Difficulty, list[CurriculumTemplate]]] = {
    ExpertDomain.DIN: {
        Difficulty.BASIC: [
            CurriculumTemplate(
                domain=ExpertDomain.DIN,
                difficulty=Difficulty.BASIC,
                instruction_templates=[
                    "Optimize this 65816 assembly code for fewer cycles:\n```\n{code}\n```",
                    "Reduce redundant operations in this code:\n```\n{code}\n```",
                    "Apply 65816 optimization techniques to this routine:\n```\n{code}\n```",
                ],
                context_hints=["loop", "register", "addressing", "stack"],
                min_output_length=30,
                quality_threshold=0.5,
                use_pattern_library=True,
                expected_patterns=["STZ", "DEX", "BPL", "REP", "SEP"],
            ),
        ],
        Difficulty.INTERMEDIATE: [
            CurriculumTemplate(
                domain=ExpertDomain.DIN,
                difficulty=Difficulty.INTERMEDIATE,
                instruction_templates=[
                    "Optimize this sprite/DMA routine for better performance:\n```\n{code}\n```",
                    "Reduce cycle count in this table lookup code:\n```\n{code}\n```",
                    "Apply 16-bit optimization to this routine:\n```\n{code}\n```",
                ],
                context_hints=["sprite", "dma", "table", "mode switching"],
                min_output_length=80,
                quality_threshold=0.6,
                use_pattern_library=True,
                expected_patterns=["REP #$20", "long addressing", "unrolled"],
            ),
        ],
        Difficulty.ADVANCED: [
            CurriculumTemplate(
                domain=ExpertDomain.DIN,
                difficulty=Difficulty.ADVANCED,
                instruction_templates=[
                    "Optimize this HDMA/DMA setup for minimal CPU overhead:\n```\n{code}\n```",
                    "Reduce the scanline budget of this rendering code:\n```\n{code}\n```",
                    "Optimize this OAM update routine:\n```\n{code}\n```",
                ],
                context_hints=["hdma", "scanline", "vblank", "oam"],
                min_output_length=120,
                quality_threshold=0.65,
                use_pattern_library=True,
                expected_patterns=["HDMA", "$420B", "repeat mode"],
            ),
        ],
        Difficulty.EXPERT: [
            CurriculumTemplate(
                domain=ExpertDomain.DIN,
                difficulty=Difficulty.EXPERT,
                instruction_templates=[
                    "Optimize this Mode 7 calculation for real-time updates:\n```\n{code}\n```",
                    "Reduce cycle count of this audio routine while maintaining quality:\n```\n{code}\n```",
                ],
                context_hints=["mode7", "audio", "collision", "physics"],
                min_output_length=200,
                quality_threshold=0.7,
                use_pattern_library=True,
                expected_patterns=["$211B", "matrix", "multiplier"],
            ),
        ],
    },
    ExpertDomain.NAYRU: {
        Difficulty.BASIC: [
            CurriculumTemplate(
                domain=ExpertDomain.NAYRU,
                difficulty=Difficulty.BASIC,
                instruction_templates=[
                    "Write a 65816 assembly routine to {task}",
                    "Implement this functionality in 65816 assembly:\n{task}",
                ],
                context_hints=["joypad", "delay", "memcpy", "brightness"],
                required_entities=["joypad_1"],
                min_output_length=30,
                quality_threshold=0.5,
                use_pattern_library=True,
                code_context="SNES hardware: $4218-$421B = joypad registers, $2100 = screen brightness",
                expected_patterns=["LDA", "STA", "RTS", "loop"],
            ),
        ],
        Difficulty.INTERMEDIATE: [
            CurriculumTemplate(
                domain=ExpertDomain.NAYRU,
                difficulty=Difficulty.INTERMEDIATE,
                instruction_templates=[
                    "Write a 65816 routine to {task}\n\nRelevant ALTTP addresses:\n{context}",
                    "Implement this ALTTP functionality:\n{task}\n\nHardware context:\n{context}",
                ],
                context_hints=["link position", "sprite spawn", "tilemap", "sfx"],
                required_entities=["link_x_coord", "link_y_coord"],
                min_output_length=80,
                quality_threshold=0.6,
                use_pattern_library=True,
                expected_patterns=["$7E00", "sprite", "position"],
            ),
        ],
        Difficulty.ADVANCED: [
            CurriculumTemplate(
                domain=ExpertDomain.NAYRU,
                difficulty=Difficulty.ADVANCED,
                instruction_templates=[
                    "Write a complete 65816 routine for {task}\n\nReference:\n{context}",
                    "Implement this SNES feature:\n{task}\n\nHardware details:\n{context}",
                ],
                context_hints=["collision", "dma chain", "interrupt", "palette"],
                min_output_length=120,
                quality_threshold=0.65,
                use_pattern_library=True,
                expected_patterns=["DMA", "NMI", "RTI", "$420B"],
            ),
        ],
        Difficulty.EXPERT: [
            CurriculumTemplate(
                domain=ExpertDomain.NAYRU,
                difficulty=Difficulty.EXPERT,
                instruction_templates=[
                    "Write a complete 65816 system for {task}",
                    "Implement this complex SNES feature:\n{task}",
                ],
                context_hints=["dialog", "hdma water", "music tempo", "mosaic"],
                min_output_length=200,
                quality_threshold=0.7,
                use_pattern_library=True,
                expected_patterns=["HDMA", "dialog", "fade", "effect"],
            ),
        ],
    },
    ExpertDomain.FARORE: {
        Difficulty.BASIC: [
            CurriculumTemplate(
                domain=ExpertDomain.FARORE,
                difficulty=Difficulty.BASIC,
                instruction_templates=[
                    "Find and fix the bug in this 65816 code:\n```\n{code}\n```\n\nExpected behavior: {expected}",
                    "Debug this assembly code and explain the issue:\n```\n{code}\n```",
                ],
                context_hints=["accumulator size", "addressing", "bank", "loop"],
                min_output_length=30,
                quality_threshold=0.5,
                use_pattern_library=True,
                expected_patterns=["REP", "SEP", "fix", "corrected"],
            ),
        ],
        Difficulty.INTERMEDIATE: [
            CurriculumTemplate(
                domain=ExpertDomain.FARORE,
                difficulty=Difficulty.INTERMEDIATE,
                instruction_templates=[
                    "Debug this routine - it has a subtle bug:\n```\n{code}\n```\n\nSymptom: {symptom}",
                    "Find and fix the issue in this code:\n```\n{code}\n```",
                ],
                context_hints=["sprite coords", "register", "dma", "stack"],
                min_output_length=80,
                quality_threshold=0.6,
                use_pattern_library=True,
                expected_patterns=["PHA", "PLA", "preserve", "corruption"],
            ),
        ],
        Difficulty.ADVANCED: [
            CurriculumTemplate(
                domain=ExpertDomain.FARORE,
                difficulty=Difficulty.ADVANCED,
                instruction_templates=[
                    "Debug this timing-sensitive code:\n```\n{code}\n```\n\nIssue: {symptom}",
                    "Find the race condition in this NMI/main loop interaction:\n```\n{code}\n```",
                ],
                context_hints=["nmi race", "hdma corruption", "flicker", "memory"],
                min_output_length=120,
                quality_threshold=0.65,
                use_pattern_library=True,
                expected_patterns=["SEI", "CLI", "VBLANK", "race"],
            ),
        ],
        Difficulty.EXPERT: [
            CurriculumTemplate(
                domain=ExpertDomain.FARORE,
                difficulty=Difficulty.EXPERT,
                instruction_templates=[
                    "Analyze and fix this complex bug:\n```\n{code}\n```\n\nContext: {symptom}",
                    "Debug this save/load corruption issue:\n```\n{code}\n```",
                ],
                context_hints=["save corruption", "hw revision", "audio desync", "bank overlap"],
                min_output_length=200,
                quality_threshold=0.7,
                use_pattern_library=True,
                expected_patterns=["checksum", "validation", "SRAM"],
            ),
        ],
    },
    ExpertDomain.VERAN: {
        Difficulty.BASIC: [
            CurriculumTemplate(
                domain=ExpertDomain.VERAN,
                difficulty=Difficulty.BASIC,
                instruction_templates=[
                    "Explain what this 65816 assembly code does:\n```\n{code}\n```",
                    "Describe the purpose and behavior of this code:\n```\n{code}\n```",
                ],
                context_hints=["lda sta", "register", "branch", "addressing"],
                min_output_length=50,
                quality_threshold=0.5,
                use_pattern_library=True,
                expected_patterns=["accumulator", "register", "memory", "address"],
            ),
        ],
        Difficulty.INTERMEDIATE: [
            CurriculumTemplate(
                domain=ExpertDomain.VERAN,
                difficulty=Difficulty.INTERMEDIATE,
                instruction_templates=[
                    "Explain this SNES hardware interaction code:\n```\n{code}\n```",
                    "Describe how this routine works step by step:\n```\n{code}\n```",
                ],
                context_hints=["dma vram", "sprite render", "mode switch", "input"],
                min_output_length=100,
                quality_threshold=0.6,
                use_pattern_library=True,
                expected_patterns=["DMA", "VRAM", "PPU", "transfer"],
            ),
        ],
        Difficulty.ADVANCED: [
            CurriculumTemplate(
                domain=ExpertDomain.VERAN,
                difficulty=Difficulty.ADVANCED,
                instruction_templates=[
                    "Analyze this HDMA/interrupt code and explain its behavior:\n```\n{code}\n```",
                    "Provide a technical analysis of this routine:\n```\n{code}\n```",
                ],
                context_hints=["hdma table", "nmi timing", "decompression", "banking"],
                min_output_length=150,
                quality_threshold=0.65,
                use_pattern_library=True,
                expected_patterns=["HDMA", "scanline", "interrupt", "timing"],
            ),
        ],
        Difficulty.EXPERT: [
            CurriculumTemplate(
                domain=ExpertDomain.VERAN,
                difficulty=Difficulty.EXPERT,
                instruction_templates=[
                    "Provide a complete technical analysis of this system:\n```\n{code}\n```",
                    "Explain the architecture and implementation of this code:\n```\n{code}\n```",
                ],
                context_hints=["mode7", "audio engine", "save system", "sprite management"],
                min_output_length=300,
                quality_threshold=0.7,
                use_pattern_library=True,
                expected_patterns=["Mode 7", "matrix", "rotation", "architecture"],
            ),
        ],
    },
}


@dataclass
class ProviderConfig:
    """Configuration for an API provider."""

    name: str
    model: str
    api_key: str | None = None
    requests_per_minute: int = 60
    max_concurrent: int = 5
    priority: int = 1  # Lower = higher priority


@dataclass
class ScaleConfig:
    """Configuration for large-scale generation."""

    # Target counts
    target_samples_per_difficulty: int = 2500  # 10K total across 4 difficulties
    max_retries_per_sample: int = 3

    # Provider settings
    # Note: Free tier limits are ~10 req/min for Gemini, ~20 req/min for Claude
    providers: list[ProviderConfig] = field(default_factory=lambda: [
        ProviderConfig(name="gemini", model="gemini-2.0-flash-exp", requests_per_minute=8),
        ProviderConfig(name="claude", model="claude-3-5-sonnet-20241022", requests_per_minute=15),
    ])

    # Difficulty distribution (weights for curriculum sampling)
    difficulty_weights: dict[Difficulty, float] = field(default_factory=lambda: {
        Difficulty.BASIC: 0.3,
        Difficulty.INTERMEDIATE: 0.35,
        Difficulty.ADVANCED: 0.25,
        Difficulty.EXPERT: 0.1,
    })

    # Quality control
    min_quality_score: float = 0.5
    validate_syntax: bool = True
    use_discriminator: bool = True

    # Checkpointing
    checkpoint_interval: int = 100  # Save every N samples
    checkpoint_dir: Path = field(default_factory=lambda: Path("~/.cache/afs/checkpoints").expanduser())

    # Output
    output_dir: Path = field(default_factory=lambda: Path("generated_data"))


@dataclass
class GenerationProgress:
    """Tracks generation progress for reporting and checkpointing."""

    total_generated: int = 0
    total_failed: int = 0
    by_difficulty: dict[Difficulty, int] = field(default_factory=dict)
    by_provider: dict[str, int] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    samples: list[TrainingSample] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_sample(self, sample: TrainingSample, provider: str, difficulty: Difficulty) -> None:
        """Record a successful sample."""
        self.total_generated += 1
        self.samples.append(sample)
        self.by_difficulty[difficulty] = self.by_difficulty.get(difficulty, 0) + 1
        self.by_provider[provider] = self.by_provider.get(provider, 0) + 1

    def add_failure(self, error: str) -> None:
        """Record a failed generation."""
        self.total_failed += 1
        self.errors.append(error)

    def elapsed_seconds(self) -> float:
        """Time elapsed since start."""
        return time.time() - self.start_time

    def samples_per_minute(self) -> float:
        """Current generation rate."""
        elapsed = self.elapsed_seconds()
        if elapsed < 1:
            return 0.0
        return (self.total_generated / elapsed) * 60

    def to_dict(self) -> dict[str, Any]:
        """Serialize for checkpointing."""
        return {
            "total_generated": self.total_generated,
            "total_failed": self.total_failed,
            "by_difficulty": {k.value: v for k, v in self.by_difficulty.items()},
            "by_provider": self.by_provider,
            "start_time": self.start_time,
            "errors": self.errors[-100:],  # Keep last 100 errors
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationProgress":
        """Deserialize from checkpoint."""
        progress = cls(
            total_generated=data.get("total_generated", 0),
            total_failed=data.get("total_failed", 0),
            by_difficulty={Difficulty(k): v for k, v in data.get("by_difficulty", {}).items()},
            by_provider=data.get("by_provider", {}),
            start_time=data.get("start_time", time.time()),
            errors=data.get("errors", []),
        )
        return progress

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Generated: {self.total_generated} samples ({self.samples_per_minute():.1f}/min)",
            f"Failed: {self.total_failed}",
            f"Elapsed: {self.elapsed_seconds() / 60:.1f} minutes",
            "\nBy difficulty:",
        ]
        for diff in Difficulty:
            count = self.by_difficulty.get(diff, 0)
            lines.append(f"  {diff.value}: {count}")
        lines.append("\nBy provider:")
        for provider, count in self.by_provider.items():
            lines.append(f"  {provider}: {count}")
        return "\n".join(lines)


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, requests_per_minute: int):
        self.rate = requests_per_minute / 60.0  # Requests per second
        self.tokens = requests_per_minute
        self.max_tokens = requests_per_minute
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request can be made."""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class CurriculumGenerator(BaseGenerator):
    """Generator with curriculum learning and multi-provider parallelization.

    Features:
    - Difficulty-stratified sample generation
    - Multi-provider parallel API calls
    - Automatic rate limiting per provider
    - Checkpointing for long runs
    - Progress tracking and reporting
    """

    def __init__(
        self,
        domain: ExpertDomain,
        config: ScaleConfig | None = None,
    ):
        super().__init__(name="CurriculumGenerator", domain=domain.value)
        self.expert_domain = domain
        self.config = config or ScaleConfig()
        self._clients: dict[str, "LLMClient"] = {}
        self._rate_limiters: dict[str, RateLimiter] = {}
        self._progress = GenerationProgress()

    def _get_client(self, provider: ProviderConfig) -> "LLMClient":
        """Get or create LLM client for provider."""
        if provider.name not in self._clients:
            from afs.generators.cot.client import get_client

            kwargs = {"model": provider.model}
            if provider.api_key:
                kwargs["api_key"] = provider.api_key

            self._clients[provider.name] = get_client(provider.name, **kwargs)
            self._rate_limiters[provider.name] = RateLimiter(provider.requests_per_minute)

        return self._clients[provider.name]

    def _sample_difficulty(self) -> Difficulty:
        """Sample a difficulty level based on weights."""
        weights = self.config.difficulty_weights
        difficulties = list(weights.keys())
        probs = [weights[d] for d in difficulties]
        return random.choices(difficulties, weights=probs, k=1)[0]

    def _sample_template(self, difficulty: Difficulty) -> CurriculumTemplate:
        """Sample a template for the given difficulty."""
        templates = CURRICULUM_TEMPLATES.get(self.expert_domain, {}).get(difficulty, [])
        if not templates:
            # Fallback to any template from this domain
            all_templates = []
            for diff_templates in CURRICULUM_TEMPLATES.get(self.expert_domain, {}).values():
                all_templates.extend(diff_templates)
            if not all_templates:
                raise ValueError(f"No templates found for domain {self.expert_domain}")
            return random.choice(all_templates)
        return random.choice(templates)

    def _generate_instruction(self, template: CurriculumTemplate) -> str:
        """Generate a concrete instruction from template with code injection."""
        from .template_libraries import (
            get_din_pattern,
            get_farore_bug,
            get_nayru_template,
            get_veran_example,
            get_hardware_context,
        )

        base_instruction = random.choice(template.instruction_templates)
        difficulty = template.difficulty.value

        # Inject code from pattern library if enabled
        if template.use_pattern_library:
            if template.domain == ExpertDomain.DIN and "{code}" in base_instruction:
                # Get optimization pattern (before code)
                before, after, desc = get_din_pattern(difficulty)
                if before:
                    base_instruction = base_instruction.replace("{code}", before)

            elif template.domain == ExpertDomain.FARORE and "{code}" in base_instruction:
                # Get bug pattern
                bug = get_farore_bug(difficulty)
                if bug.get("buggy"):
                    base_instruction = base_instruction.replace("{code}", bug["buggy"])
                    if "{expected}" in base_instruction:
                        base_instruction = base_instruction.replace("{expected}", bug.get("issue", "correct behavior"))
                    if "{symptom}" in base_instruction:
                        base_instruction = base_instruction.replace("{symptom}", bug.get("issue", "unexpected behavior"))

            elif template.domain == ExpertDomain.NAYRU and "{task}" in base_instruction:
                # Get generation task
                task_info = get_nayru_template(difficulty)
                if task_info.get("task"):
                    base_instruction = base_instruction.replace("{task}", task_info["task"])
                    if "{context}" in base_instruction:
                        context = get_hardware_context(template.context_hints[:2])
                        base_instruction = base_instruction.replace("{context}", context or template.code_context)

            elif template.domain == ExpertDomain.VERAN and "{code}" in base_instruction:
                # Get code to explain
                example = get_veran_example(difficulty)
                if example.get("code"):
                    base_instruction = base_instruction.replace("{code}", example["code"])

        # Use static code snippets if provided and template not replaced
        if "{code}" in base_instruction and template.code_snippets:
            code = random.choice(template.code_snippets)
            base_instruction = base_instruction.replace("{code}", code)

        # Add hardware context if provided
        if template.code_context and "{context}" not in base_instruction:
            if random.random() < 0.5:
                base_instruction = f"{template.code_context}\n\n{base_instruction}"

        # Add context hints with some probability
        if template.context_hints and random.random() < 0.2:
            hint = random.choice(template.context_hints)
            base_instruction = f"{base_instruction}\n\n(Focus on: {hint})"

        return base_instruction

    async def _generate_one_async(
        self,
        provider: ProviderConfig,
        instruction: str,
        template: CurriculumTemplate,
    ) -> TrainingSample | None:
        """Generate a single sample using async API call."""
        client = self._get_client(provider)
        rate_limiter = self._rate_limiters[provider.name]

        # Wait for rate limit
        await rate_limiter.acquire()

        try:
            # Build prompt with domain-specific system prompt
            system_prompt = self._get_system_prompt(template)

            response = client.generate(
                prompt=instruction,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=2048,
            )

            if not response or len(response) < template.min_output_length:
                return None

            sample = TrainingSample(
                sample_id=str(uuid.uuid4()),
                instruction=instruction,
                output=response,
                domain=f"{self.expert_domain.value}_{template.difficulty.value}",
                source="curriculum_generated",
                _metadata={
                    "provider": provider.name,
                    "model": provider.model,
                    "difficulty": template.difficulty.value,
                    "template_domain": template.domain.value,
                },
            )

            return sample

        except Exception as e:
            logger.warning(f"Generation failed with {provider.name}: {e}")
            return None

    def _get_system_prompt(self, template: CurriculumTemplate) -> str:
        """Get domain-specific system prompt with ASAR syntax guidance."""
        from .template_libraries import get_asar_context

        # Get ASAR syntax rules
        asar_context = get_asar_context(include_examples=False)

        prompts = {
            ExpertDomain.DIN: f"""You are Din, a 65816 assembly optimization expert for SNES/ALTTP.
Your role is to optimize assembly code for:
- Reduced cycle count
- Smaller code size
- Better VRAM/WRAM usage
- Efficient DMA and HDMA usage

CRITICAL: Output code using ASAR assembler syntax.

{asar_context}

Provide optimized assembly code in a ```asm block with clear comments explaining improvements.""",

            ExpertDomain.NAYRU: f"""You are Nayru, a 65816 assembly code generation expert for SNES/ALTTP.
Your role is to write clean, efficient assembly code that:
- Uses proper 65816 opcodes and addressing modes
- Follows SNES hardware constraints
- References correct WRAM addresses for ALTTP
- Includes clear comments

CRITICAL: Output code using ASAR assembler syntax.

{asar_context}

Generate working 65816 assembly code in a ```asm block. Start with 'lorom' and 'org $address'.""",

            ExpertDomain.FARORE: f"""You are Farore, a 65816 assembly debugging expert for SNES/ALTTP.
Your role is to:
- Identify bugs and errors in assembly code
- Explain the root cause of issues
- Provide corrected code with explanations
- Consider SNES hardware quirks

CRITICAL: Output code using ASAR assembler syntax.

{asar_context}

Analyze the code carefully and provide fixes in a ```asm block with detailed explanations.""",

            ExpertDomain.VERAN: f"""You are Veran, a 65816 assembly code analysis expert for SNES/ALTTP.
Your role is to:
- Explain what assembly code does step by step
- Describe register usage and memory access
- Identify patterns and techniques used
- Reference SNES hardware where relevant

NOTE: Code uses ASAR assembler syntax (label&$FFFF, ^label, db/dw, org).

Provide clear, educational explanations suitable for learning.""",
        }

        return prompts.get(template.domain, prompts[ExpertDomain.NAYRU])

    def generate_batch_sync(
        self,
        count: int,
        difficulty: Difficulty | None = None,
        progress_callback: Callable[[GenerationProgress], None] | None = None,
    ) -> list[TrainingSample]:
        """Generate samples synchronously (convenience wrapper)."""
        return asyncio.run(self.generate_batch_async(count, difficulty, progress_callback))

    async def generate_batch_async(
        self,
        count: int,
        difficulty: Difficulty | None = None,
        progress_callback: Callable[[GenerationProgress], None] | None = None,
    ) -> list[TrainingSample]:
        """Generate multiple samples with parallelization."""
        samples = []
        tasks = []

        # Create generation tasks
        for _ in range(count):
            diff = difficulty or self._sample_difficulty()
            template = self._sample_template(diff)
            instruction = self._generate_instruction(template)

            # Round-robin across providers
            provider_idx = len(tasks) % len(self.config.providers)
            provider = self.config.providers[provider_idx]

            task = self._generate_one_async(provider, instruction, template)
            tasks.append((task, provider, diff))

        # Process with limited concurrency
        semaphore = asyncio.Semaphore(10)  # Max 10 concurrent requests

        async def bounded_generate(task, provider, diff):
            async with semaphore:
                result = await task
                if result:
                    self._progress.add_sample(result, provider.name, diff)
                else:
                    self._progress.add_failure(f"Failed: {provider.name}")

                if progress_callback and self._progress.total_generated % 10 == 0:
                    progress_callback(self._progress)

                return result

        results = await asyncio.gather(*[
            bounded_generate(task, provider, diff)
            for task, provider, diff in tasks
        ])

        samples = [r for r in results if r is not None]
        return samples

    def generate(self) -> GenerationResult:
        """Generate samples using curriculum learning."""
        target = sum(
            int(self.config.target_samples_per_difficulty * weight)
            for weight in self.config.difficulty_weights.values()
        )

        samples = self.generate_batch_sync(target)

        return GenerationResult(
            samples=samples,
            skipped=self._progress.total_failed,
            errors=self._progress.errors,
            source_count=target,
        )

    def save_checkpoint(self, checkpoint_path: Path | None = None) -> Path:
        """Save current progress to checkpoint file."""
        if checkpoint_path is None:
            self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = (
                self.config.checkpoint_dir
                / f"{self.expert_domain.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        checkpoint = {
            "domain": self.expert_domain.value,
            "config": {
                "target_samples_per_difficulty": self.config.target_samples_per_difficulty,
            },
            "progress": self._progress.to_dict(),
            "samples": [s.to_dict() for s in self._progress.samples],
            "timestamp": datetime.now().isoformat(),
        }

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"Saved checkpoint to {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load progress from checkpoint file."""
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)

        self._progress = GenerationProgress.from_dict(checkpoint["progress"])
        self._progress.samples = [
            TrainingSample.from_dict(s) for s in checkpoint.get("samples", [])
        ]

        logger.info(f"Loaded checkpoint: {self._progress.total_generated} samples")

    def export_samples(self, output_path: Path) -> int:
        """Export generated samples to JSONL file."""
        return write_jsonl(self._progress.samples, output_path)


def create_curriculum_generator(
    domain: str | ExpertDomain,
    target_per_difficulty: int = 2500,
    providers: list[str] | None = None,
) -> CurriculumGenerator:
    """Factory function to create a curriculum generator.

    Args:
        domain: Expert domain (din, nayru, farore, veran)
        target_per_difficulty: Target samples per difficulty level
        providers: List of provider names to use (default: gemini, claude)

    Returns:
        Configured CurriculumGenerator
    """
    if isinstance(domain, str):
        domain = ExpertDomain(domain.lower())

    provider_configs = []
    provider_list = providers or ["gemini", "claude"]

    for name in provider_list:
        if name == "gemini":
            provider_configs.append(
                ProviderConfig(
                    name="gemini",
                    model="gemini-2.0-flash-exp",
                    requests_per_minute=8,  # Free tier limit
                )
            )
        elif name == "claude":
            provider_configs.append(
                ProviderConfig(
                    name="claude",
                    model="claude-3-5-sonnet-20241022",
                    requests_per_minute=15,  # Conservative rate
                )
            )
        elif name == "openai":
            provider_configs.append(
                ProviderConfig(
                    name="openai",
                    model="gpt-4o",
                    requests_per_minute=20,  # Conservative rate
                )
            )

    config = ScaleConfig(
        target_samples_per_difficulty=target_per_difficulty,
        providers=provider_configs,
    )

    return CurriculumGenerator(domain=domain, config=config)


# CLI interface
async def main():
    """CLI for curriculum-based data generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate training data for expert models")
    parser.add_argument("--domain", type=str, required=True,
                        choices=["din", "nayru", "farore", "veran"],
                        help="Expert domain to generate data for")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of samples to generate")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--providers", type=str, default="gemini",
                        help="Comma-separated list of providers (gemini,claude,openai)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint file")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Parse providers
    providers = [p.strip() for p in args.providers.split(",")]

    # Create generator
    generator = create_curriculum_generator(
        domain=args.domain,
        target_per_difficulty=args.count // 4,  # Distribute across difficulties
        providers=providers,
    )

    # Load checkpoint if specified
    if args.checkpoint:
        generator.load_checkpoint(Path(args.checkpoint))

    # Progress callback
    def progress_callback(progress: GenerationProgress):
        logger.info(f"Progress: {progress.total_generated} samples, "
                   f"{progress.samples_per_minute():.1f}/min")

    # Generate
    logger.info(f"Generating {args.count} samples for {args.domain}...")
    samples = await generator.generate_batch_async(
        count=args.count,
        progress_callback=progress_callback,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = generator.export_samples(output_path)
    logger.info(f"Saved {count} samples to {output_path}")

    # Save final checkpoint
    checkpoint = generator.save_checkpoint()
    logger.info(f"Saved checkpoint to {checkpoint}")

    # Print summary
    print("\n" + generator._progress.summary())


if __name__ == "__main__":
    asyncio.run(main())
