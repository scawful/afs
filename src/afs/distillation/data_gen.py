"""Distillation data generation with multi-provider support.

Generates high-quality training data from teacher ensemble with
quality filtering and checkpointing.
"""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from ..history import log_event
from .teacher import TeacherEnsemble, TeacherResponse

logger = logging.getLogger(__name__)


@dataclass
class DistillationSample:
    """A single distillation training sample."""
    id: str
    prompt: str
    response: str
    system_prompt: str | None = None
    provider: str = ""
    model: str = ""
    quality_score: float = 0.0
    domain: str = ""
    difficulty: str = ""
    metadata: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    def to_training_format(self, format_type: str = "alpaca") -> dict:
        """Convert to training format."""
        if format_type == "alpaca":
            return {
                "instruction": self.prompt,
                "input": "",
                "output": self.response,
            }
        elif format_type == "chatml":
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": self.prompt})
            messages.append({"role": "assistant", "content": self.response})
            return {"messages": messages}
        elif format_type == "sharegpt":
            conversations = []
            if self.system_prompt:
                conversations.append({"from": "system", "value": self.system_prompt})
            conversations.append({"from": "human", "value": self.prompt})
            conversations.append({"from": "gpt", "value": self.response})
            return {"conversations": conversations}
        else:
            raise ValueError(f"Unknown format: {format_type}")


@dataclass
class GenerationProgress:
    """Progress tracking for data generation."""
    total_requested: int
    generated: int = 0
    failed: int = 0
    quality_filtered: int = 0
    start_time: float = field(default_factory=time.time)
    provider_counts: dict = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        total = self.generated + self.failed
        return self.generated / total if total > 0 else 0

    @property
    def samples_per_minute(self) -> float:
        elapsed = time.time() - self.start_time
        return (self.generated / elapsed) * 60 if elapsed > 0 else 0

    def record(self, response: TeacherResponse, passed_quality: bool) -> None:
        if response.success:
            if passed_quality:
                self.generated += 1
                provider = response.provider.value
                self.provider_counts[provider] = self.provider_counts.get(provider, 0) + 1
            else:
                self.quality_filtered += 1
        else:
            self.failed += 1


@dataclass
class DistillationConfig:
    """Configuration for data generation."""
    target_count: int = 1000
    min_quality_score: float = 0.5
    checkpoint_interval: int = 100
    output_dir: Path = Path("distillation_data")
    domains: list[str] = field(default_factory=lambda: ["din", "nayru", "farore", "veran"])
    format_type: str = "chatml"


class DistillationDataGenerator:
    """Generates distillation training data from teacher ensemble."""

    def __init__(
        self,
        ensemble: TeacherEnsemble | None = None,
        config: DistillationConfig | None = None,
        quality_scorer: Callable[[str, str], float] | None = None,
    ):
        self.ensemble = ensemble or TeacherEnsemble.default_ensemble()
        self.config = config or DistillationConfig()
        self.quality_scorer = quality_scorer or self._default_quality_scorer
        self._samples: list[DistillationSample] = []
        self._checkpoint_path: Path | None = None

    def _default_quality_scorer(self, prompt: str, response: str) -> float:
        """Default quality scoring based on response characteristics."""
        score = 0.0

        # Length check (not too short, not too long)
        length = len(response)
        if length < 50:
            return 0.0  # Too short
        elif 50 <= length <= 500:
            score += 0.3
        elif 500 < length <= 2000:
            score += 0.4
        else:
            score += 0.2  # Too long

        # Check for code blocks (good for assembly tasks)
        if "```" in response or any(op in response for op in ["LDA", "STA", "JSR", "RTS"]):
            score += 0.3

        # Check for explanation patterns
        if any(word in response.lower() for word in ["because", "this", "the", "will"]):
            score += 0.2

        # Penalize error messages
        if any(err in response.lower() for err in ["error", "sorry", "cannot", "i can't"]):
            score -= 0.3

        return max(0.0, min(1.0, score))

    def _generate_prompt(self, domain: str, difficulty: str) -> tuple[str, str]:
        """Generate a prompt for the given domain and difficulty."""
        # Domain-specific prompt templates
        prompts = {
            "din": {
                "basic": [
                    "Optimize this 65816 assembly code for fewer cycles:\n```\nLDA #$00\nSTA $10\nLDA #$00\nSTA $11\n```",
                    "How can I reduce the byte count of this SNES code:\n```\nLDX #$00\nloop:\nINX\nCPX #$10\nBNE loop\n```",
                ],
                "intermediate": [
                    "Optimize this DMA setup for ALTTP:\n```\nLDA #$01\nSTA $420B\nLDA #$00\nSTA $4300\n```",
                ],
                "advanced": [
                    "Optimize this tile decompression routine for SNES, considering both cycles and size.",
                ],
            },
            "nayru": {
                "basic": [
                    "Write 65816 assembly to store the value $42 at address $7E0100",
                    "Generate SNES code to copy 16 bytes from $7E0000 to $7E0100",
                ],
                "intermediate": [
                    "Write code to update Link's X coordinate at $7E0022 by adding 4 pixels",
                ],
                "advanced": [
                    "Generate a routine to spawn a new sprite in ALTTP using the OAM system",
                ],
            },
            "farore": {
                "basic": [
                    "Find the bug in this code:\n```\nREP #$20\nLDA #$1234\nSEP #$20\nSTA $10\n```",
                ],
                "intermediate": [
                    "Debug this DMA transfer that isn't working:\n```\nLDA #$80\nSTA $2115\nLDA $4300\n```",
                ],
                "advanced": [
                    "Diagnose why this interrupt handler causes crashes in ALTTP",
                ],
            },
            "veran": {
                "basic": [
                    "Explain what the REP #$20 instruction does in 65816 assembly",
                    "What does JSR (Jump to Subroutine) do and how does it use the stack?",
                ],
                "intermediate": [
                    "Explain how SNES DMA works and what the $43xx registers control",
                ],
                "advanced": [
                    "Describe the ALTTP sprite animation system and how it manages OAM",
                ],
            },
        }

        system_prompts = {
            "din": "You are Din, an expert at optimizing 65816 assembly code for the SNES. Focus on reducing cycles and bytes while maintaining correctness.",
            "nayru": "You are Nayru, an expert at generating 65816 assembly code for SNES games. Write clean, well-commented code.",
            "farore": "You are Farore, an expert at debugging 65816 assembly code. Identify bugs, explain their causes, and provide fixes.",
            "veran": "You are Veran, an expert at explaining 65816 assembly concepts. Provide clear, educational explanations.",
        }

        import random
        domain_prompts = prompts.get(domain, prompts["din"])
        difficulty_prompts = domain_prompts.get(difficulty, domain_prompts.get("basic", []))
        prompt = random.choice(difficulty_prompts) if difficulty_prompts else "Write some 65816 assembly code."
        system = system_prompts.get(domain, "You are an expert 65816 assembly programmer.")

        return prompt, system

    async def generate_batch(
        self,
        count: int | None = None,
        progress_callback: Callable[[GenerationProgress], None] | None = None,
    ) -> list[DistillationSample]:
        """Generate a batch of distillation samples."""
        count = count or self.config.target_count
        progress = GenerationProgress(total_requested=count)

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_path = self.config.output_dir / "checkpoint.jsonl"

        # Load existing checkpoint
        if self._checkpoint_path.exists():
            self._load_checkpoint()
            progress.generated = len(self._samples)
            logger.info(f"Resumed from checkpoint with {progress.generated} samples")

        difficulties = ["basic", "intermediate", "advanced", "expert"]
        sample_id = len(self._samples)

        while progress.generated < count:
            # Generate prompt
            import random
            domain = random.choice(self.config.domains)
            difficulty = random.choice(difficulties)
            prompt, system_prompt = self._generate_prompt(domain, difficulty)

            # Generate response from ensemble
            response = await self.ensemble.generate(prompt, system_prompt)

            if response.success:
                quality = self.quality_scorer(prompt, response.content)
                log_event(
                    "model",
                    "afs.distillation",
                    op="teacher_response",
                    metadata={
                        "provider": response.provider.value,
                        "model": response.model,
                        "domain": domain,
                        "difficulty": difficulty,
                        "quality_score": quality,
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.completion_tokens,
                        "latency_ms": response.latency_ms,
                    },
                    payload={
                        "prompt": prompt,
                        "system": system_prompt,
                        "response": response.content,
                    },
                )

                if quality >= self.config.min_quality_score:
                    sample = DistillationSample(
                        id=f"distill_{sample_id:06d}",
                        prompt=prompt,
                        response=response.content,
                        system_prompt=system_prompt,
                        provider=response.provider.value,
                        model=response.model,
                        quality_score=quality,
                        domain=domain,
                        difficulty=difficulty,
                        metadata={
                            "latency_ms": response.latency_ms,
                            "prompt_tokens": response.prompt_tokens,
                            "completion_tokens": response.completion_tokens,
                        }
                    )
                    self._samples.append(sample)
                    sample_id += 1

                    progress.record(response, passed_quality=True)
                else:
                    progress.record(response, passed_quality=False)
            else:
                progress.record(response, passed_quality=False)

            # Checkpoint
            if progress.generated % self.config.checkpoint_interval == 0:
                self._save_checkpoint()

            # Progress callback
            if progress_callback and progress.generated % 10 == 0:
                progress_callback(progress)

        # Final save
        self._save_checkpoint()

        return self._samples

    def _save_checkpoint(self) -> None:
        """Save current progress to checkpoint file."""
        if not self._checkpoint_path:
            return

        with open(self._checkpoint_path, "w") as f:
            for sample in self._samples:
                f.write(json.dumps(sample.to_dict()) + "\n")

        logger.info(f"Checkpoint saved: {len(self._samples)} samples")

    def _load_checkpoint(self) -> None:
        """Load samples from checkpoint file."""
        if not self._checkpoint_path or not self._checkpoint_path.exists():
            return

        self._samples = []
        with open(self._checkpoint_path) as f:
            for line in f:
                data = json.loads(line.strip())
                self._samples.append(DistillationSample(**data))

    def export_training_data(
        self,
        output_path: Path,
        format_type: str | None = None,
    ) -> int:
        """Export samples to training format."""
        format_type = format_type or self.config.format_type
        output_path = Path(output_path)

        with open(output_path, "w") as f:
            for sample in self._samples:
                f.write(json.dumps(sample.to_training_format(format_type)) + "\n")

        logger.info(f"Exported {len(self._samples)} samples to {output_path}")
        return len(self._samples)

    def get_statistics(self) -> dict:
        """Get generation statistics."""
        if not self._samples:
            return {}

        provider_counts = {}
        domain_counts = {}
        quality_scores = []

        for sample in self._samples:
            provider_counts[sample.provider] = provider_counts.get(sample.provider, 0) + 1
            domain_counts[sample.domain] = domain_counts.get(sample.domain, 0) + 1
            quality_scores.append(sample.quality_score)

        return {
            "total_samples": len(self._samples),
            "providers": provider_counts,
            "domains": domain_counts,
            "avg_quality": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "min_quality": min(quality_scores) if quality_scores else 0,
            "max_quality": max(quality_scores) if quality_scores else 0,
        }
