"""Chain of Thought Generator for training data."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ..base import (
    BaseGenerator,
    GenerationResult,
    TrainingSample,
    clean_instruction,
    is_malformed_output,
    read_jsonl,
)
from .client import LLMClient, get_client
from .formats import CotFormat, format_cot_sample
from .prompts import ASM_COT_SYSTEM_PROMPT, build_analysis_prompt
from .rate_limiter import RateLimiter, batch_items

if TYPE_CHECKING:
    pass


@dataclass
class CotConfig:
    """Configuration for Chain of Thought generation."""

    # LLM settings
    api_provider: str = "gemini"  # gemini, claude, openai
    model_name: str = "gemini-2.0-flash-exp"
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: str | None = None

    # Output format
    cot_format: CotFormat = CotFormat.SEPARATE

    # Rate limiting
    requests_per_minute: int = 60
    batch_size: int = 10
    retry_count: int = 3
    retry_delay: float = 1.0

    # Processing
    skip_existing_thinking: bool = True
    domains: tuple[str, ...] = (
        "asm",
        "asm_optimize",
        "asm_debug",
        "asm_hook",
    )
    min_output_length: int = 50  # Skip samples with very short outputs

    # Analysis focus areas
    analysis_focus: list[str] = field(
        default_factory=lambda: [
            "instruction_purpose",
            "memory_addressing",
            "register_usage",
            "cycle_optimization",
            "snes_hardware",
        ]
    )


class CotGenerator(BaseGenerator):
    """Generator that adds Chain of Thought reasoning to samples.

    Uses LLM APIs (Gemini, Claude, OpenAI) to generate step-by-step
    reasoning for assembly code samples.

    Example:
        ```python
        config = CotConfig(
            api_provider="gemini",
            cot_format=CotFormat.SEPARATE,
        )
        generator = CotGenerator(
            input_path=Path("samples.jsonl"),
            config=config,
        )
        result = generator.generate()
        ```
    """

    def __init__(
        self,
        input_path: Path,
        config: CotConfig | None = None,
        client: LLMClient | None = None,
    ):
        super().__init__(name="CotGenerator", domain="asm-cot")
        self.input_path = Path(input_path)
        self.config = config or CotConfig()
        self.client = client or self._create_client()
        self.rate_limiter = RateLimiter(self.config.requests_per_minute)

    def _create_client(self) -> LLMClient:
        """Create LLM client based on config."""
        return get_client(
            self.config.api_provider,
            model=self.config.model_name,
        )

    def generate(self) -> GenerationResult:
        """Generate CoT-enhanced training samples."""
        result = GenerationResult()

        # Load source samples
        source_samples = read_jsonl(self.input_path)
        result.source_count = len(source_samples)

        print(f"Loaded {len(source_samples)} samples from {self.input_path}")

        # Filter samples
        filtered_samples = self._filter_samples(source_samples)
        print(f"After filtering: {len(filtered_samples)} samples")

        # Process in batches
        batches = batch_items(filtered_samples, self.config.batch_size)
        total_batches = len(batches)

        for batch_idx, batch in enumerate(batches):
            print(f"Processing batch {batch_idx + 1}/{total_batches}...")

            for sample in batch:
                try:
                    cot_sample = self._process_sample(sample)
                    if cot_sample:
                        result.samples.append(cot_sample)
                except Exception as e:
                    error_msg = f"Error processing {sample.sample_id}: {e}"
                    result.errors.append(error_msg)
                    print(f"  {error_msg}")
                    result.skipped += 1

            # Progress update
            processed = (batch_idx + 1) * self.config.batch_size
            print(f"  Processed {min(processed, len(filtered_samples))}/{len(filtered_samples)}")

        return result

    def _filter_samples(self, samples: list[TrainingSample]) -> list[TrainingSample]:
        """Filter samples based on config criteria."""
        filtered = []

        for sample in samples:
            # Check domain
            if sample.domain not in self.config.domains:
                continue

            # Check output length
            if len(sample.output) < self.config.min_output_length:
                continue

            # Skip if already has thinking and config says so
            if self.config.skip_existing_thinking and sample.thinking:
                continue

            # Skip samples with malformed outputs (file markers as summaries)
            if is_malformed_output(sample.output, sample.instruction):
                continue

            # Clean the instruction before processing
            cleaned_instruction, was_cleaned = clean_instruction(sample.instruction)

            # Skip if instruction is empty after cleaning
            if not cleaned_instruction.strip():
                continue

            # Update the sample with cleaned instruction
            if was_cleaned:
                sample.instruction = cleaned_instruction
                sample._metadata["instruction_cleaned"] = True

            filtered.append(sample)

        return filtered

    def _process_sample(self, sample: TrainingSample) -> TrainingSample | None:
        """Process a single sample to add CoT reasoning."""
        # Rate limit
        self.rate_limiter.acquire_sync()

        # Build prompt
        prompt = build_analysis_prompt(
            instruction=sample.instruction,
            output=sample.output,
            input_text=sample.input,
            domain=sample.domain,
        )

        # Generate thinking
        system_prompt = self.config.system_prompt or ASM_COT_SYSTEM_PROMPT

        thinking = self.client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        # Clean up thinking
        thinking = self._clean_thinking(thinking)

        if not thinking:
            return None

        # Format sample with CoT
        cot_sample = format_cot_sample(
            sample=sample,
            thinking=thinking,
            format_type=self.config.cot_format,
        )

        # Update metadata
        cot_sample._metadata["cot_model"] = self.config.model_name
        cot_sample._metadata["cot_provider"] = self.config.api_provider
        cot_sample._metadata["cot_timestamp"] = datetime.now().isoformat()

        return cot_sample

    def _clean_thinking(self, thinking: str) -> str:
        """Clean up generated thinking text."""
        # Remove common LLM artifacts
        thinking = thinking.strip()

        # Remove markdown headers if they're just repeating the prompt
        lines = thinking.split("\n")
        cleaned_lines = []

        for line in lines:
            # Skip lines that are just restating the task
            if line.startswith("## Task") or line.startswith("## Instruction"):
                continue
            if line.startswith("## Assembly Code"):
                continue
            if line.startswith("---"):
                continue
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()


def generate_cot_for_file(
    input_path: Path,
    output_path: Path | None = None,
    config: CotConfig | None = None,
    limit: int | None = None,
) -> GenerationResult:
    """Convenience function to generate CoT for a file.

    Args:
        input_path: Path to input JSONL file
        output_path: Path for output (default: input_cot.jsonl)
        config: CotConfig (default: use defaults)
        limit: Maximum samples to process (for testing)

    Returns:
        GenerationResult with generated samples
    """
    from ..base import write_jsonl

    generator = CotGenerator(input_path=input_path, config=config)
    result = generator.generate()

    if limit and len(result.samples) > limit:
        result.samples = result.samples[:limit]

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_cot.jsonl"

    write_jsonl(result.samples, output_path)
    print(f"Wrote {len(result.samples)} samples to {output_path}")

    return result
