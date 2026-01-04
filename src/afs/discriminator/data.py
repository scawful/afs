"""Data utilities for ASM-ELECTRA training."""

from dataclasses import dataclass, field
from pathlib import Path
import json
import random
from typing import Iterator

from .fake_generators import CompositeGenerator, FakeGenerator


@dataclass
class ElectraSample:
    """A single training sample for ELECTRA."""
    text: str
    label: int  # 0 = real, 1 = fake
    source: str = ""
    error_type: str = ""


@dataclass
class ElectraDataset:
    """Dataset for ELECTRA training."""
    samples: list[ElectraSample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[ElectraSample]:
        return iter(self.samples)

    def add(self, sample: ElectraSample) -> None:
        self.samples.append(sample)

    def shuffle(self) -> None:
        random.shuffle(self.samples)

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> tuple["ElectraDataset", "ElectraDataset", "ElectraDataset"]:
        """Split into train/val/test sets."""
        n = len(self.samples)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        self.shuffle()

        train = ElectraDataset(samples=self.samples[:train_end])
        val = ElectraDataset(samples=self.samples[train_end:val_end])
        test = ElectraDataset(samples=self.samples[val_end:])

        return train, val, test

    def balance(self) -> "ElectraDataset":
        """Balance real/fake samples by undersampling majority class."""
        real = [s for s in self.samples if s.label == 0]
        fake = [s for s in self.samples if s.label == 1]

        min_count = min(len(real), len(fake))

        balanced = ElectraDataset(
            samples=random.sample(real, min_count) + random.sample(fake, min_count)
        )
        balanced.shuffle()
        return balanced

    def stats(self) -> dict:
        """Get dataset statistics."""
        real_count = sum(1 for s in self.samples if s.label == 0)
        fake_count = len(self.samples) - real_count

        return {
            "total": len(self.samples),
            "real": real_count,
            "fake": fake_count,
            "ratio": fake_count / real_count if real_count > 0 else 0,
        }

    def to_jsonl(self, path: Path) -> None:
        """Save dataset to JSONL format."""
        with open(path, "w") as f:
            for sample in self.samples:
                f.write(json.dumps({
                    "text": sample.text,
                    "label": sample.label,
                    "source": sample.source,
                    "error_type": sample.error_type,
                }) + "\n")

    @classmethod
    def from_jsonl(cls, path: Path) -> "ElectraDataset":
        """Load dataset from JSONL format."""
        samples = []
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                samples.append(ElectraSample(
                    text=data["text"],
                    label=data["label"],
                    source=data.get("source", ""),
                    error_type=data.get("error_type", ""),
                ))
        return cls(samples=samples)

    def to_hf_format(self) -> list[dict]:
        """Convert to HuggingFace datasets format."""
        return [
            {"text": s.text, "label": s.label}
            for s in self.samples
        ]


def extract_code_blocks(text: str) -> list[str]:
    """Extract assembly code blocks from text."""
    blocks = []

    # Try markdown code blocks first
    import re
    pattern = r"```(?:asm|assembly|65816)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        blocks.extend(matches)

    # Also try to find raw assembly (lines starting with opcodes or labels)
    lines = text.split("\n")
    current_block = []

    for line in lines:
        stripped = line.strip()
        # Check if line looks like assembly
        if (
            stripped
            and not stripped.startswith("#")
            and not stripped.startswith("//")
            and (
                stripped[0].isupper()  # Opcode
                or stripped.endswith(":")  # Label
                or stripped.startswith(".")  # Directive
                or stripped.startswith(";")  # Comment
            )
        ):
            current_block.append(line)
        elif current_block:
            if len(current_block) >= 3:  # Minimum 3 lines
                blocks.append("\n".join(current_block))
            current_block = []

    if current_block and len(current_block) >= 3:
        blocks.append("\n".join(current_block))

    return blocks


def create_training_data(
    real_sources: list[Path],
    fake_ratio: float = 0.5,
    generator: FakeGenerator | None = None,
    min_lines: int = 3,
    max_lines: int = 50,
) -> ElectraDataset:
    """Create ELECTRA training data from real assembly sources.

    Args:
        real_sources: Paths to files/directories containing real assembly
        fake_ratio: Proportion of fake samples (0.5 = balanced)
        generator: FakeGenerator to use (default: CompositeGenerator)
        min_lines: Minimum lines per sample
        max_lines: Maximum lines per sample

    Returns:
        ElectraDataset with real and fake samples
    """
    if generator is None:
        generator = CompositeGenerator()

    dataset = ElectraDataset()

    # Collect real code samples
    real_samples = []
    for source in real_sources:
        source = Path(source)
        if source.is_file():
            files = [source]
        else:
            # Recursively find assembly files
            files = list(source.rglob("*.asm")) + list(source.rglob("*.s"))

        for file in files:
            try:
                content = file.read_text(errors="ignore")
                blocks = extract_code_blocks(content)

                # Also treat whole file as potential source
                if not blocks:
                    blocks = [content]

                for block in blocks:
                    lines = [l for l in block.split("\n") if l.strip()]
                    if min_lines <= len(lines) <= max_lines:
                        real_samples.append((block, str(file)))
            except Exception:
                continue

    print(f"Collected {len(real_samples)} real code samples")

    # Add real samples
    for code, source in real_samples:
        dataset.add(ElectraSample(
            text=code,
            label=0,
            source=source,
            error_type="",
        ))

    # Generate fake samples
    target_fake = int(len(real_samples) * fake_ratio / (1 - fake_ratio))
    fake_count = 0

    # Shuffle and generate fakes
    random.shuffle(real_samples)
    for code, source in real_samples:
        if fake_count >= target_fake:
            break

        result = generator.generate(code)
        if result:
            dataset.add(ElectraSample(
                text=result.modified,
                label=1,
                source=source,
                error_type=result.error_type,
            ))
            fake_count += 1

    print(f"Generated {fake_count} fake samples")
    dataset.shuffle()

    return dataset


def load_from_training_jsonl(
    path: Path,
    field: str = "output",
    generator: FakeGenerator | None = None,
    fake_ratio: float = 0.5,
) -> ElectraDataset:
    """Create ELECTRA data from existing training JSONL.

    Extracts assembly code from the specified field of training samples.
    """
    if generator is None:
        generator = CompositeGenerator()

    dataset = ElectraDataset()
    samples = []

    with open(path) as f:
        for line in f:
            data = json.loads(line)
            code = data.get(field, "")
            if code:
                # Extract code blocks if in markdown format
                blocks = extract_code_blocks(code)
                if blocks:
                    samples.extend(blocks)
                elif len(code.split("\n")) >= 3:
                    samples.append(code)

    print(f"Extracted {len(samples)} code samples from {path}")

    # Add real and fake
    for code in samples:
        # Real
        dataset.add(ElectraSample(text=code, label=0, source=str(path)))

        # Maybe fake
        if random.random() < fake_ratio:
            result = generator.generate(code)
            if result:
                dataset.add(ElectraSample(
                    text=result.modified,
                    label=1,
                    source=str(path),
                    error_type=result.error_type,
                ))

    dataset.shuffle()
    return dataset
