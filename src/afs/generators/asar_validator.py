"""Asar (SNES assembler) validation pipeline for training data.

This module provides validation of assembly code in training samples
by attempting to assemble them with the asar SNES assembler.

Usage:
    python -m afs generators validate --input samples.jsonl \
        --output-pass valid.jsonl --output-fail invalid.jsonl

The validator:
1. Reads JSONL training samples
2. Extracts assembly code from the output field
3. Creates temp .asm files with necessary context
4. Runs asar to validate compilation
5. Splits samples into pass/fail output files
6. Reports validation statistics
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import TrainingSample, read_jsonl, write_jsonl


@dataclass
class ValidationStats:
    """Statistics from validation run."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Percentage of samples that passed validation."""
        validated = self.passed + self.failed
        if validated == 0:
            return 0.0
        return self.passed / validated

    @property
    def skip_rate(self) -> float:
        """Percentage of samples that were skipped."""
        if self.total == 0:
            return 0.0
        return self.skipped / self.total

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"Total samples: {self.total}",
            f"Passed: {self.passed} ({self.pass_rate:.1%})",
            f"Failed: {self.failed}",
            f"Skipped: {self.skipped} ({self.skip_rate:.1%})",
        ]
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
        return "\n".join(lines)


@dataclass
class ValidationResult:
    """Result of validating a single sample."""

    sample: TrainingSample
    passed: bool
    error_message: str = ""
    warnings: list[str] = field(default_factory=list)
    asar_output: str = ""

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


@dataclass
class AsarValidatorConfig:
    """Configuration for asar validation."""

    # Path to asar executable (None = search PATH)
    asar_path: str | None = None

    # ROM mapping mode for asar
    # lorom, hirom, exlorom, exhirom, sa1rom, sfxrom, norom
    mapping: str = "lorom"

    # Optional base ROM to patch (copied into temp workspace)
    base_rom_path: Path | None = None

    # Include paths for asar
    include_paths: list[Path] = field(default_factory=list)

    # Whether to include ALTTP-specific context/headers
    include_alttp_context: bool = True

    # Minimum output length to validate (skip short snippets)
    min_output_length: int = 10

    # Skip samples with these domains (e.g., non-assembly domains)
    skip_domains: list[str] = field(default_factory=lambda: ["text", "docs"])

    # Patterns that indicate code is incomplete/partial
    # Note: These are checked with re.MULTILINE so ^ matches line starts
    skip_patterns: list[str] = field(
        default_factory=lambda: [
            r"\.\.\.",  # Ellipsis indicating truncation
            r"\[omitted\]",  # Explicitly omitted
        ]
    )

    # Check if output is ONLY comments (no actual code)
    comment_only_check: bool = True

    # Timeout for asar execution (seconds)
    timeout: float = 10.0

    # Keep temp files for debugging
    keep_temp_files: bool = False


# Path to ALTTP stubs file
ALTTP_STUBS_PATH = Path(__file__).parent.parent / "knowledge" / "alttp_stubs.asm"


def get_alttp_header() -> str:
    """Load ALTTP header from stubs file."""
    if ALTTP_STUBS_PATH.exists():
        return ALTTP_STUBS_PATH.read_text()
    # Fallback minimal header if stubs file not found
    return """\
; ALTTP Assembly Validation Header (minimal fallback)
lorom
warnings disable W1018

org $008000

"""


# Cached header content
_ALTTP_HEADER: str | None = None


def get_cached_alttp_header() -> str:
    """Get cached ALTTP header."""
    global _ALTTP_HEADER
    if _ALTTP_HEADER is None:
        _ALTTP_HEADER = get_alttp_header()
    return _ALTTP_HEADER

# Minimal header for non-ALTTP code
MINIMAL_HEADER = """\
; Minimal asar validation header

lorom
warnings disable W1018

org $008000

"""


class AsarValidator:
    """Validates assembly code using the asar SNES assembler."""

    def __init__(self, config: AsarValidatorConfig | None = None):
        self.config = config or AsarValidatorConfig()
        self._asar_path: str | None = None
        self._temp_dir: Path | None = None

    @property
    def asar_path(self) -> str:
        """Get path to asar executable."""
        if self._asar_path is None:
            self._asar_path = self._find_asar()
        return self._asar_path

    def _find_asar(self) -> str:
        """Find the asar executable."""
        if self.config.asar_path:
            return self.config.asar_path

        # Check common names
        for name in ["asar", "asar.exe", "asar-cli"]:
            path = shutil.which(name)
            if path:
                return path

        raise FileNotFoundError(
            "asar executable not found. Install asar or set asar_path in config."
        )

    def _get_temp_dir(self) -> Path:
        """Get or create temp directory for validation files."""
        if self._temp_dir is None or not self._temp_dir.exists():
            self._temp_dir = Path(tempfile.mkdtemp(prefix="asar_validate_"))
        return self._temp_dir

    def _cleanup_temp_dir(self) -> None:
        """Clean up temp directory."""
        if self._temp_dir and self._temp_dir.exists() and not self.config.keep_temp_files:
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None

    def _should_skip(self, sample: TrainingSample) -> tuple[bool, str]:
        """Check if sample should be skipped."""
        # Check domain
        if sample.domain in self.config.skip_domains:
            return True, f"domain '{sample.domain}' in skip list"

        # Check output length
        output = sample.output.strip()
        if len(output) < self.config.min_output_length:
            return True, f"output too short ({len(output)} chars)"

        # Check skip patterns
        for pattern in self.config.skip_patterns:
            if re.search(pattern, output, re.IGNORECASE | re.MULTILINE):
                return True, f"matches skip pattern '{pattern}'"

        # Check if output is ONLY comments (no actual assembly instructions)
        if self.config.comment_only_check:
            # Remove all comment lines and see if anything remains
            non_comment_lines = []
            for line in output.split("\n"):
                stripped = line.strip()
                # Skip empty lines and pure comment lines
                if stripped and not stripped.startswith(";"):
                    # Also skip lines that are just separators (=== or ---)
                    if not re.match(r"^[=\-]+$", stripped):
                        non_comment_lines.append(stripped)
            if not non_comment_lines:
                return True, "output contains only comments"

        return False, ""

    def _prepare_asm_content(self, sample: TrainingSample) -> str:
        """Prepare assembly content for validation.

        Wraps the sample output with appropriate headers and context.
        """
        output = sample.output.strip()

        # Detect if code already has ORG directive
        has_org = bool(re.search(r"^\s*org\s+\$", output, re.IGNORECASE | re.MULTILINE))

        # Choose header based on config and content
        if self.config.include_alttp_context:
            header = get_cached_alttp_header()
        else:
            header = MINIMAL_HEADER

        if self.config.mapping:
            header = self._apply_mapping(header)

        # If code has its own ORG, use minimal header without ORG
        if has_org:
            header = re.sub(r"org \$[0-9A-Fa-f]+\s*\n?", "", header)

        # Combine header with sample code
        content = header + "\n; === Sample Code ===\n" + output + "\n"

        # Add footer marker
        content += "\n; === End Sample ===\n"

        return content

    def _apply_mapping(self, header: str) -> str:
        """Ensure the header contains the configured mapping directive."""
        mapping = (self.config.mapping or "").strip()
        if not mapping:
            return header

        pattern = r"^(lorom|hirom|exlorom|exhirom|sa1rom|sfxrom|norom)\s*$"
        if re.search(pattern, header, flags=re.IGNORECASE | re.MULTILINE):
            return re.sub(pattern, mapping, header, flags=re.IGNORECASE | re.MULTILINE)

        lines = header.splitlines()
        insert_at = 0
        while insert_at < len(lines) and lines[insert_at].strip().startswith(";"):
            insert_at += 1
        lines.insert(insert_at, mapping)
        return "\n".join(lines) + ("\n" if header.endswith("\n") else "")

    def _extract_asm_blocks(self, text: str) -> list[str]:
        """Extract assembly code blocks from text.

        Handles markdown code blocks and raw assembly.
        """
        blocks = []

        # Try to extract markdown code blocks
        # Match ```asm, ```65816, ```assembly, or plain ```
        pattern = r"```(?:asm|65816|assembly|snes|s)?\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

        if matches:
            blocks.extend(matches)
        else:
            # No code blocks found, treat entire text as assembly
            blocks.append(text)

        return blocks

    def validate_sample(self, sample: TrainingSample) -> ValidationResult:
        """Validate a single training sample.

        Returns ValidationResult with pass/fail status and details.
        """
        # Check if should skip
        should_skip, reason = self._should_skip(sample)
        if should_skip:
            return ValidationResult(
                sample=sample,
                passed=False,
                error_message=f"Skipped: {reason}",
            )

        # Extract assembly blocks
        blocks = self._extract_asm_blocks(sample.output)
        if not blocks:
            return ValidationResult(
                sample=sample,
                passed=False,
                error_message="No assembly code found in output",
            )

        # Validate each block (for now, just use the first/main block)
        main_block = blocks[0]
        asm_content = self._prepare_asm_content(
            TrainingSample(
                instruction=sample.instruction,
                output=main_block,
                input=sample.input,
            )
        )

        # Write to temp file
        temp_dir = self._get_temp_dir()
        asm_file = temp_dir / f"{sample.sample_id}.asm"
        rom_file = temp_dir / f"{sample.sample_id}.sfc"

        try:
            if self.config.base_rom_path:
                shutil.copy(self.config.base_rom_path, rom_file)
            asm_file.write_text(asm_content, encoding="utf-8")
        except Exception as e:
            return ValidationResult(
                sample=sample,
                passed=False,
                error_message=f"Failed to write temp file: {e}",
            )

        # Run asar
        try:
            result = self._run_asar(asm_file, rom_file)
            return ValidationResult(
                sample=sample,
                passed=result["success"],
                error_message=result.get("error", ""),
                warnings=result.get("warnings", []),
                asar_output=result.get("output", ""),
            )
        except Exception as e:
            return ValidationResult(
                sample=sample,
                passed=False,
                error_message=f"asar execution failed: {e}",
            )
        finally:
            # Clean up individual files if not keeping
            if not self.config.keep_temp_files:
                if asm_file.exists():
                    asm_file.unlink()
                if rom_file.exists():
                    rom_file.unlink()

    def _run_asar(self, asm_file: Path, rom_file: Path) -> dict[str, Any]:
        """Run asar on an assembly file.

        Returns dict with:
            success: bool
            output: str (stdout/stderr)
            error: str (error message if failed)
            warnings: list[str]
        """
        cmd = [self.asar_path]

        # Add include paths
        for inc_path in self.config.include_paths:
            cmd.extend(["-I", str(inc_path)])

        # Add input and output files
        cmd.append(str(asm_file))
        cmd.append(str(rom_file))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
            )
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"asar timed out after {self.config.timeout}s",
                "warnings": [],
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "warnings": [],
            }

        output = result.stdout + result.stderr
        warnings = []
        errors = []

        # Parse asar output
        for line in output.splitlines():
            line_lower = line.lower()
            if "warning:" in line_lower:
                warnings.append(line.strip())
            elif "error:" in line_lower:
                errors.append(line.strip())

        # asar returns 0 on success, non-zero on error
        success = result.returncode == 0 and len(errors) == 0

        return {
            "success": success,
            "output": output,
            "error": "; ".join(errors) if errors else "",
            "warnings": warnings,
        }

    def validate_file(
        self,
        input_path: Path,
        output_pass_path: Path | None = None,
        output_fail_path: Path | None = None,
        workers: int | None = None,
    ) -> tuple[ValidationStats, list[TrainingSample], list[TrainingSample]]:
        """Validate all samples in a JSONL file.

        Args:
            input_path: Path to input JSONL file
            output_pass_path: Path to write passing samples (optional)
            output_fail_path: Path to write failing samples (optional)
            workers: Number of parallel workers (optional)

        Returns:
            Tuple of (stats, passed_samples, failed_samples)
        """
        samples = read_jsonl(input_path)
        stats = ValidationStats(total=len(samples))

        passed_samples: list[TrainingSample] = []
        failed_samples: list[TrainingSample] = []

        worker_count = max(1, workers or 1)
        if worker_count > 1:
            self._get_temp_dir()
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(self.validate_sample, sample): (i, sample)
                    for i, sample in enumerate(samples)
                }
                for future in as_completed(future_map):
                    i, sample = future_map[future]
                    try:
                        result = future.result()

                        if "Skipped:" in result.error_message:
                            stats.skipped += 1
                            failed_samples.append(sample)
                        elif result.passed:
                            stats.passed += 1
                            passed_samples.append(sample)
                        else:
                            stats.failed += 1
                            sample._metadata["validation_error"] = result.error_message
                            sample._metadata["asar_output"] = result.asar_output
                            failed_samples.append(sample)
                    except Exception as e:
                        stats.errors.append(f"Sample {i}: {e}")
                        failed_samples.append(sample)
        else:
            for i, sample in enumerate(samples):
                try:
                    result = self.validate_sample(sample)

                    if "Skipped:" in result.error_message:
                        stats.skipped += 1
                        # Skipped samples go to failed output
                        failed_samples.append(sample)
                    elif result.passed:
                        stats.passed += 1
                        passed_samples.append(sample)
                    else:
                        stats.failed += 1
                        # Store error info in metadata
                        sample._metadata["validation_error"] = result.error_message
                        sample._metadata["asar_output"] = result.asar_output
                        failed_samples.append(sample)

                except Exception as e:
                    stats.errors.append(f"Sample {i}: {e}")
                    failed_samples.append(sample)

        # Write output files
        if output_pass_path:
            write_jsonl(passed_samples, output_pass_path)
        if output_fail_path:
            write_jsonl(failed_samples, output_fail_path)

        # Clean up
        self._cleanup_temp_dir()

        return stats, passed_samples, failed_samples

    def validate_string(self, asm_code: str) -> ValidationResult:
        """Validate a raw assembly code string.

        Convenience method for validating assembly without a TrainingSample.
        """
        sample = TrainingSample(
            instruction="(direct validation)",
            output=asm_code,
        )
        return self.validate_sample(sample)


def check_asar_available() -> bool:
    """Check if asar is available on the system."""
    for name in ["asar", "asar.exe", "asar-cli"]:
        if shutil.which(name):
            return True
    return False


def validate_training_data(
    input_path: Path,
    output_pass_path: Path,
    output_fail_path: Path,
    config: AsarValidatorConfig | None = None,
    verbose: bool = False,
    workers: int | None = None,
) -> ValidationStats:
    """Validate training data and split into pass/fail files.

    Main entry point for CLI usage.

    Args:
        input_path: Path to input JSONL file
        output_pass_path: Path to write passing samples
        output_fail_path: Path to write failing samples
        config: Validator configuration
        verbose: Print progress information
        workers: Number of parallel workers

    Returns:
        ValidationStats with results
    """
    if not check_asar_available() and not (config and config.asar_path):
        raise FileNotFoundError(
            "asar not found. Install asar SNES assembler and ensure it's in PATH."
        )

    validator = AsarValidator(config)

    if verbose:
        print(f"Validating: {input_path}")
        print(f"Pass output: {output_pass_path}")
        print(f"Fail output: {output_fail_path}")

    stats, passed, failed = validator.validate_file(
        input_path=input_path,
        output_pass_path=output_pass_path,
        output_fail_path=output_fail_path,
        workers=workers,
    )

    if verbose:
        print()
        print(stats.summary())

    return stats
