"""Semantic evaluation of assembly code via emulator execution.

Uses the yaze emulator to validate that generated assembly code:
1. Compiles successfully with asar
2. Executes without crashing
3. Produces expected register/memory state changes
4. Matches semantic expectations from the instruction
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from afs.generators.base import TrainingSample

logger = logging.getLogger(__name__)


# Default paths (can be overridden via environment)
ASAR_PATH = os.environ.get(
    "AFS_ASAR_PATH",
    "/Users/scawful/src/third_party/asar-repo/build/asar/bin/asar"
)
DUMMY_ROM_PATH = os.environ.get(
    "AFS_ASAR_ROM",
    "/Users/scawful/src/training/roms/dummy.sfc"
)
YAZE_GRPC_TARGET = os.environ.get("YAZE_GRPC_TARGET", "127.0.0.1:50051")


@dataclass
class CPUState:
    """Represents 65816 CPU state after execution."""

    a: int = 0       # Accumulator (16-bit)
    x: int = 0       # X register (16-bit)
    y: int = 0       # Y register (16-bit)
    sp: int = 0      # Stack pointer
    pc: int = 0      # Program counter
    db: int = 0      # Data bank
    pb: int = 0      # Program bank
    d: int = 0       # Direct page
    status: int = 0  # Status register

    # Flags
    flag_n: bool = False  # Negative
    flag_v: bool = False  # Overflow
    flag_z: bool = False  # Zero
    flag_c: bool = False  # Carry

    cycles: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": self.a, "x": self.x, "y": self.y,
            "sp": self.sp, "pc": self.pc,
            "db": self.db, "pb": self.pb, "d": self.d,
            "status": self.status,
            "flags": {
                "n": self.flag_n, "v": self.flag_v,
                "z": self.flag_z, "c": self.flag_c,
            },
            "cycles": self.cycles,
        }

    def format(self) -> str:
        return (
            f"PC: ${self.pb:02X}:{self.pc:04X} | "
            f"A: ${self.a:04X} X: ${self.x:04X} Y: ${self.y:04X} | "
            f"SP: ${self.sp:04X} DB: ${self.db:02X} D: ${self.d:04X} | "
            f"Flags: {'N' if self.flag_n else '-'}{'V' if self.flag_v else '-'}"
            f"{'Z' if self.flag_z else '-'}{'C' if self.flag_c else '-'}"
        )

    def diff(self, other: CPUState) -> dict[str, tuple[Any, Any]]:
        """Return differences between two CPU states."""
        changes = {}
        for field_name in ["a", "x", "y", "sp", "pc", "db", "pb", "d", "status"]:
            old_val = getattr(self, field_name)
            new_val = getattr(other, field_name)
            if old_val != new_val:
                changes[field_name] = (old_val, new_val)

        for flag in ["n", "v", "z", "c"]:
            old_flag = getattr(self, f"flag_{flag}")
            new_flag = getattr(other, f"flag_{flag}")
            if old_flag != new_flag:
                changes[f"flag_{flag}"] = (old_flag, new_flag)

        return changes


@dataclass
class ExecutionResult:
    """Result of executing assembly code in the emulator."""

    # Compilation
    compiled: bool = False
    compile_errors: list[str] = field(default_factory=list)
    compile_warnings: list[str] = field(default_factory=list)
    bytecode_size: int = 0

    # Execution
    executed: bool = False
    crashed: bool = False
    execution_error: str = ""
    frame_count: int = 0

    # State
    initial_state: CPUState | None = None
    final_state: CPUState | None = None
    state_changes: dict[str, tuple[Any, Any]] = field(default_factory=dict)

    # Memory changes
    memory_writes: list[tuple[int, int, int]] = field(default_factory=list)  # (addr, old, new)

    def to_dict(self) -> dict[str, Any]:
        return {
            "compiled": self.compiled,
            "compile_errors": self.compile_errors,
            "compile_warnings": self.compile_warnings,
            "bytecode_size": self.bytecode_size,
            "executed": self.executed,
            "crashed": self.crashed,
            "execution_error": self.execution_error,
            "frame_count": self.frame_count,
            "initial_state": self.initial_state.to_dict() if self.initial_state else None,
            "final_state": self.final_state.to_dict() if self.final_state else None,
            "state_changes": self.state_changes,
            "memory_writes": self.memory_writes,
        }


@dataclass
class SemanticScore:
    """Semantic evaluation score for a sample."""

    # Component scores (0.0 - 1.0)
    compile_score: float = 0.0      # Did it compile?
    execute_score: float = 0.0       # Did it run without crashing?
    behavior_score: float = 0.0      # Did it produce expected behavior?
    efficiency_score: float = 0.0    # Cycle count / code size efficiency

    # Overall semantic score
    overall: float = 0.0

    # Weights for combining scores
    weights: dict[str, float] = field(default_factory=lambda: {
        "compile": 0.3,
        "execute": 0.3,
        "behavior": 0.3,
        "efficiency": 0.1,
    })

    def compute_overall(self) -> float:
        """Compute weighted overall score."""
        self.overall = (
            self.weights["compile"] * self.compile_score +
            self.weights["execute"] * self.execute_score +
            self.weights["behavior"] * self.behavior_score +
            self.weights["efficiency"] * self.efficiency_score
        )
        return self.overall

    def to_dict(self) -> dict[str, Any]:
        return {
            "compile_score": self.compile_score,
            "execute_score": self.execute_score,
            "behavior_score": self.behavior_score,
            "efficiency_score": self.efficiency_score,
            "overall": self.overall,
        }


@dataclass
class SemanticEvalConfig:
    """Configuration for semantic evaluation."""

    # ASAR settings
    asar_path: Path = field(default_factory=lambda: Path(ASAR_PATH))
    base_rom_path: Path = field(default_factory=lambda: Path(DUMMY_ROM_PATH))
    rom_type: str = "lorom"

    # Execution settings
    injection_address: int = 0x7F0000  # WRAM by default
    frame_count: int = 60
    timeout: float = 10.0

    # gRPC settings
    grpc_target: str = field(default_factory=lambda: YAZE_GRPC_TARGET)

    # Scoring thresholds
    min_bytecode_size: int = 4
    max_bytecode_size: int = 4096
    max_cycles: int = 10000


class SemanticEvaluator:
    """Evaluates assembly code semantics via emulator execution.

    This evaluator:
    1. Compiles assembly code using asar
    2. Injects bytecode into the emulator
    3. Executes for N frames
    4. Captures before/after CPU and memory state
    5. Computes semantic correctness scores
    """

    def __init__(self, config: SemanticEvalConfig | None = None):
        self.config = config or SemanticEvalConfig()
        self._grpc_client = None

    @property
    def grpc_client(self):
        """Lazy-load gRPC client for yaze emulator."""
        if self._grpc_client is None:
            try:
                # Try to import the yaze-mcp client
                import sys
                sys.path.insert(0, "/Users/scawful/src/tools/yaze-mcp")
                from core.emulator_client import EmulatorClient, SymbolResolver

                symbols_path = os.environ.get(
                    "YAZE_SYMBOLS_PATH",
                    str(Path.home() / ".context/knowledge/alttp/symbols.json")
                )
                resolver = SymbolResolver(symbols_path)
                self._grpc_client = EmulatorClient(
                    target=self.config.grpc_target,
                    resolver=resolver,
                )
            except ImportError as e:
                logger.warning(f"Could not import yaze-mcp client: {e}")
                self._grpc_client = None
            except Exception as e:
                logger.warning(f"Could not connect to yaze emulator: {e}")
                self._grpc_client = None

        return self._grpc_client

    def compile_code(self, code: str) -> tuple[bool, bytes, list[str], list[str]]:
        """Compile assembly code using asar.

        Args:
            code: 65816 assembly code

        Returns:
            Tuple of (success, bytecode, errors, warnings)
        """
        if not self.config.asar_path.exists():
            return False, b"", [f"asar not found: {self.config.asar_path}"], []

        # Wrap code with ROM type directive
        wrapped_code = f"{self.config.rom_type}\n" + code

        with tempfile.TemporaryDirectory() as tmpdir:
            asm_path = Path(tmpdir) / "test.asm"
            rom_path = Path(tmpdir) / "test.sfc"

            # Write assembly file
            asm_path.write_text(wrapped_code)

            # Copy base ROM if exists
            if self.config.base_rom_path.exists():
                rom_path.write_bytes(self.config.base_rom_path.read_bytes())
            else:
                # Create minimal ROM
                rom_path.write_bytes(b"\x00" * 0x8000)

            # Run asar
            try:
                result = subprocess.run(
                    [str(self.config.asar_path), "--no-title-check", str(asm_path), str(rom_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout,
                )
            except subprocess.TimeoutExpired:
                return False, b"", ["asar timed out"], []

            # Parse output
            errors = []
            warnings = []
            for line in result.stdout.split("\n") + result.stderr.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if "error:" in line.lower():
                    errors.append(line)
                elif "warning:" in line.lower():
                    warnings.append(line)

            if result.returncode != 0 or errors:
                return False, b"", errors, warnings

            # Read assembled bytecode
            bytecode = rom_path.read_bytes()
            return True, bytecode, [], warnings

    def execute_code(
        self,
        bytecode: bytes,
        address: int | None = None,
        frames: int | None = None,
    ) -> ExecutionResult:
        """Execute bytecode in the emulator.

        Args:
            bytecode: Assembled machine code
            address: Injection address (default: config.injection_address)
            frames: Number of frames to run (default: config.frame_count)

        Returns:
            ExecutionResult with state information
        """
        result = ExecutionResult()
        result.bytecode_size = len(bytecode)

        if self.grpc_client is None:
            result.execution_error = "Emulator not connected"
            return result

        address = address or self.config.injection_address
        frames = frames or self.config.frame_count

        try:
            # Get initial state
            success, initial = self.grpc_client.get_cpu_state()
            if success and initial:
                result.initial_state = CPUState(
                    a=initial.a, x=initial.x, y=initial.y,
                    sp=initial.sp, pc=initial.pc,
                    db=initial.db, pb=initial.pb, d=initial.d,
                    status=initial.status,
                    flag_n=initial.flag_n, flag_v=initial.flag_v,
                    flag_z=initial.flag_z, flag_c=initial.flag_c,
                    cycles=initial.cycles,
                )

            # Execute
            success, message, final_cpu, crashed = self.grpc_client.test_run(
                address=address,
                data=bytecode,
                frame_count=frames,
                reset_after=True,
            )

            result.executed = success
            result.crashed = crashed
            result.frame_count = frames

            if not success:
                result.execution_error = message
                return result

            # Capture final state
            if final_cpu:
                result.final_state = CPUState(
                    a=final_cpu.a, x=final_cpu.x, y=final_cpu.y,
                    sp=final_cpu.sp, pc=final_cpu.pc,
                    db=final_cpu.db, pb=final_cpu.pb, d=final_cpu.d,
                    status=final_cpu.status,
                    flag_n=final_cpu.flag_n, flag_v=final_cpu.flag_v,
                    flag_z=final_cpu.flag_z, flag_c=final_cpu.flag_c,
                    cycles=final_cpu.cycles,
                )

                # Compute state changes
                if result.initial_state:
                    result.state_changes = result.initial_state.diff(result.final_state)

        except Exception as e:
            result.execution_error = str(e)
            logger.exception("Execution failed")

        return result

    def evaluate_sample(
        self,
        sample: TrainingSample,
        expected_behavior: dict[str, Any] | None = None,
    ) -> tuple[SemanticScore, ExecutionResult]:
        """Evaluate a training sample semantically.

        Args:
            sample: TrainingSample with assembly code in output
            expected_behavior: Optional dict specifying expected state changes

        Returns:
            Tuple of (SemanticScore, ExecutionResult)
        """
        score = SemanticScore()
        code = sample.output

        # Step 1: Compile
        compiled, bytecode, errors, warnings = self.compile_code(code)

        if not compiled:
            result = ExecutionResult(
                compiled=False,
                compile_errors=errors,
                compile_warnings=warnings,
            )
            score.compile_score = 0.0
            score.compute_overall()
            return score, result

        score.compile_score = 1.0

        # Step 2: Execute
        result = self.execute_code(bytecode)
        result.compiled = True
        result.compile_warnings = warnings

        if not result.executed:
            score.execute_score = 0.0
            score.compute_overall()
            return score, result

        # Penalize crashes
        if result.crashed:
            score.execute_score = 0.2
        else:
            score.execute_score = 1.0

        # Step 3: Evaluate behavior
        score.behavior_score = self._evaluate_behavior(
            sample, result, expected_behavior
        )

        # Step 4: Evaluate efficiency
        score.efficiency_score = self._evaluate_efficiency(result)

        score.compute_overall()
        return score, result

    def _evaluate_behavior(
        self,
        sample: TrainingSample,
        result: ExecutionResult,
        expected: dict[str, Any] | None,
    ) -> float:
        """Evaluate if the code produced expected behavior."""
        if not result.state_changes:
            # No state changes might be valid for some code
            return 0.5

        # If we have explicit expectations, check them
        if expected:
            matches = 0
            total = len(expected)

            for key, expected_value in expected.items():
                if key in result.state_changes:
                    _, actual = result.state_changes[key]
                    if actual == expected_value:
                        matches += 1

            return matches / total if total > 0 else 0.5

        # Otherwise, heuristic scoring based on instruction intent
        instruction_lower = sample.instruction.lower()

        # Check for register modification expectations
        if any(word in instruction_lower for word in ["load", "lda", "set"]):
            if "a" in result.state_changes:
                return 0.8

        if any(word in instruction_lower for word in ["clear", "zero", "reset"]):
            if result.final_state and result.final_state.a == 0:
                return 0.9

        if any(word in instruction_lower for word in ["increment", "inc", "add"]):
            if "a" in result.state_changes or "x" in result.state_changes or "y" in result.state_changes:
                return 0.8

        # Default: some changes is better than none
        return 0.6 if result.state_changes else 0.3

    def _evaluate_efficiency(self, result: ExecutionResult) -> float:
        """Evaluate code efficiency (size and cycles)."""
        if not result.executed:
            return 0.0

        # Size score: smaller is better (within reasonable bounds)
        size = result.bytecode_size
        if size < self.config.min_bytecode_size:
            size_score = 0.3  # Suspiciously small
        elif size > self.config.max_bytecode_size:
            size_score = 0.2  # Too large
        else:
            # Linear interpolation: optimal around 50-200 bytes
            if size <= 200:
                size_score = 1.0
            else:
                size_score = max(0.3, 1.0 - (size - 200) / 1000)

        # Cycle score
        if result.final_state:
            cycles = result.final_state.cycles
            if cycles == 0:
                cycle_score = 0.5  # Didn't run
            elif cycles > self.config.max_cycles:
                cycle_score = 0.2  # Too slow / infinite loop
            else:
                cycle_score = 1.0 - (cycles / self.config.max_cycles) * 0.3
        else:
            cycle_score = 0.5

        return (size_score + cycle_score) / 2

    def batch_evaluate(
        self,
        samples: list[TrainingSample],
        skip_execution: bool = False,
    ) -> list[tuple[SemanticScore, ExecutionResult]]:
        """Evaluate multiple samples.

        Args:
            samples: List of training samples
            skip_execution: If True, only check compilation (faster)

        Returns:
            List of (SemanticScore, ExecutionResult) tuples
        """
        results = []

        for sample in samples:
            if skip_execution:
                # Compile-only evaluation
                compiled, bytecode, errors, warnings = self.compile_code(sample.output)
                score = SemanticScore(
                    compile_score=1.0 if compiled else 0.0,
                    execute_score=0.5,  # Unknown
                    behavior_score=0.5,
                    efficiency_score=0.5,
                )
                score.compute_overall()
                result = ExecutionResult(
                    compiled=compiled,
                    compile_errors=errors,
                    compile_warnings=warnings,
                    bytecode_size=len(bytecode) if compiled else 0,
                )
                results.append((score, result))
            else:
                results.append(self.evaluate_sample(sample))

        return results


def create_semantic_evaluator(
    asar_path: str | None = None,
    base_rom_path: str | None = None,
    grpc_target: str | None = None,
) -> SemanticEvaluator:
    """Factory function to create a semantic evaluator.

    Args:
        asar_path: Path to asar executable
        base_rom_path: Path to base ROM for compilation
        grpc_target: gRPC target for yaze emulator

    Returns:
        Configured SemanticEvaluator
    """
    config = SemanticEvalConfig()

    if asar_path:
        config.asar_path = Path(asar_path)
    if base_rom_path:
        config.base_rom_path = Path(base_rom_path)
    if grpc_target:
        config.grpc_target = grpc_target

    return SemanticEvaluator(config)
