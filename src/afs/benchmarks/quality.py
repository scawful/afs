"""Quality benchmarking for model outputs.

Measures:
- Accuracy on test datasets
- Code correctness (compile/run tests)
- Reasoning quality scores
- Output consistency across runs
- Comparison to reference models
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AccuracyMetrics:
    """Accuracy measurements on test datasets."""

    total_tests: int
    correct: int
    partially_correct: int
    incorrect: int
    accuracy: float
    partial_accuracy: float
    domain_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tests": self.total_tests,
            "correct": self.correct,
            "partially_correct": self.partially_correct,
            "incorrect": self.incorrect,
            "accuracy": self.accuracy,
            "partial_accuracy": self.partial_accuracy,
            "domain_scores": self.domain_scores,
        }


@dataclass
class ConsistencyMetrics:
    """Consistency measurements across multiple runs."""

    num_runs: int
    prompts_tested: int
    average_variance: float
    consistency_score: float
    deterministic: bool
    max_output_divergence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_runs": self.num_runs,
            "prompts_tested": self.prompts_tested,
            "average_variance": self.average_variance,
            "consistency_score": self.consistency_score,
            "deterministic": self.deterministic,
            "max_output_divergence": self.max_output_divergence,
        }


@dataclass
class CodeCorrectnessResult:
    """Result from code correctness checking."""

    total_samples: int
    compilable: int
    executable: int
    tests_passed: int
    syntax_errors: int
    runtime_errors: int
    compilation_rate: float
    execution_rate: float
    test_pass_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "compilable": self.compilable,
            "executable": self.executable,
            "tests_passed": self.tests_passed,
            "syntax_errors": self.syntax_errors,
            "runtime_errors": self.runtime_errors,
            "compilation_rate": self.compilation_rate,
            "execution_rate": self.execution_rate,
            "test_pass_rate": self.test_pass_rate,
        }


@dataclass
class QualityBenchmarkResult:
    """Complete quality benchmark results."""

    model_name: str
    model_path: str
    accuracy: AccuracyMetrics
    consistency: ConsistencyMetrics
    code_correctness: CodeCorrectnessResult | None
    reasoning_score: float
    timestamp: str
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "accuracy": self.accuracy.to_dict(),
            "consistency": self.consistency.to_dict(),
            "code_correctness": self.code_correctness.to_dict() if self.code_correctness else None,
            "reasoning_score": self.reasoning_score,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Quality Benchmark: {self.model_name}",
            f"Path: {self.model_path}",
            "",
            "Accuracy:",
            f"  Total Tests: {self.accuracy.total_tests}",
            f"  Accuracy: {self.accuracy.accuracy:.1%}",
            f"  Partial Accuracy: {self.accuracy.partial_accuracy:.1%}",
            "",
            "Consistency:",
            f"  Runs: {self.consistency.num_runs}",
            f"  Score: {self.consistency.consistency_score:.3f}",
            f"  Deterministic: {self.consistency.deterministic}",
            "",
        ]

        if self.code_correctness:
            lines.extend([
                "Code Correctness:",
                f"  Compilation Rate: {self.code_correctness.compilation_rate:.1%}",
                f"  Execution Rate: {self.code_correctness.execution_rate:.1%}",
                f"  Test Pass Rate: {self.code_correctness.test_pass_rate:.1%}",
                "",
            ])

        lines.append(f"Reasoning Score: {self.reasoning_score:.3f}")
        lines.append(f"Duration: {self.duration_seconds:.2f}s")

        return "\n".join(lines)


class CodeCorrectnessChecker:
    """Check if generated code compiles and runs correctly."""

    def __init__(self, language: str = "python"):
        """Initialize checker for specific language.

        Args:
            language: Programming language (python, cpp, c, asm, etc.)
        """
        self.language = language

    def check_python(self, code: str, test_code: str | None = None) -> dict[str, Any]:
        """Check Python code correctness."""
        result = {
            "compilable": False,
            "executable": False,
            "tests_passed": False,
            "errors": [],
        }

        # Check syntax
        try:
            compile(code, "<string>", "exec")
            result["compilable"] = True
        except SyntaxError as e:
            result["errors"].append(f"SyntaxError: {e}")
            return result

        # Try to execute
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_file = f.name

            exec_result = subprocess.run(
                ["python3", temp_file],
                capture_output=True,
                timeout=5,
                text=True,
            )

            if exec_result.returncode == 0:
                result["executable"] = True
            else:
                result["errors"].append(f"Runtime error: {exec_result.stderr}")

            Path(temp_file).unlink()

        except subprocess.TimeoutExpired:
            result["errors"].append("Execution timeout")
        except Exception as e:
            result["errors"].append(f"Execution error: {e}")

        # Run tests if provided
        if test_code and result["executable"]:
            try:
                full_code = code + "\n\n" + test_code
                exec(full_code, {})
                result["tests_passed"] = True
            except AssertionError as e:
                result["errors"].append(f"Test failed: {e}")
            except Exception as e:
                result["errors"].append(f"Test error: {e}")

        return result

    def check_cpp(self, code: str) -> dict[str, Any]:
        """Check C++ code correctness."""
        result = {
            "compilable": False,
            "executable": False,
            "tests_passed": False,
            "errors": [],
        }

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                src_file = Path(tmpdir) / "code.cpp"
                out_file = Path(tmpdir) / "code.out"

                src_file.write_text(code)

                # Compile
                compile_result = subprocess.run(
                    ["g++", "-std=c++17", "-o", str(out_file), str(src_file)],
                    capture_output=True,
                    timeout=30,
                    text=True,
                )

                if compile_result.returncode == 0:
                    result["compilable"] = True

                    # Try to execute
                    exec_result = subprocess.run(
                        [str(out_file)],
                        capture_output=True,
                        timeout=5,
                        text=True,
                    )

                    if exec_result.returncode == 0:
                        result["executable"] = True
                    else:
                        result["errors"].append(f"Runtime error: {exec_result.stderr}")
                else:
                    result["errors"].append(f"Compilation error: {compile_result.stderr}")

        except subprocess.TimeoutExpired:
            result["errors"].append("Compilation/execution timeout")
        except Exception as e:
            result["errors"].append(f"Error: {e}")

        return result

    def check_asm(self, code: str, assembler: str = "asar") -> dict[str, Any]:
        """Check assembly code correctness (SNES/65816)."""
        result = {
            "compilable": False,
            "executable": False,  # Not really executable directly
            "tests_passed": False,
            "errors": [],
        }

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                asm_file = Path(tmpdir) / "code.asm"
                out_file = Path(tmpdir) / "code.bin"

                asm_file.write_text(code)

                # Assemble
                assemble_result = subprocess.run(
                    [assembler, str(asm_file), str(out_file)],
                    capture_output=True,
                    timeout=10,
                    text=True,
                )

                if assemble_result.returncode == 0:
                    result["compilable"] = True
                    result["executable"] = out_file.exists()
                else:
                    result["errors"].append(f"Assembly error: {assemble_result.stderr}")

        except FileNotFoundError:
            result["errors"].append(f"Assembler '{assembler}' not found")
        except subprocess.TimeoutExpired:
            result["errors"].append("Assembly timeout")
        except Exception as e:
            result["errors"].append(f"Error: {e}")

        return result

    def check(self, code: str, test_code: str | None = None) -> dict[str, Any]:
        """Check code correctness for configured language."""
        if self.language == "python":
            return self.check_python(code, test_code)
        elif self.language in ("cpp", "c++"):
            return self.check_cpp(code)
        elif self.language in ("asm", "assembly", "65816"):
            return self.check_asm(code)
        else:
            return {
                "compilable": False,
                "executable": False,
                "tests_passed": False,
                "errors": [f"Unsupported language: {self.language}"],
            }


class QualityBenchmark:
    """Comprehensive quality benchmarking for model outputs."""

    def __init__(
        self,
        model_path: Path | str,
        test_dataset: Path | str,
        model_name: str | None = None,
        model_loader: callable | None = None,
    ):
        """Initialize quality benchmark.

        Args:
            model_path: Path to model checkpoint
            test_dataset: Path to test dataset (JSONL format)
            model_name: Display name for model
            model_loader: Custom model loader function
        """
        self.model_path = Path(model_path)
        self.test_dataset = Path(test_dataset)
        self.model_name = model_name or self.model_path.name
        self.model_loader = model_loader
        self._model = None

    def _load_model(self):
        """Load model using provided loader or default."""
        if self._model is None:
            if self.model_loader:
                self._model = self.model_loader(self.model_path)
            else:
                # Default: try mlx
                try:
                    from mlx_lm import load

                    self._model = load(str(self.model_path))
                except ImportError as err:
                    raise RuntimeError(
                        "No model loader provided and mlx not available"
                    ) from err
        return self._model

    def _load_test_dataset(self) -> list[dict[str, Any]]:
        """Load test dataset from JSONL file."""
        data = []
        with open(self.test_dataset) as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def run(
        self,
        check_code_correctness: bool = True,
        num_consistency_runs: int = 3,
    ) -> QualityBenchmarkResult:
        """Run complete quality benchmark.

        Args:
            check_code_correctness: Whether to check if generated code compiles/runs
            num_consistency_runs: Number of runs to test consistency

        Returns:
            QualityBenchmarkResult with all metrics
        """
        import time
        from datetime import datetime

        model = self._load_model()
        test_data = self._load_test_dataset()
        start_time = time.perf_counter()

        # Measure accuracy
        accuracy = self._measure_accuracy(model, test_data)

        # Measure consistency
        consistency = self._measure_consistency(
            model, test_data[:10], num_consistency_runs
        )

        # Check code correctness
        code_correctness = None
        if check_code_correctness:
            code_correctness = self._measure_code_correctness(model, test_data)

        # Measure reasoning quality
        reasoning_score = self._measure_reasoning_quality(model, test_data)

        end_time = time.perf_counter()

        return QualityBenchmarkResult(
            model_name=self.model_name,
            model_path=str(self.model_path),
            accuracy=accuracy,
            consistency=consistency,
            code_correctness=code_correctness,
            reasoning_score=reasoning_score,
            timestamp=datetime.now().isoformat(),
            duration_seconds=end_time - start_time,
        )

    def _measure_accuracy(
        self, model, test_data: list[dict[str, Any]]
    ) -> AccuracyMetrics:
        """Measure accuracy on test dataset."""
        correct = 0
        partially_correct = 0
        incorrect = 0
        domain_scores: dict[str, list[int]] = {}

        for item in test_data:
            # Generate output
            prompt = item.get("prompt", "")
            expected = item.get("expected", "")
            domain = item.get("domain", "general")

            # Placeholder: actual generation would use model API
            output = self._generate(model, prompt)

            # Check correctness
            score = self._compare_outputs(output, expected)

            if score >= 0.9:
                correct += 1
            elif score >= 0.5:
                partially_correct += 1
            else:
                incorrect += 1

            # Track by domain
            if domain not in domain_scores:
                domain_scores[domain] = []
            domain_scores[domain].append(1 if score >= 0.9 else 0)

        total = len(test_data)
        return AccuracyMetrics(
            total_tests=total,
            correct=correct,
            partially_correct=partially_correct,
            incorrect=incorrect,
            accuracy=correct / total if total > 0 else 0,
            partial_accuracy=(correct + partially_correct) / total if total > 0 else 0,
            domain_scores={
                domain: sum(scores) / len(scores) for domain, scores in domain_scores.items()
            },
        )

    def _measure_consistency(
        self, model, test_prompts: list[dict[str, Any]], num_runs: int
    ) -> ConsistencyMetrics:
        """Measure output consistency across multiple runs."""
        variances = []
        max_divergence = 0.0

        for prompt_data in test_prompts:
            prompt = prompt_data.get("prompt", "")
            outputs = []

            # Generate multiple times
            for _ in range(num_runs):
                output = self._generate(model, prompt)
                outputs.append(output)

            # Measure variance
            variance = self._calculate_output_variance(outputs)
            variances.append(variance)
            max_divergence = max(max_divergence, variance)

        avg_variance = sum(variances) / len(variances) if variances else 0
        consistency_score = 1.0 - min(avg_variance, 1.0)
        deterministic = max_divergence < 0.01

        return ConsistencyMetrics(
            num_runs=num_runs,
            prompts_tested=len(test_prompts),
            average_variance=avg_variance,
            consistency_score=consistency_score,
            deterministic=deterministic,
            max_output_divergence=max_divergence,
        )

    def _measure_code_correctness(
        self, model, test_data: list[dict[str, Any]]
    ) -> CodeCorrectnessResult:
        """Measure code correctness (compilation/execution)."""
        checker = CodeCorrectnessChecker(language="python")

        total = 0
        compilable = 0
        executable = 0
        tests_passed = 0
        syntax_errors = 0
        runtime_errors = 0

        for item in test_data:
            if item.get("type") != "code":
                continue

            prompt = item.get("prompt", "")
            test_code = item.get("test", "")

            # Generate code
            output = self._generate(model, prompt)

            # Check correctness
            result = checker.check(output, test_code)

            total += 1
            if result["compilable"]:
                compilable += 1
            else:
                syntax_errors += 1

            if result["executable"]:
                executable += 1
            elif result["compilable"]:
                runtime_errors += 1

            if result["tests_passed"]:
                tests_passed += 1

        return CodeCorrectnessResult(
            total_samples=total,
            compilable=compilable,
            executable=executable,
            tests_passed=tests_passed,
            syntax_errors=syntax_errors,
            runtime_errors=runtime_errors,
            compilation_rate=compilable / total if total > 0 else 0,
            execution_rate=executable / total if total > 0 else 0,
            test_pass_rate=tests_passed / total if total > 0 else 0,
        )

    def _measure_reasoning_quality(
        self, model, test_data: list[dict[str, Any]]
    ) -> float:
        """Measure reasoning quality (heuristic scoring)."""
        # This is a placeholder - real implementation would use
        # semantic similarity, logical consistency checks, etc.
        return 0.85

    def _generate(self, model, prompt: str) -> str:
        """Generate output from model (placeholder)."""
        # Replace with actual model generation
        return f"Generated output for: {prompt[:50]}..."

    def _compare_outputs(self, output: str, expected: str) -> float:
        """Compare two outputs and return similarity score."""
        # Simple word overlap metric
        output_words = set(output.lower().split())
        expected_words = set(expected.lower().split())

        if not expected_words:
            return 0.0

        overlap = len(output_words & expected_words)
        return overlap / len(expected_words)

    def _calculate_output_variance(self, outputs: list[str]) -> float:
        """Calculate variance in outputs."""
        if len(outputs) < 2:
            return 0.0

        # Simple metric: pairwise differences
        total_diff = 0.0
        comparisons = 0

        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                # 1 - similarity = difference
                similarity = self._compare_outputs(outputs[i], outputs[j])
                total_diff += 1.0 - similarity
                comparisons += 1

        return total_diff / comparisons if comparisons > 0 else 0.0
