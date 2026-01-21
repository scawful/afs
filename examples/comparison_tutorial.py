#!/usr/bin/env python3
"""Tutorial: Using the Model Comparison Framework

This example demonstrates basic usage of the comparison framework for
evaluating multiple model versions.
"""

import json
import sys
from pathlib import Path

# Add afs to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from afs.comparison import (
    ModelComparator,
    ComparisonMode,
    BasicScorer,
    ResponseScorer,
    ScoredResponse,
    ModelResponse,
)


class SimpleModelMock:
    """Mock model for demonstration (replaces real model in examples)."""

    def __init__(self, name: str, quality_multiplier: float = 1.0):
        self.name = name
        self.quality_multiplier = quality_multiplier

    def generate(self, prompt: str, **kwargs) -> str:
        """Simulate model generation."""
        # Simulate different quality levels
        responses = {
            "Create a simple loop": f"LDX #$00\nLOOP:\nINX\nCMPX #$10\nBNE LOOP\nRTS",
            "Handle interrupts": f"SEI\n...\nCLI\nRTS",
            "Implement function": f"JSR MyFunc\nBNE Error\nBRA Success",
        }
        return responses.get(prompt, "Generated code")


def example_1_basic_comparison():
    """Example 1: Head-to-head comparison between two models."""
    print("=" * 70)
    print("EXAMPLE 1: Head-to-Head Comparison")
    print("=" * 70)

    # Create comparator
    comparator = ModelComparator(
        comparison_mode=ComparisonMode.HEAD_TO_HEAD,
        scorer=BasicScorer()
    )

    # Load two models
    model_v5 = SimpleModelMock("v5", quality_multiplier=0.85)
    model_v6 = SimpleModelMock("v6", quality_multiplier=0.92)

    comparator.load_model("v5", lambda: model_v5)
    comparator.load_model("v6", lambda: model_v6)

    # Define prompts
    prompts = [
        "Create a simple loop",
        "Handle interrupts",
        "Implement function",
    ]

    # Run comparison
    print(f"\nComparing {len(comparator.models)} models on {len(prompts)} prompts...")
    report = comparator.run_prompts(prompts)

    # Print results
    print("\n" + "-" * 70)
    print("RESULTS")
    print("-" * 70)

    ranked = report.get_ranked_models()
    for rank, (model_name, stats) in enumerate(ranked, 1):
        print(f"\n{rank}. {model_name}")
        print(f"   Mean Score: {stats['mean_overall_score']:.3f}")
        print(f"   Std Dev: {stats['stdev_overall_score']:.3f}")
        print(f"   Win Rate: {stats['win_rate']:.1%}")
        print(f"   Avg Latency: {stats['mean_latency_ms']:.0f}ms")

    # Generate markdown report
    md_report = comparator.generate_markdown_report()
    print("\n" + "-" * 70)
    print("Markdown Report Preview:")
    print("-" * 70)
    print(md_report[:500] + "...")


def example_2_tournament():
    """Example 2: Tournament mode ranking 4 models."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Tournament Mode (4 Models)")
    print("=" * 70)

    # Create comparator
    comparator = ModelComparator(
        comparison_mode=ComparisonMode.TOURNAMENT,
        scorer=BasicScorer()
    )

    # Load 4 models with varying quality
    models = {
        "v5": SimpleModelMock("v5", quality_multiplier=0.80),
        "v6": SimpleModelMock("v6", quality_multiplier=0.88),
        "v7": SimpleModelMock("v7", quality_multiplier=0.94),
        "v8": SimpleModelMock("v8", quality_multiplier=0.85),
    }

    for model_name, model in models.items():
        comparator.load_model(model_name, lambda m=model: m)
        print(f"Loaded: {model_name}")

    # Define prompts
    prompts = [
        "Create a simple loop",
        "Handle interrupts",
        "Implement function",
        "Manage memory",
        "Optimize speed",
    ]

    # Run tournament
    print(f"\nRunning tournament on {len(prompts)} prompts...")
    report = comparator.run_prompts(prompts)

    # Print rankings
    print("\n" + "-" * 70)
    print("TOURNAMENT STANDINGS")
    print("-" * 70)

    ranked = report.get_ranked_models()
    for rank, (model_name, stats) in enumerate(ranked, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{medal} {rank}. {model_name:8} Score: {stats['mean_overall_score']:.3f} "
              f"Wins: {stats['win_count']}/{stats['total_comparisons']}")


def example_3_custom_scorer():
    """Example 3: Using a custom scorer for domain-specific evaluation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Custom Scorer")
    print("=" * 70)

    class AssemblyQualityScorer(ResponseScorer):
        """Custom scorer for assembly code quality."""

        def score(self, response: ModelResponse, reference=None) -> ScoredResponse:
            scored = ScoredResponse(model_response=response)

            # Check for assembly-specific keywords
            asm_keywords = ["LDX", "STX", "JSR", "RTS", "SEI", "CLI", "BNE"]
            keyword_count = sum(1 for kw in asm_keywords if kw in response.response)

            # Correctness: based on presence of assembly keywords
            scored.correctness_score = min(keyword_count / 3.0, 1.0)

            # Completeness: based on response length
            scored.completeness_score = min(len(response.response) / 200.0, 1.0)

            # Clarity: based on newlines (well-structured code)
            line_count = response.response.count("\n")
            scored.clarity_score = min(line_count / 5.0, 1.0)

            # Efficiency: inverse of tokens
            scored.efficiency_score = max(0, 1.0 - response.total_tokens / 500.0)

            # Speed: normalized to 50 tokens/sec
            scored.speed_score = min(response.tokens_per_second / 50.0, 1.0)

            # Custom weighted average
            scored.overall_score = (
                0.40 * scored.correctness_score +
                0.25 * scored.completeness_score +
                0.15 * scored.clarity_score +
                0.10 * scored.efficiency_score +
                0.10 * scored.speed_score
            )
            scored.scoring_method = "assembly_quality"

            return scored

    # Create comparator with custom scorer
    comparator = ModelComparator(
        comparison_mode=ComparisonMode.HEAD_TO_HEAD,
        scorer=AssemblyQualityScorer()
    )

    # Load models
    model_v5 = SimpleModelMock("v5")
    model_v6 = SimpleModelMock("v6")

    comparator.load_model("v5", lambda: model_v5)
    comparator.load_model("v6", lambda: model_v6)

    # Run comparison
    prompts = ["Create a simple loop", "Handle interrupts"]
    print(f"\nRunning comparison with custom AssemblyQualityScorer...")
    report = comparator.run_prompts(prompts)

    # Print results
    print("\n" + "-" * 70)
    print("CUSTOM SCORING RESULTS")
    print("-" * 70)

    for i, result in enumerate(report.results, 1):
        print(f"\nPrompt {i}: {result.prompt}")
        for model_name, scored in result.responses.items():
            print(f"  {model_name}:")
            print(f"    Correctness: {scored.correctness_score:.2f}")
            print(f"    Completeness: {scored.completeness_score:.2f}")
            print(f"    Clarity: {scored.clarity_score:.2f}")
            print(f"    Efficiency: {scored.efficiency_score:.2f}")
            print(f"    Speed: {scored.speed_score:.2f}")
            print(f"    Overall: {scored.overall_score:.2f}")


def example_4_statistical_analysis():
    """Example 4: Statistical significance testing."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Statistical Analysis")
    print("=" * 70)

    from afs.comparison import StatisticalTester

    # Run a small tournament
    comparator = ModelComparator(
        comparison_mode=ComparisonMode.TOURNAMENT,
        scorer=BasicScorer()
    )

    models = {
        "Model A": SimpleModelMock("A", quality_multiplier=0.90),
        "Model B": SimpleModelMock("B", quality_multiplier=0.92),
    }

    for name, model in models.items():
        comparator.load_model(name, lambda m=model: m)

    # Generate 10 test prompts
    prompts = [f"Test prompt {i}" for i in range(10)]
    report = comparator.run_prompts(prompts)

    # Extract scores
    scores_a = [
        result.responses["Model A"].overall_score
        for result in report.results
        if "Model A" in result.responses
    ]
    scores_b = [
        result.responses["Model B"].overall_score
        for result in report.results
        if "Model B" in result.responses
    ]

    # Run statistical tests
    t_stat, is_significant = StatisticalTester.t_test(scores_a, scores_b)
    effect_size = StatisticalTester.effect_size(scores_a, scores_b)

    print(f"\nModel A Scores: {[f'{s:.3f}' for s in scores_a]}")
    print(f"Model B Scores: {[f'{s:.3f}' for s in scores_b]}")
    print(f"\nMean A: {sum(scores_a)/len(scores_a):.3f}")
    print(f"Mean B: {sum(scores_b)/len(scores_b):.3f}")
    print(f"\nStatistical Test Results:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  Significant (α=0.05): {is_significant}")
    print(f"  Cohen's d (effect size): {effect_size:.3f}")

    # Interpret effect size
    if abs(effect_size) < 0.2:
        interpretation = "negligible"
    elif abs(effect_size) < 0.5:
        interpretation = "small"
    elif abs(effect_size) < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    direction = "improves" if effect_size > 0 else "degrades"
    print(f"\nInterpretation: Model B {direction} on Model A ({interpretation})")


def example_5_report_generation():
    """Example 5: Generating comprehensive reports."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Report Generation")
    print("=" * 70)

    # Run comparison
    comparator = ModelComparator(
        comparison_mode=ComparisonMode.TOURNAMENT,
        scorer=BasicScorer()
    )

    models = {
        "v5": SimpleModelMock("v5", 0.85),
        "v6": SimpleModelMock("v6", 0.91),
        "v7": SimpleModelMock("v7", 0.93),
    }

    for name, model in models.items():
        comparator.load_model(name, lambda m=model: m)

    prompts = [f"Prompt {i}" for i in range(5)]
    report = comparator.run_prompts(prompts)

    # Generate different report formats
    print("\n1. Generating Markdown Report...")
    md_report = comparator.generate_markdown_report()
    print(f"   Generated: {len(md_report)} characters")
    print(f"   Preview:\n{md_report[:300]}...")

    print("\n2. Generating JSON Report...")
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "report.json"
        comparator.save_report_json(json_path)
        print(f"   Saved to: {json_path}")
        with open(json_path) as f:
            data = json.load(f)
            print(f"   Contains {len(data.get('results', []))} results")

    print("\n3. Generating HTML Dashboard...")
    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = Path(tmpdir) / "dashboard.html"
        comparator.generate_html_report(html_path)
        print(f"   Saved to: {html_path}")
        html_content = html_path.read_text()
        print(f"   Generated: {len(html_content)} characters")
        print(f"   Contains charts: {html_content.count('Plotly')}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MODEL COMPARISON FRAMEWORK TUTORIAL")
    print("=" * 70)

    # Run all examples
    example_1_basic_comparison()
    example_2_tournament()
    example_3_custom_scorer()
    example_4_statistical_analysis()
    example_5_report_generation()

    print("\n" + "=" * 70)
    print("TUTORIAL COMPLETE")
    print("=" * 70)
    print("""
Next steps:
1. Read the documentation: docs/COMPARISON_FRAMEWORK.md
2. Try the CLI: python3 -m afs comparison --help
3. Run a real comparison with your models
4. Implement custom scorers for your domain
5. Integrate with your CI/CD pipeline for regression testing

For questions or issues, check the framework source code:
/Users/scawful/src/lab/afs/src/afs/comparison/framework.py
""")
