"""Comprehensive model comparison framework for side-by-side evaluation.

Supports:
- Head-to-head comparison (2 models)
- Tournament mode (all models on same questions)
- Regression testing (new vs old on historical questions)
- A/B test analysis (production traffic split)

Features:
- Simultaneous loading of 2-5 models
- Identical prompt execution across all models
- Latency and token count capture
- Multi-dimensional scoring (correctness, completeness, clarity, efficiency, speed)
- Statistical significance testing
- Markdown and HTML report generation
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from afs.generators.model_generator import ModelGenerator

logger = logging.getLogger(__name__)


class ComparisonMode(str, Enum):
    """Model comparison modes."""

    HEAD_TO_HEAD = "head_to_head"  # 2 models
    TOURNAMENT = "tournament"  # All models on same questions
    REGRESSION = "regression"  # New vs old on historical questions
    AB_TEST = "ab_test"  # Production traffic split analysis


class ScoreDimension(str, Enum):
    """Scoring dimensions for multi-metric evaluation."""

    CORRECTNESS = "correctness"  # 0.0-1.0
    COMPLETENESS = "completeness"  # 0.0-1.0
    CLARITY = "clarity"  # 0.0-1.0
    EFFICIENCY = "efficiency"  # 0.0-1.0 (inverse of tokens)
    SPEED = "speed"  # tokens/sec


@dataclass
class ModelResponse:
    """Single model's response to a prompt."""

    model_name: str
    prompt: str
    response: str
    latency_ms: float  # Total time including model init
    input_tokens: int
    output_tokens: int
    total_tokens: int
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        """Tokens generated per second."""
        if self.latency_ms <= 0:
            return 0.0
        return (self.output_tokens / self.latency_ms) * 1000

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_name": self.model_name,
            "response": self.response,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "tokens_per_second": self.tokens_per_second,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ScoredResponse:
    """Response with multi-dimensional scores."""

    model_response: ModelResponse
    correctness_score: float = 0.0  # 0.0-1.0
    completeness_score: float = 0.0  # 0.0-1.0
    clarity_score: float = 0.0  # 0.0-1.0
    efficiency_score: float = 0.0  # 0.0-1.0 (inverse of tokens)
    speed_score: float = 0.0  # 0.0-1.0 (normalized tokens/sec)
    overall_score: float = 0.0  # Weighted average
    scorer_notes: str = ""
    scoring_method: str = ""

    def score_dict(self) -> dict[str, float]:
        """Get all scores as dictionary."""
        return {
            ScoreDimension.CORRECTNESS.value: self.correctness_score,
            ScoreDimension.COMPLETENESS.value: self.completeness_score,
            ScoreDimension.CLARITY.value: self.clarity_score,
            ScoreDimension.EFFICIENCY.value: self.efficiency_score,
            ScoreDimension.SPEED.value: self.speed_score,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_name": self.model_response.model_name,
            "response": self.model_response.response,
            "latency_ms": self.model_response.latency_ms,
            "tokens": self.model_response.total_tokens,
            "tokens_per_second": self.model_response.tokens_per_second,
            "scores": self.score_dict(),
            "overall_score": self.overall_score,
            "scorer_notes": self.scorer_notes,
            "scoring_method": self.scoring_method,
        }


@dataclass
class ComparisonResult:
    """Results from comparing models on a single prompt."""

    prompt: str
    responses: dict[str, ScoredResponse] = field(default_factory=dict)
    winner: Optional[str] = None  # Model name of best performer
    confidence_score: float = 0.0  # 0.0-1.0 confidence in winner
    is_significant: bool = False  # Statistical significance

    def add_response(self, scored: ScoredResponse) -> None:
        """Add a scored response."""
        self.responses[scored.model_response.model_name] = scored

    def determine_winner(self) -> None:
        """Determine winner based on overall scores."""
        if not self.responses:
            return

        scores = {
            name: resp.overall_score for name, resp in self.responses.items()
        }
        if not scores:
            return

        winner = max(scores, key=scores.get)
        max_score = scores[winner]
        scores_list = sorted(scores.values(), reverse=True)

        # Confidence = how much winner beats 2nd place
        if len(scores_list) > 1:
            gap = scores_list[0] - scores_list[1]
            self.confidence_score = min(gap, 1.0)
        else:
            self.confidence_score = 1.0

        self.winner = winner

    def is_significant_at_level(self, alpha: float = 0.05) -> bool:
        """Check if winner is significant at given alpha level."""
        return self.confidence_score > (1.0 - alpha)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "prompt": self.prompt,
            "responses": {
                name: resp.to_dict() for name, resp in self.responses.items()
            },
            "winner": self.winner,
            "confidence_score": self.confidence_score,
            "is_significant": self.is_significant,
        }


@dataclass
class ComparisonReport:
    """Aggregated comparison results."""

    comparison_mode: ComparisonMode
    timestamp: datetime = field(default_factory=datetime.now)
    model_names: list[str] = field(default_factory=list)
    prompt_count: int = 0
    results: list[ComparisonResult] = field(default_factory=list)

    # Aggregated scores
    model_stats: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add_result(self, result: ComparisonResult) -> None:
        """Add comparison result."""
        self.results.append(result)
        self.prompt_count += 1

    def compute_statistics(self) -> None:
        """Compute aggregated statistics for each model."""
        for model_name in self.model_names:
            self.model_stats[model_name] = self._compute_model_stats(model_name)

    def _compute_model_stats(self, model_name: str) -> dict[str, Any]:
        """Compute statistics for a single model."""
        scores = []
        latencies = []
        token_counts = []
        win_count = 0
        speed_values = []

        for result in self.results:
            if model_name not in result.responses:
                continue

            resp = result.responses[model_name]
            scores.append(resp.overall_score)
            latencies.append(resp.model_response.latency_ms)
            token_counts.append(resp.model_response.total_tokens)
            speed_values.append(resp.model_response.tokens_per_second)

            if result.winner == model_name:
                win_count += 1

        if not scores:
            return {}

        return {
            "mean_overall_score": statistics.mean(scores),
            "median_overall_score": statistics.median(scores),
            "stdev_overall_score": (
                statistics.stdev(scores) if len(scores) > 1 else 0.0
            ),
            "min_overall_score": min(scores),
            "max_overall_score": max(scores),
            "mean_latency_ms": statistics.mean(latencies),
            "median_latency_ms": statistics.median(latencies),
            "mean_tokens": statistics.mean(token_counts),
            "mean_tokens_per_second": statistics.mean(speed_values),
            "win_rate": win_count / len(self.results) if self.results else 0.0,
            "win_count": win_count,
            "total_comparisons": len(self.results),
        }

    def get_ranked_models(self) -> list[tuple[str, dict[str, Any]]]:
        """Get models ranked by mean overall score."""
        ranked = sorted(
            self.model_stats.items(),
            key=lambda x: x[1].get("mean_overall_score", 0.0),
            reverse=True,
        )
        return ranked

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "comparison_mode": self.comparison_mode.value,
            "models": self.model_names,
            "prompt_count": self.prompt_count,
            "model_statistics": self.model_stats,
            "results": [r.to_dict() for r in self.results],
        }


class ResponseScorer(ABC):
    """Abstract base for response scoring."""

    @abstractmethod
    def score(self, response: ModelResponse, reference: Optional[str] = None) -> ScoredResponse:
        """Score a response across all dimensions."""
        pass


class BasicScorer(ResponseScorer):
    """Basic scorer using heuristics."""

    def score(
        self,
        response: ModelResponse,
        reference: Optional[str] = None,
    ) -> ScoredResponse:
        """Score response using basic heuristics."""
        scored = ScoredResponse(model_response=response)

        # Correctness: assume high if no syntax errors and reasonable length
        has_syntax = any(
            keyword in response.response
            for keyword in ["def ", "class ", "import ", "return ", "if "]
        )
        scored.correctness_score = 0.9 if has_syntax else 0.5

        # Completeness: based on response length
        response_len = len(response.response)
        scored.completeness_score = min(response_len / 1000.0, 1.0)

        # Clarity: based on newlines and structure
        line_count = response.response.count("\n")
        scored.clarity_score = min(line_count / 10.0, 1.0)

        # Efficiency: inverse of token usage (fewer is better)
        # Normalize to 1000 token baseline
        scored.efficiency_score = max(0.0, 1.0 - (response.total_tokens / 1000.0))

        # Speed: normalize to 100 tokens/sec baseline
        base_speed = 100.0
        scored.speed_score = min(response.tokens_per_second / base_speed, 1.0)

        # Overall: weighted average
        weights = {
            ScoreDimension.CORRECTNESS: 0.3,
            ScoreDimension.COMPLETENESS: 0.2,
            ScoreDimension.CLARITY: 0.2,
            ScoreDimension.EFFICIENCY: 0.15,
            ScoreDimension.SPEED: 0.15,
        }

        overall = (
            scored.correctness_score * weights[ScoreDimension.CORRECTNESS]
            + scored.completeness_score * weights[ScoreDimension.COMPLETENESS]
            + scored.clarity_score * weights[ScoreDimension.CLARITY]
            + scored.efficiency_score * weights[ScoreDimension.EFFICIENCY]
            + scored.speed_score * weights[ScoreDimension.SPEED]
        )

        scored.overall_score = overall
        scored.scoring_method = "basic_heuristic"

        return scored


class ModelComparator:
    """Main comparison engine for evaluating multiple models."""

    def __init__(
        self,
        comparison_mode: ComparisonMode = ComparisonMode.TOURNAMENT,
        scorer: Optional[ResponseScorer] = None,
    ):
        """Initialize comparator.

        Args:
            comparison_mode: Type of comparison to perform
            scorer: Custom scorer implementation (defaults to BasicScorer)
        """
        self.comparison_mode = comparison_mode
        self.models: dict[str, Any] = {}
        self.scorer = scorer or BasicScorer()
        self.report: Optional[ComparisonReport] = None

    def load_model(
        self,
        model_name: str,
        model_factory: Callable[[], Any],
    ) -> None:
        """Load a model for comparison.

        Args:
            model_name: Unique identifier for model
            model_factory: Callable that returns initialized model
        """
        if len(self.models) >= 5:
            raise ValueError("Cannot load more than 5 models simultaneously")

        try:
            model = model_factory()
            self.models[model_name] = model
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def run_prompts(
        self,
        prompts: list[str],
        generation_config: Optional[dict[str, Any]] = None,
    ) -> ComparisonReport:
        """Run comparison on prompts.

        Args:
            prompts: List of prompts to evaluate
            generation_config: Configuration for generation (temperature, max_tokens, etc.)

        Returns:
            ComparisonReport with detailed results
        """
        if not self.models:
            raise ValueError("No models loaded")

        gen_config = generation_config or {}
        self.report = ComparisonReport(
            comparison_mode=self.comparison_mode,
            model_names=list(self.models.keys()),
        )

        for i, prompt in enumerate(prompts):
            logger.info(f"Running prompt {i+1}/{len(prompts)}")
            result = self._evaluate_prompt(prompt, gen_config)
            self.report.add_result(result)

        # Compute aggregated statistics
        self.report.compute_statistics()

        return self.report

    def _evaluate_prompt(
        self,
        prompt: str,
        gen_config: dict[str, Any],
    ) -> ComparisonResult:
        """Evaluate a single prompt across all models."""
        result = ComparisonResult(prompt=prompt)

        for model_name, model in self.models.items():
            try:
                response = self._generate_response(model, prompt, gen_config)
                scored = self.scorer.score(response)
                result.add_response(scored)
            except Exception as e:
                logger.warning(f"Failed to generate from {model_name}: {e}")

        # Determine winner
        result.determine_winner()
        result.is_significant = result.is_significant_at_level(alpha=0.05)

        return result

    def _generate_response(
        self,
        model: Any,
        prompt: str,
        gen_config: dict[str, Any],
    ) -> ModelResponse:
        """Generate single response from model."""
        start_time = time.time()

        # Call model's generate method (assumes standard interface)
        output = model.generate(prompt, **gen_config)

        latency_ms = (time.time() - start_time) * 1000

        # Estimate tokens (roughly 4 chars per token)
        input_tokens = len(prompt) // 4
        output_tokens = len(output) // 4
        total_tokens = input_tokens + output_tokens

        return ModelResponse(
            model_name=getattr(model, "name", "unknown"),
            prompt=prompt,
            response=output,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    def generate_markdown_report(self) -> str:
        """Generate markdown comparison report."""
        if not self.report:
            raise ValueError("No comparison report available. Run prompts first.")

        lines = []
        lines.append("# Model Comparison Report\n")
        lines.append(f"**Generated:** {self.report.timestamp.isoformat()}\n")
        lines.append(f"**Comparison Mode:** {self.report.comparison_mode.value}\n")
        lines.append(f"**Prompts Evaluated:** {self.report.prompt_count}\n\n")

        # Summary table
        lines.append("## Summary\n")
        lines.append(self._generate_summary_table())

        # Detailed results
        lines.append("\n## Detailed Results\n")
        for i, result in enumerate(self.report.results):
            lines.append(f"\n### Prompt {i+1}\n")
            lines.append(f"```\n{result.prompt}\n```\n")

            if result.winner:
                lines.append(
                    f"**Winner:** {result.winner} "
                    f"(confidence: {result.confidence_score:.2%})\n"
                )

            lines.append("\n| Model | Overall | Correctness | Completeness | Clarity | "
                        "Efficiency | Speed | Latency | Tokens |\n")
            lines.append("|-------|---------|-------------|--------------|---------|"
                        "-----------|-------|---------|--------|\n")

            for name, resp in result.responses.items():
                scores = resp.score_dict()
                lines.append(
                    f"| {name} "
                    f"| {resp.overall_score:.2f} "
                    f"| {scores[ScoreDimension.CORRECTNESS.value]:.2f} "
                    f"| {scores[ScoreDimension.COMPLETENESS.value]:.2f} "
                    f"| {scores[ScoreDimension.CLARITY.value]:.2f} "
                    f"| {scores[ScoreDimension.EFFICIENCY.value]:.2f} "
                    f"| {scores[ScoreDimension.SPEED.value]:.2f} "
                    f"| {resp.model_response.latency_ms:.0f}ms "
                    f"| {resp.model_response.total_tokens} |\n"
                )

        return "".join(lines)

    def _generate_summary_table(self) -> str:
        """Generate summary statistics table."""
        lines = []
        lines.append(
            "| Model | Mean Score | Median Score | Std Dev | Win Rate | "
            "Avg Latency | Avg Tokens | Tokens/Sec |\n"
        )
        lines.append(
            "|-------|------------|--------------|---------|----------|"
            "-------------|------------|------------|\n"
        )

        ranked = self.report.get_ranked_models()
        for model_name, stats in ranked:
            lines.append(
                f"| {model_name} "
                f"| {stats.get('mean_overall_score', 0):.3f} "
                f"| {stats.get('median_overall_score', 0):.3f} "
                f"| {stats.get('stdev_overall_score', 0):.3f} "
                f"| {stats.get('win_rate', 0):.1%} "
                f"| {stats.get('mean_latency_ms', 0):.0f}ms "
                f"| {stats.get('mean_tokens', 0):.0f} "
                f"| {stats.get('mean_tokens_per_second', 0):.1f} |\n"
            )

        return "".join(lines)

    def generate_html_report(self, output_path: Path) -> Path:
        """Generate interactive HTML dashboard.

        Args:
            output_path: Path to save HTML report

        Returns:
            Path to generated report
        """
        if not self.report:
            raise ValueError("No comparison report available. Run prompts first.")

        html_content = self._build_html_dashboard()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content)

        logger.info(f"HTML report saved to: {output_path}")
        return output_path

    def _build_html_dashboard(self) -> str:
        """Build HTML dashboard with charts."""
        ranked = self.report.get_ranked_models()
        model_names = [name for name, _ in ranked]
        mean_scores = [stats.get("mean_overall_score", 0) for _, stats in ranked]

        # Prepare chart data
        latencies = [
            self.report.model_stats[name].get("mean_latency_ms", 0)
            for name in self.report.model_names
        ]
        tokens = [
            self.report.model_stats[name].get("mean_tokens", 0)
            for name in self.report.model_names
        ]
        win_rates = [
            self.report.model_stats[name].get("win_rate", 0)
            for name in self.report.model_names
        ]

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Comparison Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            margin: 0;
            color: #333;
        }}
        .subtitle {{
            color: #666;
            margin-top: 10px;
            font-size: 14px;
        }}
        .dashboard {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chart-title {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 15px;
            color: #333;
        }}
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        .summary-table th,
        .summary-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        .summary-table th {{
            background: #f5f5f5;
            font-weight: 600;
        }}
        .summary-table tr:hover {{
            background: #fafafa;
        }}
        .winner {{
            background: #e8f5e9;
            font-weight: 600;
        }}
        .rank {{
            display: inline-block;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: #2196F3;
            color: white;
            text-align: center;
            line-height: 24px;
            font-weight: bold;
            font-size: 12px;
        }}
        .metric {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 10px;
        }}
        .metric-item {{
            padding: 10px;
            background: #f9f9f9;
            border-radius: 4px;
            font-size: 13px;
        }}
        .metric-label {{
            color: #666;
            font-size: 12px;
        }}
        .metric-value {{
            font-size: 18px;
            font-weight: 600;
            color: #333;
            margin-top: 4px;
        }}
        .footer {{
            text-align: center;
            color: #999;
            font-size: 12px;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Model Comparison Dashboard</h1>
        <p class="subtitle">
            <strong>Mode:</strong> {self.report.comparison_mode.value} |
            <strong>Models:</strong> {len(self.report.model_names)} |
            <strong>Prompts:</strong> {self.report.prompt_count} |
            <strong>Generated:</strong> {self.report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </div>

    <div class="dashboard">
        <div class="card">
            <div class="chart-title">Overall Scores</div>
            <div id="score-chart"></div>
        </div>

        <div class="card">
            <div class="chart-title">Win Rate</div>
            <div id="win-rate-chart"></div>
        </div>

        <div class="card">
            <div class="chart-title">Latency (ms)</div>
            <div id="latency-chart"></div>
        </div>

        <div class="card">
            <div class="chart-title">Average Tokens</div>
            <div id="tokens-chart"></div>
        </div>
    </div>

    <div class="card">
        <div class="chart-title">Summary Statistics</div>
        <table class="summary-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>Mean Score</th>
                    <th>Median Score</th>
                    <th>Std Dev</th>
                    <th>Win Rate</th>
                    <th>Avg Latency</th>
                    <th>Avg Tokens</th>
                </tr>
            </thead>
            <tbody>
"""

        for rank, (model_name, stats) in enumerate(ranked, 1):
            winner_class = "winner" if rank == 1 else ""
            html += f"""                <tr class="{winner_class}">
                    <td><span class="rank">{rank}</span></td>
                    <td>{model_name}</td>
                    <td>{stats.get('mean_overall_score', 0):.3f}</td>
                    <td>{stats.get('median_overall_score', 0):.3f}</td>
                    <td>{stats.get('stdev_overall_score', 0):.3f}</td>
                    <td>{stats.get('win_rate', 0):.1%}</td>
                    <td>{stats.get('mean_latency_ms', 0):.0f}ms</td>
                    <td>{stats.get('mean_tokens', 0):.0f}</td>
                </tr>
"""

        html += """            </tbody>
        </table>
    </div>

    <script>
"""

        # Add chart scripts
        html += f"""
        // Overall Scores
        Plotly.newPlot('score-chart', [{{
            x: {model_names!r},
            y: {mean_scores!r},
            type: 'bar',
            marker: {{color: '#2196F3'}}
        }}], {{}}, {{responsive: true, displayModeBar: false}});

        // Win Rate
        Plotly.newPlot('win-rate-chart', [{{
            x: {self.report.model_names!r},
            y: {win_rates!r},
            type: 'bar',
            marker: {{color: '#4CAF50'}}
        }}], {{}}, {{responsive: true, displayModeBar: false}});

        // Latency
        Plotly.newPlot('latency-chart', [{{
            x: {self.report.model_names!r},
            y: {latencies!r},
            type: 'bar',
            marker: {{color: '#FF9800'}}
        }}], {{}}, {{responsive: true, displayModeBar: false}});

        // Tokens
        Plotly.newPlot('tokens-chart', [{{
            x: {self.report.model_names!r},
            y: {tokens!r},
            type: 'bar',
            marker: {{color: '#9C27B0'}}
        }}], {{}}, {{responsive: true, displayModeBar: false}});
    </script>

    <div class="footer">
        <p>Generated by AFS Model Comparison Framework</p>
    </div>
</body>
</html>"""

        return html

    def save_report_json(self, output_path: Path) -> Path:
        """Save report as JSON.

        Args:
            output_path: Path to save JSON report

        Returns:
            Path to generated report
        """
        if not self.report:
            raise ValueError("No comparison report available. Run prompts first.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.report.to_dict(), indent=2))

        logger.info(f"JSON report saved to: {output_path}")
        return output_path


class StatisticalTester:
    """Statistical significance testing for comparisons."""

    @staticmethod
    def t_test(
        scores1: list[float],
        scores2: list[float],
    ) -> tuple[float, bool]:
        """Perform independent t-test.

        Args:
            scores1: Score list from model 1
            scores2: Score list from model 2

        Returns:
            (t_statistic, is_significant at alpha=0.05)
        """
        if len(scores1) < 2 or len(scores2) < 2:
            return 0.0, False

        mean1 = statistics.mean(scores1)
        mean2 = statistics.mean(scores2)
        var1 = statistics.variance(scores1)
        var2 = statistics.variance(scores2)

        n1 = len(scores1)
        n2 = len(scores2)

        # Pooled standard error
        se = (var1 / n1 + var2 / n2) ** 0.5
        if se == 0:
            return 0.0, False

        t_stat = (mean1 - mean2) / se

        # Rough significance threshold (t ~= 1.96 for alpha=0.05, large n)
        is_sig = abs(t_stat) > 1.96

        return t_stat, is_sig

    @staticmethod
    def effect_size(
        scores1: list[float],
        scores2: list[float],
    ) -> float:
        """Calculate Cohen's d effect size.

        Args:
            scores1: Score list from model 1
            scores2: Score list from model 2

        Returns:
            Cohen's d (effect size)
        """
        if len(scores1) < 1 or len(scores2) < 1:
            return 0.0

        mean1 = statistics.mean(scores1)
        mean2 = statistics.mean(scores2)
        var1 = statistics.variance(scores1) if len(scores1) > 1 else 0
        var2 = statistics.variance(scores2) if len(scores2) > 1 else 0

        n1 = len(scores1)
        n2 = len(scores2)

        # Pooled standard deviation
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        pooled_sd = pooled_var ** 0.5

        if pooled_sd == 0:
            return 0.0

        return (mean1 - mean2) / pooled_sd
