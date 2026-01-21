#!/usr/bin/env python3
"""
Comprehensive Model Comparison Suite

Runs all 5 models on the same question set and generates:
- Accuracy metrics per category
- Speed comparisons
- Quality assessments
- HTML dashboard with charts
- Visual comparison screenshots
"""

import json
import os
import sys
import time
import requests
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import statistics

# Add src to path to allow importing lmstudio_client
sys.path.append(str(Path(__file__).parent.parent))
from lmstudio_client import LMStudioClient

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: matplotlib not available, chart generation disabled")

# AFS project root
AFS_ROOT = Path(__file__).parent.parent
EVAL_SUITE = Path.home() / ".context/training/evals/unified_eval_suite.jsonl"
RESULTS_DIR = Path.home() / ".context/training/evals/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class ModelEvaluator:
    """Evaluate a single model across the test suite."""

    def __init__(self, model_key: str, endpoint: str, category: str = "general"):
        self.model_key = model_key
        self.name = model_key
        self.endpoint = endpoint
        self.timeout = 30
        self.results = []
        self.metrics = defaultdict(lambda: {
            "correct": 0,
            "total": 0,
            "avg_time": 0,
            "times": [],
            "quality_scores": []
        })

    def query_model(self, prompt: str) -> Tuple[str, float]:
        """Send query to model, return response and time taken."""
        start = time.time()
        try:
            response = requests.post(
                self.endpoint,
                json={"prompt": prompt},
                timeout=self.timeout
            )
            elapsed = time.time() - start
            if response.status_code == 200:
                return response.json().get("response", ""), elapsed
            else:
                return f"[ERROR {response.status_code}]", elapsed
        except requests.exceptions.Timeout:
            return "[TIMEOUT]", self.timeout
        except requests.exceptions.ConnectionError:
            return "[CONNECTION ERROR]", 0
        except Exception as e:
            return f"[ERROR: {str(e)}]", 0

    def evaluate_response(self, question: Dict, response: str) -> float:
        """Score response quality (0-1)."""
        if not response or response.startswith("["):
            return 0.0

        score = 0.0

        # Check for expected features/keywords
        if "expected_features" in question:
            found = sum(1 for feature in question["expected_features"]
                       if feature.lower() in response.lower())
            score += (found / len(question["expected_features"])) * 0.5

        # Check for exact answer match
        if "expected_answer" in question:
            if question["expected_answer"].lower() in response.lower():
                score += 0.3
            elif any(part.lower() in response.lower()
                    for part in question["expected_answer"].split("/")):
                score += 0.15

        # Length heuristic: longer responses usually better for code
        if question.get("category") in ["code_generation", "assembly_generation"]:
            if len(response) > 100:
                score += 0.2

        return min(score, 1.0)

    def run_evaluation(self, questions: List[Dict]) -> Dict:
        """Run evaluation on all questions."""
        print(f"\nEvaluating {self.name}...")
        self.results = []

        for i, question in enumerate(questions):
            print(f"  [{i+1}/{len(questions)}] {question['id']}", end=" ... ")
            sys.stdout.flush()

            prompt = question.get("prompt") or question.get("question")
            response, elapsed = self.query_model(prompt)
            score = self.evaluate_response(question, response)

            category = question.get("category", "unknown")
            self.metrics[category]["times"].append(elapsed)
            self.metrics[category]["quality_scores"].append(score)
            self.metrics[category]["total"] += 1
            if score >= 0.7:
                self.metrics[category]["correct"] += 1

            self.results.append({
                "id": question["id"],
                "category": category,
                "difficulty": question.get("difficulty", "unknown"),
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "response": response[:200] + "..." if len(response) > 200 else response,
                "score": score,
                "time": elapsed
            })

            status = "✓" if score >= 0.7 else "✗"
            print(f"{status} {score:.2f} ({elapsed:.2f}s)")

        # Aggregate metrics
        for category, metrics in self.metrics.items():
            if metrics["times"]:
                metrics["avg_time"] = statistics.mean(metrics["times"])
            if metrics["quality_scores"]:
                metrics["avg_quality"] = statistics.mean(metrics["quality_scores"])

        return self.get_summary()

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        all_scores = [r["score"] for r in self.results]
        all_times = [r["time"] for r in self.results if r["time"] > 0]

        return {
            "model": self.name,
            "model_key": self.model_key,
            "total_questions": len(self.results),
            "avg_score": statistics.mean(all_scores) if all_scores else 0,
            "median_score": statistics.median(all_scores) if all_scores else 0,
            "avg_time": statistics.mean(all_times) if all_times else 0,
            "success_rate": sum(1 for s in all_scores if s >= 0.7) / len(all_scores) if all_scores else 0,
            "category_metrics": dict(self.metrics)
        }


def load_eval_suite() -> List[Dict]:
    """Load the unified evaluation suite."""
    if not EVAL_SUITE.exists():
        print(f"ERROR: Evaluation suite not found at {EVAL_SUITE}")
        sys.exit(1)

    questions = []
    with open(EVAL_SUITE, 'r') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    return questions


def create_comparison_table(summaries: List[Dict]) -> str:
    """Create markdown comparison table."""
    table = "# Model Comparison Results\n\n"
    table += f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    table += f"**Questions Evaluated:** {summaries[0]['total_questions']}\n\n"

    # Main metrics table
    table += "## Overall Metrics\n\n"
    table += "| Model | Avg Score | Median Score | Success Rate | Avg Time (s) |\n"
    table += "|-------|-----------|--------------|--------------|---------------|\n"

    for s in sorted(summaries, key=lambda x: x["avg_score"], reverse=True):
        table += (f"| {s['model']} | "
                 f"{s['avg_score']:.3f} | "
                 f"{s['median_score']:.3f} | "
                 f"{s['success_rate']:.1%} | "
                 f"{s['avg_time']:.2f} |\n")

    # Category breakdown
    table += "\n## Performance by Category\n\n"

    # Get all categories
    all_categories = set()
    for s in summaries:
        all_categories.update(s["category_metrics"].keys())

    for category in sorted(all_categories):
        table += f"### {category.replace('_', ' ').title()}\n\n"
        table += "| Model | Success | Avg Quality | Avg Time |\n"
        table += "|-------|---------|-------------|----------|\n"

        for s in sorted(summaries, key=lambda x: x["model"]):
            metrics = s["category_metrics"].get(category, {})
            success = f"{metrics.get('correct', 0)}/{metrics.get('total', 0)}"
            avg_quality = metrics.get("avg_quality", 0)
            avg_time = metrics.get("avg_time", 0)
            table += (f"| {s['model']} | {success} | "
                     f"{avg_quality:.3f} | {avg_time:.2f}s |\n")

        table += "\n"

    return table


def create_html_dashboard(summaries: List[Dict], detailed_results: Dict) -> str:
    """Create interactive HTML dashboard."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Model Comparison Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 { color: #333; text-align: center; }
        .summary { background: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .chart-container {
            position: relative;
            width: 48%;
            display: inline-block;
            margin: 1%;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .metric { display: inline-block; margin: 10px 20px; }
        .good { color: #4CAF50; font-weight: bold; }
        .bad { color: #f44336; font-weight: bold; }
        .neutral { color: #ff9800; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Model Comparison Dashboard</h1>
    <div class="summary">
        <p><strong>Evaluation Date:</strong> """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        <p><strong>Models Evaluated:</strong> """ + str(len(summaries)) + """</p>
        <p><strong>Total Questions:</strong> """ + str(summaries[0]['total_questions'] if summaries else 0) + """</p>
    </div>

    <div class="chart-container">
        <canvas id="scoreChart"></canvas>
    </div>
    <div class="chart-container">
        <canvas id="timeChart"></canvas>
    </div>

    <h2>Overall Metrics</h2>
    <table>
        <tr>
            <th>Model</th>
            <th>Avg Score</th>
            <th>Success Rate</th>
            <th>Avg Time (s)</th>
        </tr>
"""

    for s in sorted(summaries, key=lambda x: x["avg_score"], reverse=True):
        html += f"""        <tr>
            <td>{s['model']}</td>
            <td><span class="good">{s['avg_score']:.3f}</span></td>
            <td><span class="good">{s['success_rate']:.1%}</span></td>
            <td>{s['avg_time']:.2f}</td>
        </tr>
"""

    html += """    </table>

    <h2>Detailed Results</h2>
    <div id="detailsContainer"></div>

    <script>
        const summaries = """ + json.dumps(summaries) + """;

        // Score chart
        const scoreCtx = document.getElementById('scoreChart').getContext('2d');
        new Chart(scoreCtx, {
            type: 'bar',
            data: {
                labels: summaries.map(s => s.model),
                datasets: [{
                    label: 'Average Score',
                    data: summaries.map(s => s.avg_score),
                    backgroundColor: '#4CAF50',
                    borderColor: '#45a049',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: { title: { display: true, text: 'Model Accuracy Comparison' } },
                scales: { y: { min: 0, max: 1 } }
            }
        });

        // Time chart
        const timeCtx = document.getElementById('timeChart').getContext('2d');
        new Chart(timeCtx, {
            type: 'bar',
            data: {
                labels: summaries.map(s => s.model),
                datasets: [{
                    label: 'Average Time (s)',
                    data: summaries.map(s => s.avg_time),
                    backgroundColor: '#2196F3',
                    borderColor: '#0b7dda',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: { title: { display: true, text: 'Response Time Comparison' } }
            }
        });
    </script>
</body>
</html>
"""
    return html


def generate_charts(summaries: List[Dict]) -> None:
    """Generate matplotlib comparison charts."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping chart generation (matplotlib not available)")
        return

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig)

    models = [s["model"] for s in sorted(summaries, key=lambda x: x["model"])]
    colors = plt.cm.Set3(range(len(models)))

    # 1. Average Score Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    scores = [s["avg_score"] for s in summaries]
    ax1.bar(models, scores, color=colors)
    ax1.set_ylabel("Average Score")
    ax1.set_title("Model Accuracy Comparison")
    ax1.set_ylim([0, 1])
    ax1.tick_params(axis='x', rotation=45)

    # 2. Success Rate Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    success_rates = [s["success_rate"] for s in summaries]
    ax2.bar(models, success_rates, color=colors)
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_title("Success Rate (score >= 0.7)")
    ax2.set_ylim([0, 1])
    ax2.tick_params(axis='x', rotation=45)

    # 3. Speed Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    avg_times = [s["avg_time"] for s in summaries]
    ax3.bar(models, avg_times, color=colors)
    ax3.set_ylabel("Time (seconds)")
    ax3.set_title("Average Response Time")
    ax3.tick_params(axis='x', rotation=45)

    # 4. Category Performance
    ax4 = fig.add_subplot(gs[1, 1])
    categories = set()
    for s in summaries:
        categories.update(s["category_metrics"].keys())
    categories = sorted(categories)[:5]  # Top 5 categories

    x = range(len(models))
    width = 0.15
    for i, category in enumerate(categories):
        scores = []
        for s in summaries:
            metrics = s["category_metrics"].get(category, {})
            avg_quality = metrics.get("avg_quality", 0)
            scores.append(avg_quality)
        ax4.bar([p + width * i for p in x], scores, width, label=category)

    ax4.set_ylabel("Quality Score")
    ax4.set_title("Performance by Category")
    ax4.set_xticks([p + width * 2 for p in x])
    ax4.set_xticklabels(models, rotation=45)
    ax4.legend(fontsize=8)

    # 5. Efficiency (Score vs Time)
    ax5 = fig.add_subplot(gs[2, 0])
    for s, color in zip(summaries, colors):
        ax5.scatter(s["avg_time"], s["avg_score"], s=200, alpha=0.6, color=color, label=s["model"])
    ax5.set_xlabel("Average Time (s)")
    ax5.set_ylabel("Average Score")
    ax5.set_title("Efficiency: Score vs Time")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Summary Stats
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    summary_text = "Summary Statistics\n" + "=" * 40 + "\n"
    for s in sorted(summaries, key=lambda x: x["avg_score"], reverse=True):
        summary_text += f"\n{s['model']}\n"
        summary_text += f"  Score: {s['avg_score']:.3f} | Time: {s['avg_time']:.2f}s\n"
        summary_text += f"  Success: {s['success_rate']:.1%}\n"

    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    chart_path = RESULTS_DIR / f"comparison_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"\nCharts saved to: {chart_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare model performance")
    # Initialize client to get available models
    client = LMStudioClient()
    available_models = client.models

    parser.add_argument("--models", nargs="+", default=list(available_models.keys()),
                       help="Models to evaluate")
    parser.add_argument("--no-charts", action="store_true",
                       help="Skip chart generation")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Evaluate only first N questions")
    args = parser.parse_args()

    # Load evaluation suite
    print("Loading evaluation suite...")
    questions = load_eval_suite()
    print(f"Loaded {len(questions)} questions")

    if args.sample_size:
        questions = questions[:args.sample_size]
        print(f"Using sample size: {len(questions)}")

    # Evaluate models
    summaries = []
    detailed_results = {}

    for model_key in args.models:
        if model_key not in available_models:
            print(f"WARNING: Unknown model {model_key}")
            continue

        endpoint = available_models[model_key]
        evaluator = ModelEvaluator(model_key, endpoint)
        summary = evaluator.run_evaluation(questions)
        summaries.append(summary)
        detailed_results[model_key] = evaluator.results

    # Generate reports
    print("\n" + "=" * 60)
    print("GENERATING REPORTS")
    print("=" * 60)

    # Markdown table
    markdown_report = create_comparison_table(summaries)
    markdown_path = RESULTS_DIR / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(markdown_path, 'w') as f:
        f.write(markdown_report)
    print(f"\nMarkdown report: {markdown_path}")

    # HTML dashboard
    html_dashboard = create_html_dashboard(summaries, detailed_results)
    html_path = RESULTS_DIR / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(html_path, 'w') as f:
        f.write(html_dashboard)
    print(f"HTML dashboard: {html_path}")

    # JSON detailed results
    json_results = {
        "timestamp": datetime.now().isoformat(),
        "summaries": summaries,
        "detailed_results": detailed_results
    }
    json_path = RESULTS_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"JSON results: {json_path}")

    # Charts
    if not args.no_charts:
        print("\nGenerating charts...")
        generate_charts(summaries)

    # Print summary to console
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for s in sorted(summaries, key=lambda x: x["avg_score"], reverse=True):
        print(f"\n{s['model']}")
        print(f"  Average Score: {s['avg_score']:.3f}")
        print(f"  Success Rate:  {s['success_rate']:.1%}")
        print(f"  Avg Time:      {s['avg_time']:.2f}s")


if __name__ == "__main__":
    main()
