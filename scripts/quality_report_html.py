#!/usr/bin/env python3
"""Generate an interactive HTML report for dataset quality analysis.

Usage:
    python scripts/quality_report_html.py <dataset.jsonl> --output report.html
    python scripts/quality_report_html.py data.json --domain assembly --output results/report.html
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from afs.quality import DatasetAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_html_report(
    dataset_paths: list[Path],
    output_path: Path,
    domain: str = "general",
) -> None:
    """Generate an interactive HTML quality report.

    Args:
        dataset_paths: Paths to dataset files
        output_path: Path to save HTML report
        domain: Domain type for analysis
    """
    analyzer = DatasetAnalyzer(domain=domain)

    # Load samples
    all_samples = []
    for path in dataset_paths:
        logger.info(f"Loading: {path}")
        with open(path) as f:
            if path.suffix == ".jsonl":
                for line in f:
                    if line.strip():
                        all_samples.append(json.loads(line))
            elif path.suffix == ".json":
                data = json.load(f)
                if isinstance(data, list):
                    all_samples.extend(data)
                else:
                    all_samples.append(data)

    logger.info(f"Analyzing {len(all_samples)} samples...")
    report = analyzer.analyze(all_samples, dataset_name=dataset_paths[0].stem, domain=domain)

    # Generate HTML
    html = _generate_html(report, dataset_paths)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    logger.info(f"Saved HTML report to {output_path}")


def _generate_html(report, dataset_paths) -> str:
    """Generate HTML content for the report."""
    stats = report.statistics
    bias = report.bias_report

    # Calculate metrics
    high_quality = sum(1 for s in report.sample_qualities if s.overall_quality_score >= 0.8)
    medium_quality = sum(1 for s in report.sample_qualities if 0.5 <= s.overall_quality_score < 0.8)
    low_quality = sum(1 for s in report.sample_qualities if s.overall_quality_score < 0.5)
    duplicates = sum(1 for s in report.sample_qualities if s.is_duplicate)
    anomalies = sum(1 for s in report.sample_qualities if s.is_anomaly)

    # Build quality distribution chart data
    quality_dist = report.quality_distribution
    chart_labels = list(quality_dist.keys())
    chart_values = list(quality_dist.values())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Quality Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}

        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
        }}

        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        .content {{
            padding: 40px;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}

        .metric-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}

        .metric-card.good {{
            background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        }}

        .metric-card.warning {{
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        }}

        .metric-card.danger {{
            background: linear-gradient(135deg, #ff6b6b 0%, #ff8787 100%);
        }}

        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: white;
            margin: 10px 0;
        }}

        .metric-label {{
            font-size: 0.9em;
            color: rgba(255, 255, 255, 0.9);
        }}

        .section {{
            margin-bottom: 40px;
        }}

        .section-title {{
            font-size: 1.8em;
            color: #333;
            margin-bottom: 20px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}

        .chart-container {{
            position: relative;
            height: 400px;
            margin-bottom: 30px;
        }}

        .quality-breakdown {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 30px;
        }}

        .quality-bar {{
            padding: 15px;
            border-radius: 8px;
            color: white;
            text-align: center;
        }}

        .quality-bar.high {{
            background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        }}

        .quality-bar.medium {{
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        }}

        .quality-bar.low {{
            background: linear-gradient(135deg, #ff6b6b 0%, #ff8787 100%);
        }}

        .quality-bar h3 {{
            margin-bottom: 5px;
        }}

        .quality-bar .count {{
            font-size: 2em;
            font-weight: bold;
        }}

        .quality-bar .percentage {{
            font-size: 0.9em;
            opacity: 0.9;
        }}

        .bias-metrics {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }}

        .bias-item {{
            background: #f5f7fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}

        .bias-item h4 {{
            margin-bottom: 8px;
            color: #333;
        }}

        .bias-item .score {{
            font-size: 1.3em;
            font-weight: bold;
            color: #667eea;
        }}

        .recommendations {{
            background: #f0f4ff;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}

        .recommendations h3 {{
            margin-bottom: 15px;
            color: #333;
        }}

        .recommendations ul {{
            list-style: none;
            padding-left: 0;
        }}

        .recommendations li {{
            padding: 10px 0;
            padding-left: 25px;
            position: relative;
            color: #555;
            line-height: 1.6;
        }}

        .recommendations li:before {{
            content: "✓";
            position: absolute;
            left: 0;
            color: #667eea;
            font-weight: bold;
        }}

        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}

        .stats-table th {{
            background: #f5f7fa;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #333;
            border-bottom: 2px solid #ddd;
        }}

        .stats-table td {{
            padding: 12px;
            border-bottom: 1px solid #eee;
        }}

        .stats-table tr:hover {{
            background: #fafbfc;
        }}

        footer {{
            background: #f5f7fa;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }}

        .badge.success {{
            background: #d4edda;
            color: #155724;
        }}

        .badge.warning {{
            background: #fff3cd;
            color: #856404;
        }}

        .badge.danger {{
            background: #f8d7da;
            color: #721c24;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Dataset Quality Report</h1>
            <p>{report.dataset_name}</p>
        </header>

        <div class="content">
            <!-- Quick Metrics -->
            <div class="metrics-grid">
                <div class="metric-card good">
                    <div class="metric-label">Average Quality Score</div>
                    <div class="metric-value">{report.average_quality_score:.1%}</div>
                </div>
                <div class="metric-card good">
                    <div class="metric-label">Total Samples</div>
                    <div class="metric-value">{stats.total_samples:,}</div>
                </div>
                <div class="metric-card good">
                    <div class="metric-label">Unique Samples</div>
                    <div class="metric-value">{stats.unique_samples:,}</div>
                </div>
                <div class="metric-card warning">
                    <div class="metric-label">Dataset Size</div>
                    <div class="metric-value">{stats.total_size_bytes / 1024 / 1024:.1f} MB</div>
                </div>
            </div>

            <!-- Dataset Statistics Section -->
            <div class="section">
                <h2 class="section-title">📈 Dataset Statistics</h2>

                <table class="stats-table">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Instructions Count</td>
                        <td>{stats.instruction_count}</td>
                    </tr>
                    <tr>
                        <td>Instruction Length (avg)</td>
                        <td>{stats.instruction_avg_length:.0f} words (±{stats.instruction_std_length:.0f})</td>
                    </tr>
                    <tr>
                        <td>Instruction Vocabulary</td>
                        <td>{stats.instruction_vocab_size:,} unique words</td>
                    </tr>
                    <tr>
                        <td>Outputs Count</td>
                        <td>{stats.output_count}</td>
                    </tr>
                    <tr>
                        <td>Output Length (avg)</td>
                        <td>{stats.output_avg_length:.0f} words (±{stats.output_std_length:.0f})</td>
                    </tr>
                    <tr>
                        <td>Output Vocabulary</td>
                        <td>{stats.output_vocab_size:,} unique words</td>
                    </tr>
                </table>
            </div>

            <!-- Quality Distribution -->
            <div class="section">
                <h2 class="section-title">✨ Quality Distribution</h2>

                <div class="quality-breakdown">
                    <div class="quality-bar high">
                        <h3>High Quality</h3>
                        <div class="count">{high_quality}</div>
                        <div class="percentage">{high_quality/len(report.sample_qualities)*100:.1f}%</div>
                    </div>
                    <div class="quality-bar medium">
                        <h3>Medium Quality</h3>
                        <div class="count">{medium_quality}</div>
                        <div class="percentage">{medium_quality/len(report.sample_qualities)*100:.1f}%</div>
                    </div>
                    <div class="quality-bar low">
                        <h3>Low Quality</h3>
                        <div class="count">{low_quality}</div>
                        <div class="percentage">{low_quality/len(report.sample_qualities)*100:.1f}%</div>
                    </div>
                </div>

                <div class="chart-container">
                    <canvas id="qualityChart"></canvas>
                </div>
            </div>

            <!-- Issues Detected -->
            <div class="section">
                <h2 class="section-title">⚠️ Issues Detected</h2>

                <div class="metrics-grid">
                    <div class="metric-card {'danger' if duplicates > 0 else 'good'}">
                        <div class="metric-label">Duplicates Found</div>
                        <div class="metric-value">{duplicates}</div>
                    </div>
                    <div class="metric-card {'danger' if anomalies > 0 else 'good'}">
                        <div class="metric-label">Anomalies Found</div>
                        <div class="metric-value">{anomalies}</div>
                    </div>
                    <div class="metric-card {'warning' if stats.samples_with_syntax_errors > 0 else 'good'}">
                        <div class="metric-label">Syntax Errors</div>
                        <div class="metric-value">{stats.samples_with_syntax_errors}</div>
                    </div>
                </div>
            </div>

            <!-- Bias Analysis -->
            <div class="section">
                <h2 class="section-title">🎯 Bias Analysis</h2>

                <div class="bias-metrics">
                    <div class="bias-item">
                        <h4>Gender Bias</h4>
                        <div class="score">{bias.gender_bias.bias_score:.2f}/1.0</div>
                        <p style="font-size: 0.9em; color: #666; margin-top: 5px;">
                            Male: {bias.gender_bias.pronoun_counts.get('male', 0)},
                            Female: {bias.gender_bias.pronoun_counts.get('female', 0)}
                        </p>
                    </div>
                    <div class="bias-item">
                        <h4>Cultural Bias</h4>
                        <div class="score">{bias.cultural_bias.bias_score:.2f}/1.0</div>
                        <p style="font-size: 0.9em; color: #666; margin-top: 5px;">
                            Language Diversity: {bias.cultural_bias.language_diversity:.1%}
                        </p>
                    </div>
                    <div class="bias-item">
                        <h4>Technical Bias</h4>
                        <div class="score">{bias.technical_bias.bias_score:.2f}/1.0</div>
                        <p style="font-size: 0.9em; color: #666; margin-top: 5px;">
                            Framework Diversity: {bias.technical_bias.framework_diversity:.1%}
                        </p>
                    </div>
                </div>

                <div style="margin-top: 30px;">
                    <h3 style="margin-bottom: 15px; color: #333;">Bias Recommendations</h3>
                    <div class="recommendations">
                        <ul>
                            {''.join(f'<li>{rec}</li>' for rec in bias.recommendations)}
                        </ul>
                    </div>
                </div>
            </div>

            <!-- Improvement Recommendations -->
            <div class="section">
                <h2 class="section-title">🚀 Improvement Opportunities</h2>
                <div class="recommendations">
                    <ul>
                        {''.join(f'<li>{rec}</li>' for rec in report.improvement_opportunities)}
                    </ul>
                </div>
            </div>
        </div>

        <footer>
            <p>Generated from: {', '.join(str(p) for p in dataset_paths)}</p>
            <p style="margin-top: 10px; opacity: 0.7;">
                Domain: {domain} | Timestamp: {report.analysis_timestamp}
            </p>
        </footer>
    </div>

    <script>
        // Quality distribution chart
        const ctx = document.getElementById('qualityChart').getContext('2d');
        const qualityChart = new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(chart_labels)},
                datasets: [{{
                    label: 'Number of Samples',
                    data: {json.dumps(chart_values)},
                    backgroundColor: [
                        'rgba(255, 107, 107, 0.6)',
                        'rgba(255, 193, 7, 0.6)',
                        'rgba(156, 39, 176, 0.6)',
                        'rgba(76, 175, 80, 0.6)',
                        'rgba(132, 250, 176, 0.6)',
                    ],
                    borderColor: [
                        'rgba(255, 107, 107, 1)',
                        'rgba(255, 193, 7, 1)',
                        'rgba(156, 39, 176, 1)',
                        'rgba(76, 175, 80, 1)',
                        'rgba(132, 250, 176, 1)',
                    ],
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: true
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    return html


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML quality report for datasets",
    )

    parser.add_argument(
        "datasets",
        nargs="+",
        help="Path(s) to dataset file(s)",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output path for HTML report",
    )
    parser.add_argument(
        "-d", "--domain",
        choices=["general", "assembly", "code"],
        default="general",
        help="Domain type (default: general)",
    )

    args = parser.parse_args()

    # Expand paths
    dataset_paths = []
    for pattern in args.datasets:
        path = Path(pattern)
        if "*" in pattern:
            dataset_paths.extend(sorted(Path(pattern).parent.glob(Path(pattern).name)))
        else:
            dataset_paths.append(path)

    generate_html_report(
        dataset_paths,
        Path(args.output),
        domain=args.domain,
    )

    print(f"✅ HTML report generated: {args.output}")


if __name__ == "__main__":
    main()
