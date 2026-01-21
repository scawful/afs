#!/usr/bin/env python3
"""Generate leaderboard from benchmark results.

Usage:
    python scripts/generate_leaderboard.py --results-dir benchmarks/results --html leaderboard.html
"""

import argparse
import json
from datetime import datetime
from pathlib import Path


def load_all_results(results_dir: Path) -> list[dict]:
    """Load all benchmark results from directory."""
    results = []
    for json_file in results_dir.glob("**/*.json"):
        if json_file.name == "leaderboard.json":
            continue
        try:
            data = json.loads(json_file.read_text())
            if "model_name" in data:
                results.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
    return results


def generate_html_leaderboard(results: list[dict], output_path: Path):
    """Generate HTML leaderboard with ranking tables."""
    # Sort by different metrics
    speed_sorted = sorted(
        [r for r in results if "speed" in r],
        key=lambda r: r["speed"]["throughput"]["tokens_per_second"],
        reverse=True,
    )
    quality_sorted = sorted(
        [r for r in results if "quality" in r],
        key=lambda r: r["quality"]["accuracy"]["accuracy"],
        reverse=True,
    )
    memory_sorted = sorted(
        [r for r in results if "resources" in r],
        key=lambda r: r["resources"]["memory"]["peak_rss_mb"],
        reverse=False,  # Lower is better
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AFS Model Leaderboard</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #007AFF;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #555;
            margin-top: 40px;
            margin-bottom: 20px;
        }}
        .timestamp {{
            color: #999;
            font-size: 14px;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }}
        thead {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        th {{
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #f0f0f0;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .rank {{
            font-weight: bold;
            font-size: 18px;
            color: #007AFF;
        }}
        .rank-1 {{
            color: #FFD700;
        }}
        .rank-2 {{
            color: #C0C0C0;
        }}
        .rank-3 {{
            color: #CD7F32;
        }}
        .model-name {{
            font-weight: 600;
            color: #333;
        }}
        .metric-value {{
            font-family: 'Monaco', 'Courier New', monospace;
            color: #555;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            margin-left: 8px;
        }}
        .badge-new {{
            background: #28a745;
            color: white;
        }}
        .badge-baseline {{
            background: #6c757d;
            color: white;
        }}
        .tabs {{
            display: flex;
            gap: 10px;
            margin: 30px 0 20px 0;
            border-bottom: 2px solid #e0e0e0;
        }}
        .tab {{
            padding: 10px 20px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 16px;
            color: #666;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
        }}
        .tab:hover {{
            color: #007AFF;
        }}
        .tab.active {{
            color: #007AFF;
            border-bottom-color: #007AFF;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
    </style>
    <script>
        function showTab(tabName) {{
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.classList.remove('active');
            }});
            document.querySelectorAll('.tab-content').forEach(content => {{
                content.classList.remove('active');
            }});

            document.querySelector(`[data-tab="${{tabName}}"]`).classList.add('active');
            document.getElementById(tabName).classList.add('active');
        }}
    </script>
</head>
<body>
    <div class="container">
        <h1>🏆 AFS Model Leaderboard</h1>
        <p class="timestamp">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Total models benchmarked: <strong>{len(results)}</strong></p>

        <div class="tabs">
            <button class="tab active" data-tab="speed" onclick="showTab('speed')">⚡ Speed</button>
            <button class="tab" data-tab="quality" onclick="showTab('quality')">✨ Quality</button>
            <button class="tab" data-tab="efficiency" onclick="showTab('efficiency')">💾 Efficiency</button>
        </div>

        <div id="speed" class="tab-content active">
            <h2>Speed Leaderboard</h2>
            <p>Ranked by tokens/second generation rate</p>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>Tokens/Second</th>
                        <th>TTFT (ms)</th>
                        <th>Latency/Token (ms)</th>
                        <th>Date</th>
                    </tr>
                </thead>
                <tbody>
"""

    for i, result in enumerate(speed_sorted[:20], 1):
        rank_class = f"rank-{i}" if i <= 3 else "rank"
        model_name = result["model_name"]
        speed = result["speed"]
        date = result.get("timestamp", "")[:10]

        html += f"""
                    <tr>
                        <td class="{rank_class}">#{i}</td>
                        <td class="model-name">{model_name}</td>
                        <td class="metric-value">{speed['throughput']['tokens_per_second']:.2f}</td>
                        <td class="metric-value">{speed['latency']['time_to_first_token_ms']:.1f}</td>
                        <td class="metric-value">{speed['latency']['latency_per_token_ms']:.2f}</td>
                        <td>{date}</td>
                    </tr>
"""

    html += """
                </tbody>
            </table>
        </div>

        <div id="quality" class="tab-content">
            <h2>Quality Leaderboard</h2>
            <p>Ranked by accuracy on test datasets</p>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>Consistency</th>
                        <th>Code Correctness</th>
                        <th>Date</th>
                    </tr>
                </thead>
                <tbody>
"""

    for i, result in enumerate(quality_sorted[:20], 1):
        rank_class = f"rank-{i}" if i <= 3 else "rank"
        model_name = result["model_name"]
        quality = result["quality"]
        date = result.get("timestamp", "")[:10]

        accuracy = quality["accuracy"]["accuracy"] * 100
        consistency = quality["consistency"]["consistency_score"]
        code_correctness = quality.get("code_correctness", {}).get("compilation_rate", 0) * 100

        html += f"""
                    <tr>
                        <td class="{rank_class}">#{i}</td>
                        <td class="model-name">{model_name}</td>
                        <td class="metric-value">{accuracy:.1f}%</td>
                        <td class="metric-value">{consistency:.3f}</td>
                        <td class="metric-value">{code_correctness:.1f}%</td>
                        <td>{date}</td>
                    </tr>
"""

    html += """
                </tbody>
            </table>
        </div>

        <div id="efficiency" class="tab-content">
            <h2>Resource Efficiency Leaderboard</h2>
            <p>Ranked by memory efficiency (lower is better)</p>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>Peak Memory (MB)</th>
                        <th>Avg Memory (MB)</th>
                        <th>Peak CPU (%)</th>
                        <th>Date</th>
                    </tr>
                </thead>
                <tbody>
"""

    for i, result in enumerate(memory_sorted[:20], 1):
        rank_class = f"rank-{i}" if i <= 3 else "rank"
        model_name = result["model_name"]
        resources = result["resources"]
        date = result.get("timestamp", "")[:10]

        peak_mem = resources["memory"]["peak_rss_mb"]
        avg_mem = resources["memory"]["average_rss_mb"]
        peak_cpu = resources["cpu"]["peak_percent"]

        html += f"""
                    <tr>
                        <td class="{rank_class}">#{i}</td>
                        <td class="model-name">{model_name}</td>
                        <td class="metric-value">{peak_mem:.1f}</td>
                        <td class="metric-value">{avg_mem:.1f}</td>
                        <td class="metric-value">{peak_cpu:.1f}</td>
                        <td>{date}</td>
                    </tr>
"""

    html += """
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""

    output_path.write_text(html)
    print(f"Leaderboard generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate leaderboard from benchmark results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmarks/results"),
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=Path("benchmarks/leaderboard.html"),
        help="Output HTML file",
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.results_dir}...")
    results = load_all_results(args.results_dir)
    print(f"Loaded {len(results)} benchmark results")

    # Generate HTML
    generate_html_leaderboard(results, args.html)


if __name__ == "__main__":
    main()
