#!/usr/bin/env python3
"""Screenshot-based evaluation for visual validation of model outputs.

Captures screenshots of model responses in terminal/browser and compares outputs visually.

Usage:
    python3 scripts/screenshot_eval.py \
        --model majora-v1-Q8_0.gguf \
        --eval evaluations/majora_v1_oracle_eval.jsonl \
        --output evaluations/screenshots/majora_v1/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime


def capture_terminal_screenshot(
    command: str,
    output_path: Path,
    wait_time: float = 2.0
) -> bool:
    """Capture screenshot of terminal after running command.

    Uses screencapture on macOS.
    """
    try:
        # Run command in background
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for command to produce output
        time.sleep(wait_time)

        # Capture screenshot (macOS)
        subprocess.run([
            "screencapture",
            "-x",  # No sound
            "-C",  # Capture cursor
            str(output_path)
        ], check=True)

        # Terminate command if still running
        proc.terminate()

        return True

    except Exception as e:
        print(f"Error capturing screenshot: {e}")
        return False


def run_model_with_screenshot(
    model_path: Path,
    prompt: str,
    screenshot_path: Path,
    max_tokens: int = 500
) -> tuple[str, bool]:
    """Run model query and capture screenshot of output.

    Returns:
        Tuple of (response text, screenshot success)
    """
    # Build llama.cpp command
    cmd = f"""
    llama-cli \
        -m {model_path} \
        -p "{prompt}" \
        -n {max_tokens} \
        --temp 0.1 \
        --color
    """

    # Capture output and screenshot
    try:
        # Run in terminal emulator for visual capture
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )

        response = result.stdout.strip()

        # Also capture as screenshot if available
        screenshot_success = False
        try:
            # Save output to temp file
            temp_file = screenshot_path.parent / f"temp_{screenshot_path.stem}.txt"
            temp_file.write_text(f"Question: {prompt}\n\nResponse:\n{response}")

            # Open in terminal and screenshot
            # (This is a simplified approach - real implementation would use iTerm2 API or similar)
            screenshot_success = capture_terminal_screenshot(
                f"cat {temp_file}",
                screenshot_path
            )

            temp_file.unlink()
        except:
            pass

        return response, screenshot_success

    except Exception as e:
        return f"Error: {e}", False


def create_comparison_html(
    results: list[dict],
    output_path: Path
):
    """Create HTML comparison page with screenshots."""

    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Model Evaluation Results</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            margin: 20px;
            background: #1e1e1e;
            color: #d4d4d4;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            color: #4ec9b0;
            border-bottom: 2px solid #4ec9b0;
            padding-bottom: 10px;
        }
        .result {
            background: #252526;
            border: 1px solid #3e3e42;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        .question {
            font-size: 18px;
            font-weight: bold;
            color: #569cd6;
            margin-bottom: 10px;
        }
        .category {
            display: inline-block;
            background: #264f78;
            color: #9cdcfe;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 12px;
            margin-right: 10px;
        }
        .difficulty {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 12px;
        }
        .easy { background: #106b1e; color: #4ec9b0; }
        .medium { background: #8b6914; color: #dcdcaa; }
        .hard { background: #6b1014; color: #f48771; }
        .screenshot {
            margin: 15px 0;
            border: 2px solid #3e3e42;
            border-radius: 4px;
            max-width: 100%;
        }
        .score {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }
        .score-high { color: #4ec9b0; }
        .score-medium { color: #dcdcaa; }
        .score-low { color: #f48771; }
        .expected {
            background: #1e1e1e;
            border-left: 3px solid #4ec9b0;
            padding: 10px;
            margin: 10px 0;
            font-family: 'Consolas', 'Monaco', monospace;
        }
        .response {
            background: #1e1e1e;
            border-left: 3px solid #569cd6;
            padding: 10px;
            margin: 10px 0;
            font-family: 'Consolas', 'Monaco', monospace;
            white-space: pre-wrap;
        }
        .summary {
            background: #252526;
            border: 2px solid #4ec9b0;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        .stat {
            display: inline-block;
            margin: 10px 20px 10px 0;
        }
        .stat-value {
            font-size: 32px;
            font-weight: bold;
            color: #4ec9b0;
        }
        .stat-label {
            font-size: 14px;
            color: #858585;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Model Evaluation Results</h1>
"""

    # Add summary
    total_score = sum(r['score'] for r in results) / len(results) if results else 0
    html += f"""
        <div class="summary">
            <div class="stat">
                <div class="stat-value">{total_score:.1%}</div>
                <div class="stat-label">Overall Score</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(results)}</div>
                <div class="stat-label">Questions</div>
            </div>
            <div class="stat">
                <div class="stat-value">{sum(1 for r in results if r['score'] >= 0.7)}</div>
                <div class="stat-label">Passed (≥70%)</div>
            </div>
        </div>
"""

    # Add results
    for i, result in enumerate(results, 1):
        score = result['score']
        score_class = 'score-high' if score >= 0.7 else 'score-medium' if score >= 0.4 else 'score-low'

        html += f"""
        <div class="result">
            <div class="question">Question {i}: {result['question']}</div>
            <span class="category">{result['category']}</span>
            <span class="difficulty {result['difficulty']}">{result['difficulty']}</span>

            <div class="score {score_class}">Score: {score:.1%}</div>
"""

        # Add screenshot if available
        if result.get('screenshot_path'):
            rel_path = Path(result['screenshot_path']).name
            html += f'<img src="{rel_path}" class="screenshot" alt="Model output screenshot">\n'

        html += f"""
            <div class="expected">
                <strong>Expected:</strong><br>
                {result['expected_answer']}
            </div>

            <div class="response">
                <strong>Model Response:</strong><br>
                {result['model_answer']}
            </div>

            <p><em>{result['notes']}</em></p>
        </div>
"""

    html += """
    </div>
</body>
</html>
"""

    output_path.write_text(html)
    print(f"\n✓ Created comparison page: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Screenshot-based evaluation with visual validation"
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Model file (GGUF)"
    )
    parser.add_argument(
        "--eval",
        type=Path,
        required=True,
        help="Evaluation questions (JSONL)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for screenshots"
    )
    parser.add_argument(
        "--skip-screenshots",
        action="store_true",
        help="Skip screenshot capture (faster)"
    )

    args = parser.parse_args()

    # Setup paths
    model_path = args.model.expanduser().resolve()
    eval_path = args.eval.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Screenshot-Based Evaluation")
    print(f"{'=' * 60}")
    print(f"\nModel: {model_path.name}")
    print(f"Output: {output_dir}")

    # Load questions
    questions = []
    with open(eval_path, 'r') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    print(f"Questions: {len(questions)}\n")

    # Run evaluation with screenshots
    results = []

    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['category']} ({q['difficulty']})")

        screenshot_path = output_dir / f"q{i:02d}_{q['category']}.png"

        # Query model
        prompt = f"Question: {q['question']}\n\nAnswer:"
        response, screenshot_ok = run_model_with_screenshot(
            model_path,
            prompt,
            screenshot_path if not args.skip_screenshots else output_dir / "dummy.png",
            max_tokens=500
        )

        # Simple scoring (can enhance with embeddings)
        score = 0.5  # Placeholder
        notes = "Visual validation required"

        result = {
            'question': q['question'],
            'model_answer': response,
            'expected_answer': q['expected_answer'],
            'category': q['category'],
            'difficulty': q['difficulty'],
            'score': score,
            'notes': notes,
            'screenshot_path': str(screenshot_path) if screenshot_ok else None
        }

        results.append(result)
        print(f"  Screenshot: {'✓' if screenshot_ok else '✗'}")
        print()

    # Save results JSON
    results_json = output_dir / "results.json"
    with open(results_json, 'w') as f:
        json.dump({
            'model': str(model_path),
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)

    # Create HTML comparison
    html_path = output_dir / "comparison.html"
    create_comparison_html(results, html_path)

    print(f"\n{'=' * 60}")
    print("Evaluation Complete")
    print(f"{'=' * 60}")
    print(f"\nResults: {results_json}")
    print(f"HTML: {html_path}")
    print(f"\nOpen in browser: file://{html_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
