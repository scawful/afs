#!/usr/bin/env python3
"""
Meta-Circular Evaluation System

Uses completed models to evaluate each other's outputs, creating a
feedback loop for training data generation.
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict
import statistics

# AFS project root
AFS_ROOT = Path(__file__).parent.parent
EVAL_SUITE = Path.home() / ".context/training/evals/unified_eval_suite.jsonl"
RESULTS_DIR = Path.home() / ".context/training/evals/meta_circular"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Evaluator models (used to score responses)
EVALUATORS = {
    "majora": {
        "name": "Majora v1 (Domain Expert)",
        "endpoint": "http://localhost:5000/chat",
        "traits": ["comprehensive", "quest_aware"],
        "reliability": 0.85
    },
    "veran": {
        "name": "Veran v5 (Logic Specialist)",
        "endpoint": "http://localhost:5002/chat",
        "traits": ["rigorous", "logical"],
        "reliability": 0.90
    }
}

# Models being evaluated
TARGET_MODELS = {
    "nayru": "http://localhost:5001/chat",
    "agahnim": "http://localhost:5003/chat",
    "hylia": "http://localhost:5004/chat"
}


class MetaCircularEvaluator:
    """Evaluate models using other models."""

    def __init__(self):
        self.results = defaultdict(lambda: defaultdict(list))
        self.reliability_scores = {}

    def query_model(self, endpoint: str, prompt: str, timeout: int = 30) -> str:
        """Query a model endpoint."""
        import requests
        try:
            response = requests.post(
                endpoint,
                json={"prompt": prompt},
                timeout=timeout
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            return ""
        except:
            return ""

    def create_evaluation_prompt(self, evaluator_name: str, question: Dict,
                                 response: str) -> str:
        """Create a prompt for an evaluator model to judge a response."""
        category = question.get("category", "general")
        difficulty = question.get("difficulty", "unknown")

        prompt = f"""You are {EVALUATORS[evaluator_name]['name']}, evaluating responses
to test questions.

QUESTION ({category}, {difficulty}):
{question.get('prompt') or question.get('question')}

RESPONSE TO EVALUATE:
{response}

EVALUATION CRITERIA:
1. Correctness: Does it answer the question accurately?
2. Completeness: Are all relevant aspects covered?
3. Clarity: Is the explanation clear and well-structured?
4. Relevance: Are there extraneous or off-topic elements?

Provide a score from 0-10 and brief justification.

Format your response as:
SCORE: <0-10>
JUSTIFICATION: <brief explanation>
"""
        return prompt

    def extract_score(self, evaluation_response: str) -> float:
        """Extract numeric score from evaluator response."""
        try:
            lines = evaluation_response.split('\n')
            for line in lines:
                if line.startswith("SCORE:"):
                    score_text = line.replace("SCORE:", "").strip()
                    score = float(score_text.split()[0])
                    return min(10, max(0, score)) / 10.0
        except:
            pass
        return 0.5  # Default middle score

    def evaluate_response(self, question: Dict, response: str,
                         evaluator_name: str) -> Dict:
        """Have one model evaluate another's response."""
        evaluator_endpoint = EVALUATORS[evaluator_name]["endpoint"]

        prompt = self.create_evaluation_prompt(evaluator_name, question, response)
        evaluation = self.query_model(evaluator_endpoint, prompt)

        score = self.extract_score(evaluation)
        reliability = EVALUATORS[evaluator_name]["reliability"]

        return {
            "evaluator": evaluator_name,
            "score": score,
            "raw_evaluation": evaluation,
            "adjusted_score": score * reliability
        }

    def run_meta_circular_eval(self, target_model: str,
                              questions: List[Dict],
                              sample_size: int = None) -> Dict:
        """Evaluate a target model using multiple evaluators."""
        if sample_size:
            questions = questions[:sample_size]

        print(f"\nMeta-Circular Evaluation: {target_model}")
        print("=" * 60)

        endpoint = TARGET_MODELS.get(target_model)
        if not endpoint:
            print(f"ERROR: Unknown model {target_model}")
            return {}

        results = {
            "model": target_model,
            "questions_evaluated": len(questions),
            "evaluations": [],
            "summary": {}
        }

        for i, question in enumerate(questions):
            print(f"[{i+1}/{len(questions)}] {question['id']} ... ", end="", flush=True)

            prompt = question.get("prompt") or question.get("question")
            response = self.query_model(endpoint, prompt)

            if not response:
                print("SKIP (no response)")
                continue

            # Get evaluations from multiple evaluators
            evaluations = []
            for evaluator_name in EVALUATORS.keys():
                eval_result = self.evaluate_response(question, response, evaluator_name)
                evaluations.append(eval_result)

            # Average scores
            avg_score = statistics.mean([e["adjusted_score"] for e in evaluations])

            results["evaluations"].append({
                "question_id": question["id"],
                "category": question.get("category"),
                "response": response[:500] + "..." if len(response) > 500 else response,
                "evaluations": evaluations,
                "average_score": avg_score
            })

            print(f"✓ {avg_score:.2f}")

        # Calculate summary statistics
        if results["evaluations"]:
            scores = [e["average_score"] for e in results["evaluations"]]
            results["summary"] = {
                "avg_score": statistics.mean(scores),
                "median_score": statistics.median(scores),
                "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
                "min_score": min(scores),
                "max_score": max(scores)
            }

        return results

    def generate_training_data(self, evaluation_results: List[Dict]) -> List[Dict]:
        """Convert evaluation results into training examples."""
        training_data = []

        for model_results in evaluation_results:
            model_name = model_results["model"]

            for eval_item in model_results["evaluations"]:
                # Create training example with evaluator feedback
                for evaluation in eval_item["evaluations"]:
                    training_data.append({
                        "model": model_name,
                        "evaluator": evaluation["evaluator"],
                        "question": eval_item["question_id"],
                        "response": eval_item["response"],
                        "evaluation": evaluation["raw_evaluation"],
                        "score": evaluation["score"],
                        "adjusted_score": evaluation["adjusted_score"],
                        "timestamp": datetime.now().isoformat()
                    })

        return training_data

    def create_report(self, evaluation_results: List[Dict]) -> str:
        """Create markdown report of meta-circular evaluation."""
        report = "# Meta-Circular Evaluation Report\n\n"
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Summary table
        report += "## Summary\n\n"
        report += "| Model | Avg Score | Median | Std Dev | Range |\n"
        report += "|-------|-----------|--------|---------|-------|\n"

        for results in evaluation_results:
            summary = results["summary"]
            if summary:
                report += (f"| {results['model']} | "
                          f"{summary.get('avg_score', 0):.3f} | "
                          f"{summary.get('median_score', 0):.3f} | "
                          f"{summary.get('std_dev', 0):.3f} | "
                          f"{summary.get('min_score', 0):.2f}-{summary.get('max_score', 0):.2f} |\n")

        report += "\n## Detailed Results\n\n"

        for results in evaluation_results:
            report += f"### {results['model']}\n\n"
            report += f"**Total Questions Evaluated:** {results['questions_evaluated']}\n"

            if results["summary"]:
                summary = results["summary"]
                report += f"**Average Score:** {summary.get('avg_score', 0):.3f}\n"
                report += f"**Median Score:** {summary.get('median_score', 0):.3f}\n\n"

            # Top and bottom evaluations
            if results["evaluations"]:
                sorted_evals = sorted(results["evaluations"],
                                    key=lambda x: x["average_score"], reverse=True)

                report += f"**Best Performing:**\n"
                for eval_item in sorted_evals[:3]:
                    report += f"- {eval_item['question_id']}: {eval_item['average_score']:.2f}\n"

                report += f"\n**Worst Performing:**\n"
                for eval_item in sorted_evals[-3:]:
                    report += f"- {eval_item['question_id']}: {eval_item['average_score']:.2f}\n"

            report += "\n"

        return report


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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Meta-circular model evaluation")
    parser.add_argument("--models", nargs="+", default=list(TARGET_MODELS.keys()),
                       help="Models to evaluate")
    parser.add_argument("--sample-size", type=int, default=5,
                       help="Questions per model (default: 5)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (default: ~/.context/training/evals/meta_circular)")
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Meta-Circular Evaluation System")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Sample size: {args.sample_size} questions per model\n")

    # Load evaluation suite
    print("Loading evaluation suite...")
    questions = load_eval_suite()
    print(f"Loaded {len(questions)} questions\n")

    # Run evaluations
    evaluator = MetaCircularEvaluator()
    all_results = []

    for model_name in args.models:
        if model_name not in TARGET_MODELS:
            print(f"WARNING: Unknown model {model_name}")
            continue

        results = evaluator.run_meta_circular_eval(model_name, questions,
                                                   sample_size=args.sample_size)
        if results:
            all_results.append(results)

    # Generate reports
    print("\n" + "=" * 60)
    print("GENERATING REPORTS")
    print("=" * 60 + "\n")

    # Markdown report
    report = evaluator.create_report(all_results)
    report_path = output_dir / f"meta_circular_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report: {report_path}")

    # JSON results
    json_path = output_dir / f"meta_circular_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results: {json_path}")

    # Training data
    training_data = evaluator.generate_training_data(all_results)
    training_path = output_dir / f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with open(training_path, 'w') as f:
        for item in training_data:
            f.write(json.dumps(item) + '\n')
    print(f"Training data: {training_path}")

    print(f"\n{len(training_data)} training examples generated")


if __name__ == "__main__":
    main()
