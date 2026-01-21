#!/usr/bin/env python3
"""Run evaluation on trained models.

Evaluates model performance on domain-specific question sets.

Usage:
    # Evaluate Majora v1 on Oracle questions
    python3 scripts/run_eval.py \
        --model majora-v1-Q8_0.gguf \
        --eval evaluations/majora_v1_oracle_eval.jsonl \
        --output evaluations/results/majora_v1_results.json

    # Compare multiple models
    python3 scripts/run_eval.py \
        --models majora-v1-Q8_0.gguf,base-model.gguf \
        --eval evaluations/majora_v1_oracle_eval.jsonl \
        --compare
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class EvalQuestion:
    """Single evaluation question."""
    question: str
    expected_answer: str
    category: str
    difficulty: str


@dataclass
class EvalResult:
    """Result for single question."""
    question: str
    model_answer: str
    expected_answer: str
    category: str
    difficulty: str
    score: float  # 0.0 to 1.0
    notes: str = ""


def load_eval_questions(eval_path: Path) -> list[EvalQuestion]:
    """Load evaluation questions from JSONL."""
    questions = []
    with open(eval_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                questions.append(EvalQuestion(**data))
    return questions


def query_model(model_path: Path, prompt: str, max_tokens: int = 500) -> str:
    """Query model via llama.cpp server.

    Assumes llama.cpp server running on localhost:8080
    """
    import requests

    try:
        response = requests.post(
            "http://localhost:8080/completion",
            json={
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": 0.1,  # Low temp for eval
                "stop": ["\n\n", "Question:", "Q:"],
            },
            timeout=30
        )

        if response.status_code == 200:
            return response.json()["content"].strip()
        else:
            return f"Error: HTTP {response.status_code}"

    except Exception as e:
        return f"Error: {e}"


def score_answer(model_answer: str, expected_answer: str, category: str) -> tuple[float, str]:
    """Score model answer against expected answer.

    Returns:
        Tuple of (score 0.0-1.0, notes)
    """
    # Simple keyword matching for now
    # TODO: Use semantic similarity with embeddings

    model_lower = model_answer.lower()
    expected_lower = expected_answer.lower()

    # Extract key terms from expected answer
    key_terms = [term.strip() for term in expected_lower.split(',')]

    # Count matched terms
    matches = sum(1 for term in key_terms if term in model_lower)

    if not key_terms:
        return 0.5, "No key terms to match"

    score = matches / len(key_terms)
    notes = f"Matched {matches}/{len(key_terms)} key terms"

    # Bonus for correct address format
    if category == "memory_map" and "$" in model_answer and "$" in expected_answer:
        # Extract addresses
        import re
        model_addrs = set(re.findall(r'\$[0-9A-Fa-f]+', model_answer))
        expected_addrs = set(re.findall(r'\$[0-9A-Fa-f]+', expected_answer))

        if model_addrs & expected_addrs:  # Intersection
            score = min(1.0, score + 0.2)
            notes += " (address match bonus)"

    return score, notes


def run_evaluation(
    model_path: Path,
    eval_questions: list[EvalQuestion],
    output_path: Path | None = None
) -> list[EvalResult]:
    """Run full evaluation on model."""

    print(f"\n{'=' * 60}")
    print(f"Evaluating: {model_path.name}")
    print(f"Questions: {len(eval_questions)}")
    print(f"{'=' * 60}\n")

    results = []

    for i, q in enumerate(eval_questions, 1):
        print(f"[{i}/{len(eval_questions)}] {q.category} ({q.difficulty})")

        # Format prompt
        prompt = f"Question: {q.question}\n\nAnswer:"

        # Query model
        model_answer = query_model(model_path, prompt)

        # Score answer
        score, notes = score_answer(model_answer, q.expected_answer, q.category)

        result = EvalResult(
            question=q.question,
            model_answer=model_answer,
            expected_answer=q.expected_answer,
            category=q.category,
            difficulty=q.difficulty,
            score=score,
            notes=notes
        )

        results.append(result)

        # Print result
        status = "✓" if score >= 0.7 else "✗" if score < 0.4 else "~"
        print(f"  {status} Score: {score:.2f} - {notes}")
        print()

    # Print summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}\n")

    total_score = sum(r.score for r in results) / len(results)

    by_category = {}
    for r in results:
        if r.category not in by_category:
            by_category[r.category] = []
        by_category[r.category].append(r.score)

    by_difficulty = {}
    for r in results:
        if r.difficulty not in by_difficulty:
            by_difficulty[r.difficulty] = []
        by_difficulty[r.difficulty].append(r.score)

    print(f"Overall Score: {total_score:.2%}")
    print(f"\nBy Category:")
    for cat, scores in sorted(by_category.items()):
        avg = sum(scores) / len(scores)
        print(f"  {cat}: {avg:.2%} ({len(scores)} questions)")

    print(f"\nBy Difficulty:")
    for diff, scores in sorted(by_difficulty.items()):
        avg = sum(scores) / len(scores)
        print(f"  {diff}: {avg:.2%} ({len(scores)} questions)")

    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "model": str(model_path),
            "timestamp": datetime.now().isoformat(),
            "overall_score": total_score,
            "by_category": {k: sum(v)/len(v) for k, v in by_category.items()},
            "by_difficulty": {k: sum(v)/len(v) for k, v in by_difficulty.items()},
            "results": [
                {
                    "question": r.question,
                    "model_answer": r.model_answer,
                    "expected_answer": r.expected_answer,
                    "category": r.category,
                    "difficulty": r.difficulty,
                    "score": r.score,
                    "notes": r.notes
                }
                for r in results
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation on trained models"
    )

    parser.add_argument(
        "--model",
        type=Path,
        help="Model file to evaluate (GGUF)"
    )
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of models to compare"
    )
    parser.add_argument(
        "--eval",
        type=Path,
        required=True,
        help="Evaluation questions file (JSONL)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output results file (JSON)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple models"
    )

    args = parser.parse_args()

    # Load evaluation questions
    eval_path = args.eval.expanduser().resolve()
    if not eval_path.exists():
        print(f"Error: Evaluation file not found: {eval_path}")
        return 1

    questions = load_eval_questions(eval_path)
    print(f"Loaded {len(questions)} evaluation questions")

    # Single model mode
    if args.model:
        model_path = args.model.expanduser().resolve()
        if not model_path.exists():
            print(f"Error: Model not found: {model_path}")
            return 1

        output_path = args.output.expanduser().resolve() if args.output else None

        run_evaluation(model_path, questions, output_path)

    # Multi-model comparison mode
    elif args.models:
        model_paths = [Path(m.strip()).expanduser().resolve()
                      for m in args.models.split(',')]

        all_results = {}
        for model_path in model_paths:
            if not model_path.exists():
                print(f"Warning: Model not found: {model_path}")
                continue

            results = run_evaluation(model_path, questions)
            all_results[model_path.name] = results

        # Print comparison
        print(f"\n{'=' * 60}")
        print("Model Comparison")
        print(f"{'=' * 60}\n")

        for model_name, results in all_results.items():
            avg_score = sum(r.score for r in results) / len(results)
            print(f"{model_name}: {avg_score:.2%}")

    else:
        print("Error: Specify --model or --models")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
