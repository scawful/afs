#!/usr/bin/env python3
"""META-CIRCULAR EVALUATION - Models Evaluating Models.

Uses existing trained models to evaluate new models' outputs.
Creates a feedback loop where models improve by learning from each other's evaluations.

This implements a Lambda-calculus-inspired eval loop where:
1. Model A generates answer
2. Model B evaluates quality
3. Model C validates evaluation
4. Results feed back into training data

Usage:
    # Use Veran to evaluate Majora's outputs
    python3 scripts/meta_circular_eval.py \
        --model-under-test majora-v1-Q8_0.gguf \
        --evaluator-model veran-v4-Q8_0.gguf \
        --validator-model nayru-v9-Q8_0.gguf \
        --eval evaluations/majora_v1_oracle_eval.jsonl \
        --output evaluations/meta_results/

    # Self-evaluation (model evaluates itself)
    python3 scripts/meta_circular_eval.py \
        --model-under-test majora-v1-Q8_0.gguf \
        --self-eval \
        --output evaluations/self_eval/
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
class EvaluationPrompt:
    """Prompt for model evaluation."""
    question: str
    expected_answer: str
    model_answer: str
    category: str
    difficulty: str


@dataclass
class MetaEvaluation:
    """Meta-evaluation result from evaluator model."""
    score: float  # 0.0 to 1.0
    reasoning: str
    strengths: list[str]
    weaknesses: list[str]
    suggested_improvement: str


def query_model(model_path: Path, prompt: str, max_tokens: int = 500) -> str:
    """Query model via llama.cpp."""
    try:
        result = subprocess.run(
            [
                "llama-cli",
                "-m", str(model_path),
                "-p", prompt,
                "-n", str(max_tokens),
                "--temp", "0.1",
                "--no-display-prompt"
            ],
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"


def create_evaluation_prompt(eval_prompt: EvaluationPrompt) -> str:
    """Create prompt for evaluator model."""
    return f"""You are an expert evaluator for technical questions about SNES ROM hacking and assembly.

Evaluate the following answer:

Question: {eval_prompt.question}
Expected Answer: {eval_prompt.expected_answer}
Model's Answer: {eval_prompt.model_answer}

Category: {eval_prompt.category}
Difficulty: {eval_prompt.difficulty}

Provide your evaluation in the following format:
SCORE: [0.0 to 1.0]
REASONING: [Why you gave this score]
STRENGTHS: [What the model got right]
WEAKNESSES: [What the model got wrong or missed]
IMPROVEMENT: [How the answer could be improved]

Evaluation:"""


def parse_meta_evaluation(response: str) -> MetaEvaluation:
    """Parse evaluator model's response into structured data."""
    lines = response.split('\n')

    score = 0.5
    reasoning = ""
    strengths = []
    weaknesses = []
    improvement = ""

    current_section = None

    for line in lines:
        line = line.strip()

        if line.startswith("SCORE:"):
            try:
                score_text = line.split(":", 1)[1].strip()
                score = float(score_text)
            except:
                pass

        elif line.startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip() if ":" in line else ""
            current_section = "reasoning"

        elif line.startswith("STRENGTHS:"):
            strengths_text = line.split(":", 1)[1].strip() if ":" in line else ""
            if strengths_text:
                strengths = [strengths_text]
            current_section = "strengths"

        elif line.startswith("WEAKNESSES:"):
            weaknesses_text = line.split(":", 1)[1].strip() if ":" in line else ""
            if weaknesses_text:
                weaknesses = [weaknesses_text]
            current_section = "weaknesses"

        elif line.startswith("IMPROVEMENT:"):
            improvement = line.split(":", 1)[1].strip() if ":" in line else ""
            current_section = "improvement"

        elif line and current_section:
            # Continue current section
            if current_section == "reasoning":
                reasoning += " " + line
            elif current_section == "strengths":
                strengths.append(line)
            elif current_section == "weaknesses":
                weaknesses.append(line)
            elif current_section == "improvement":
                improvement += " " + line

    return MetaEvaluation(
        score=score,
        reasoning=reasoning.strip(),
        strengths=[s.strip() for s in strengths if s.strip()],
        weaknesses=[w.strip() for w in weaknesses if w.strip()],
        suggested_improvement=improvement.strip()
    )


def validate_evaluation(
    original_question: str,
    model_answer: str,
    meta_eval: MetaEvaluation,
    validator_model: Path
) -> dict[str, Any]:
    """Use validator model to check if evaluation is reasonable."""

    prompt = f"""You are validating another model's evaluation.

Original Question: {original_question}
Model Answer: {model_answer}
Evaluator's Score: {meta_eval.score}
Evaluator's Reasoning: {meta_eval.reasoning}

Is this evaluation reasonable and accurate? Reply with:
VALID: [yes/no]
CONFIDENCE: [0.0 to 1.0]
NOTES: [Any concerns or observations]

Validation:"""

    response = query_model(validator_model, prompt, max_tokens=200)

    # Parse validation
    valid = "yes" in response.lower()
    confidence = 0.5

    try:
        for line in response.split('\n'):
            if "CONFIDENCE:" in line:
                conf_text = line.split(":", 1)[1].strip()
                confidence = float(conf_text)
    except:
        pass

    return {
        "valid": valid,
        "confidence": confidence,
        "notes": response
    }


def run_meta_circular_eval(
    model_under_test: Path,
    evaluator_model: Path,
    validator_model: Path | None,
    eval_questions: list[dict],
    output_dir: Path,
    self_eval: bool = False
) -> dict[str, Any]:
    """Run full meta-circular evaluation."""

    print(f"\n{'=' * 60}")
    print("META-CIRCULAR EVALUATION")
    print(f"{'=' * 60}")
    print(f"\nModel Under Test: {model_under_test.name}")
    print(f"Evaluator: {evaluator_model.name if not self_eval else 'SELF'}")
    if validator_model:
        print(f"Validator: {validator_model.name}")
    print(f"Questions: {len(eval_questions)}\n")

    results = []

    for i, q in enumerate(eval_questions, 1):
        print(f"[{i}/{len(eval_questions)}] {q['category']}")

        # Step 1: Get model's answer
        prompt = f"Question: {q['question']}\n\nAnswer:"
        model_answer = query_model(model_under_test, prompt)

        # Step 2: Get evaluation from evaluator model
        eval_prompt = EvaluationPrompt(
            question=q['question'],
            expected_answer=q['expected_answer'],
            model_answer=model_answer,
            category=q['category'],
            difficulty=q['difficulty']
        )

        evaluator = model_under_test if self_eval else evaluator_model
        eval_text = query_model(evaluator, create_evaluation_prompt(eval_prompt), max_tokens=800)
        meta_eval = parse_meta_evaluation(eval_text)

        print(f"  Score: {meta_eval.score:.2f}")
        print(f"  Reasoning: {meta_eval.reasoning[:80]}...")

        # Step 3: Validate evaluation (if validator provided)
        validation = None
        if validator_model:
            validation = validate_evaluation(
                q['question'],
                model_answer,
                meta_eval,
                validator_model
            )
            print(f"  Validation: {'✓' if validation['valid'] else '✗'} (confidence: {validation['confidence']:.2f})")

        result = {
            "question": q['question'],
            "expected_answer": q['expected_answer'],
            "model_answer": model_answer,
            "category": q['category'],
            "difficulty": q['difficulty'],
            "meta_evaluation": {
                "score": meta_eval.score,
                "reasoning": meta_eval.reasoning,
                "strengths": meta_eval.strengths,
                "weaknesses": meta_eval.weaknesses,
                "improvement": meta_eval.suggested_improvement
            },
            "validation": validation
        }

        results.append(result)
        print()

    # Generate summary
    avg_score = sum(r['meta_evaluation']['score'] for r in results) / len(results)

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"\nAverage Score: {avg_score:.2%}")

    if validator_model:
        valid_count = sum(1 for r in results if r['validation'] and r['validation']['valid'])
        print(f"Valid Evaluations: {valid_count}/{len(results)}")

    # Save results
    output_path = output_dir / "meta_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "model_under_test": str(model_under_test),
        "evaluator_model": str(evaluator_model) if not self_eval else "SELF",
        "validator_model": str(validator_model) if validator_model else None,
        "timestamp": datetime.now().isoformat(),
        "self_evaluation": self_eval,
        "summary": {
            "average_score": avg_score,
            "total_questions": len(results),
            "by_category": {},
            "by_difficulty": {}
        },
        "results": results
    }

    # Calculate by category/difficulty
    by_cat = {}
    by_diff = {}
    for r in results:
        cat = r['category']
        diff = r['difficulty']

        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(r['meta_evaluation']['score'])

        if diff not in by_diff:
            by_diff[diff] = []
        by_diff[diff].append(r['meta_evaluation']['score'])

    output_data['summary']['by_category'] = {k: sum(v)/len(v) for k, v in by_cat.items()}
    output_data['summary']['by_difficulty'] = {k: sum(v)/len(v) for k, v in by_diff.items()}

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Generate training samples from evaluations
    training_samples = generate_training_from_evals(results)
    training_path = output_dir / "training_samples.jsonl"

    with open(training_path, 'w') as f:
        for sample in training_samples:
            f.write(json.dumps(sample) + '\n')

    print(f"Training samples: {training_path}")
    print(f"  Generated {len(training_samples)} new training samples from evaluations")

    return output_data


def generate_training_from_evals(results: list[dict]) -> list[dict]:
    """Generate training samples from evaluation results.

    Creates new training samples incorporating:
    - High-quality model answers
    - Improvement suggestions
    - Common weaknesses to avoid
    """
    samples = []

    for r in results:
        eval_data = r['meta_evaluation']

        # If answer was good (score >= 0.7), use as positive example
        if eval_data['score'] >= 0.7:
            samples.append({
                "instruction": r['question'],
                "output": r['model_answer'],
                "thinking": f"Evaluation reasoning: {eval_data['reasoning']}",
                "domain": r['category'],
                "source": "meta_eval_positive",
                "quality_score": eval_data['score']
            })

        # If answer had issues but improvement suggested, create corrected version
        if eval_data['improvement'] and eval_data['score'] < 0.7:
            samples.append({
                "instruction": r['question'],
                "output": eval_data['improvement'],
                "thinking": f"Improved version. Original weaknesses: {', '.join(eval_data['weaknesses'])}",
                "domain": r['category'],
                "source": "meta_eval_corrected",
                "quality_score": 0.8  # Assume improvement is good
            })

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Meta-circular evaluation - models evaluating models"
    )

    parser.add_argument(
        "--model-under-test",
        type=Path,
        required=True,
        help="Model to evaluate"
    )
    parser.add_argument(
        "--evaluator-model",
        type=Path,
        help="Model to perform evaluation (if not self-eval)"
    )
    parser.add_argument(
        "--validator-model",
        type=Path,
        help="Model to validate evaluations (optional)"
    )
    parser.add_argument(
        "--self-eval",
        action="store_true",
        help="Use model to evaluate itself"
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
        help="Output directory"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.self_eval and not args.evaluator_model:
        print("Error: Must specify --evaluator-model or --self-eval")
        return 1

    # Load questions
    eval_path = args.eval.expanduser().resolve()
    questions = []
    with open(eval_path, 'r') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    # Run evaluation
    model_under_test = args.model_under_test.expanduser().resolve()
    evaluator = args.evaluator_model.expanduser().resolve() if args.evaluator_model else None
    validator = args.validator_model.expanduser().resolve() if args.validator_model else None
    output_dir = args.output.expanduser().resolve()

    run_meta_circular_eval(
        model_under_test,
        evaluator or model_under_test,
        validator,
        questions,
        output_dir,
        self_eval=args.self_eval
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
