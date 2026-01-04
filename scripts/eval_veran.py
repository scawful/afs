#!/usr/bin/env python3
"""Evaluate Veran model on code explanation tasks."""

import json
from pathlib import Path

# Test cases with expected key concepts
EVAL_CASES = [
    {
        "code": "LDA #$00\nSTA $10",
        "expected_concepts": ["store", "zero", "immediate", "$10"],
        "difficulty": "basic"
    },
    {
        "code": "LDA $10\nCLC\nADC $11\nSTA $12",
        "expected_concepts": ["add", "carry", "sum"],
        "difficulty": "basic"
    },
    {
        "code": "LDA $10\nASL A\nASL A\nASL A",
        "expected_concepts": ["shift", "multiply", "8"],
        "difficulty": "basic"
    },
    {
        "code": "LDA $10\nEOR #$FF\nINC A",
        "expected_concepts": ["negate", "complement", "two's complement"],
        "difficulty": "intermediate"
    },
    {
        "code": "REP #$20\nLDA $00\nSTA $10\nSEP #$20",
        "expected_concepts": ["16-bit", "mode", "REP", "SEP"],
        "difficulty": "intermediate"
    },
    {
        "code": "PHB\nPHK\nPLB\nJSR Routine\nPLB\nRTL",
        "expected_concepts": ["bank", "data bank", "save", "restore"],
        "difficulty": "intermediate"
    },
    {
        "code": "LDA $10\nASL A\nTAX\nLDA.l Table,X\nSTA $00\nLDA.l Table+1,X\nSTA $01\nJMP ($0000)",
        "expected_concepts": ["jump table", "indirect", "index", "double"],
        "difficulty": "advanced"
    },
    {
        "code": "REP #$30\nLDX #$0000\nLDY #$1000\nLDA #$00FF\nMVN $7E,$7E\nSEP #$30",
        "expected_concepts": ["block", "move", "copy", "MVN", "256"],
        "difficulty": "advanced"
    },
    {
        "code": "LDA #$80\nSTA $2100",
        "expected_concepts": ["screen", "blank", "force blank", "INIDISP"],
        "difficulty": "snes_hardware"
    },
    {
        "code": "LDA #$01\nSTA $420B",
        "expected_concepts": ["DMA", "transfer", "channel"],
        "difficulty": "snes_hardware"
    },
]

def evaluate_response(response: str, expected_concepts: list) -> dict:
    """Check how many expected concepts appear in response."""
    response_lower = response.lower()
    found = []
    missing = []
    
    for concept in expected_concepts:
        if concept.lower() in response_lower:
            found.append(concept)
        else:
            missing.append(concept)
    
    return {
        "found": found,
        "missing": missing,
        "score": len(found) / len(expected_concepts) if expected_concepts else 0
    }

def run_eval():
    """Run evaluation on Veran model."""
    from mlx_lm import load, generate
    
    model_path = Path(__file__).parent.parent / "models" / "veran-lora-fused"
    print(f"Loading model from {model_path}...")
    model, tokenizer = load(str(model_path))
    
    system_prompt = """You are Veran, a 65816 assembly code explanation expert. Given assembly code, explain what it does clearly and concisely.

For each code block:
1. State the PURPOSE (what does it accomplish?)
2. Walk through KEY STEPS (how does it work?)
3. Identify PATTERNS (common idioms if applicable)
4. Note ASSUMPTIONS (register modes, memory state)

Be concise. Focus on understanding, not exhaustive detail."""

    results = {"basic": [], "intermediate": [], "advanced": [], "snes_hardware": []}
    
    for i, case in enumerate(EVAL_CASES):
        print(f"\n{'='*60}")
        print(f"Test {i+1}/{len(EVAL_CASES)} [{case['difficulty']}]")
        print(f"Code:\n{case['code']}")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Explain this 65816 code:\n{case['code']}"}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = generate(model, tokenizer, prompt=prompt, max_tokens=300, verbose=False)
        
        print(f"\nResponse:\n{response}")
        
        eval_result = evaluate_response(response, case["expected_concepts"])
        print(f"\nScore: {eval_result['score']:.0%}")
        print(f"Found: {eval_result['found']}")
        print(f"Missing: {eval_result['missing']}")
        
        results[case["difficulty"]].append({
            "code": case["code"],
            "response": response,
            "score": eval_result["score"],
            "found": eval_result["found"],
            "missing": eval_result["missing"]
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    total_score = 0
    total_tests = 0
    
    for difficulty, cases in results.items():
        if cases:
            avg_score = sum(c["score"] for c in cases) / len(cases)
            total_score += sum(c["score"] for c in cases)
            total_tests += len(cases)
            print(f"{difficulty}: {avg_score:.0%} ({len(cases)} tests)")
    
    overall = total_score / total_tests if total_tests else 0
    print(f"\nOVERALL: {overall:.0%}")
    
    # Save results
    output_path = Path(__file__).parent.parent / "models" / "veran_eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    run_eval()
