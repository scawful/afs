"""Intent classification for MoE routing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class QueryIntent(str, Enum):
    """Query intent categories mapping to expert models."""

    OPTIMIZATION = "optimization"  # din - code optimization, size/cycle reduction
    GENERATION = "generation"      # nayru - code generation, writing new code
    DEBUGGING = "debugging"        # farore - debugging, error analysis (future)
    GENERAL = "general"            # base model fallback


@dataclass
class ClassificationResult:
    """Result of intent classification."""

    intent: QueryIntent
    confidence: float
    matched_patterns: list[str]


class IntentClassifier:
    """Keyword-based intent classifier for 65816 assembly queries."""

    # Pattern weights for each intent
    OPTIMIZATION_PATTERNS = {
        # Direct optimization keywords
        r"\boptimize\b": 1.0,
        r"\boptimization\b": 1.0,
        r"\bfaster\b": 0.8,
        r"\bsmaller\b": 0.8,
        r"\bsave\s+(bytes?|cycles?)\b": 1.0,
        r"\breduce\s+(size|cycles?|bytes?)\b": 1.0,
        r"\bimprove\b": 0.6,
        r"\befficient\b": 0.7,
        # Specific optimization techniques
        r"\bSTZ\b": 0.9,
        r"\bMVN\b": 0.9,
        r"\bMVP\b": 0.9,
        r"\bjump\s+table\b": 0.8,
        r"\bloop\s+unroll": 0.8,
        r"\bbranchless\b": 0.9,
        # Hardware optimization
        r"\$420[234567]\b": 0.8,  # Hardware multiplier/divider registers
        r"\bhardware\s+(mult|div)": 0.9,
        # Comparison words
        r"\bbetter\s+way\b": 0.7,
        r"\bcan\s+this\s+be\b": 0.5,
        r"\bhow\s+(can|do)\s+I\s+(make|improve)\b": 0.6,
    }

    GENERATION_PATTERNS = {
        # Direct generation keywords
        r"\bwrite\b": 0.8,
        r"\bgenerate\b": 1.0,
        r"\bcreate\b": 0.8,
        r"\bimplement\b": 0.9,
        r"\bbuild\b": 0.7,
        r"\bcode\s+for\b": 0.8,
        r"\bcode\s+that\b": 0.8,
        # Specification patterns
        r"\bfunction\s+(that|to|which)\b": 0.9,
        r"\broutine\s+(that|to|which)\b": 0.9,
        r"\bsubroutine\b": 0.7,
        # Question patterns for new code
        r"\bhow\s+(do|would)\s+I\s+(write|implement|create)\b": 0.9,
        r"\bwhat.*code.*for\b": 0.7,
        # Feature requests
        r"\badd\s+(a|the)?\s*(feature|function|routine)\b": 0.8,
        r"\bneed\s+(a|to)\s*(write|create|implement)\b": 0.8,
    }

    DEBUGGING_PATTERNS = {
        # Error/bug keywords - high weight
        r"\bbug\b": 1.2,
        r"\berror\b": 1.0,
        r"\bcrash(es|ed|ing)?\b": 1.2,
        r"\bfreeze\b": 1.0,
        r"\bhang\b": 0.9,
        r"\bwrong\b": 0.9,
        r"\bbroken\b": 1.0,
        r"\bfail(s|ed|ing)?\b": 0.9,
        r"\bdoesn't\s+(work|store|transfer|return)\b": 1.2,
        r"\bnot\s+(storing|working|transferring|returning)\b": 1.0,
        # Debug actions
        r"\bdebug\b": 1.2,
        r"\bfix\b": 0.8,
        r"\btrace\b": 0.8,
        r"\bstep\s+through\b": 0.9,
        r"\bdiagnose\b": 1.0,
        # Investigation patterns
        r"\bwhy\s+(is|does|doesn't|won't|can't)\b": 1.0,
        r"\bwhat's\s+wrong\b": 1.2,
        r"\bnot\s+working\b": 1.0,
        r"\bexpected.*but\s+got\b": 1.2,
        # Negative results - strong debugging signals
        r"\bgives?\s+wrong\b": 1.2,
        r"\breturns?\s+wrong\b": 1.0,
        r"\bstops?\s+early\b": 1.0,
        r"\bdoesn't\s+transfer\b": 1.2,
    }

    def __init__(self, optimization_threshold: float = 0.3,
                 generation_threshold: float = 0.3,
                 debugging_threshold: float = 0.4):
        """Initialize classifier with confidence thresholds."""
        self.optimization_threshold = optimization_threshold
        self.generation_threshold = generation_threshold
        self.debugging_threshold = debugging_threshold

        # Compile patterns
        self._opt_patterns = {
            re.compile(p, re.IGNORECASE): w
            for p, w in self.OPTIMIZATION_PATTERNS.items()
        }
        self._gen_patterns = {
            re.compile(p, re.IGNORECASE): w
            for p, w in self.GENERATION_PATTERNS.items()
        }
        self._dbg_patterns = {
            re.compile(p, re.IGNORECASE): w
            for p, w in self.DEBUGGING_PATTERNS.items()
        }

    def _score_patterns(
        self,
        text: str,
        patterns: dict[re.Pattern, float]
    ) -> tuple[float, list[str]]:
        """Score text against pattern set, return (score, matched_patterns)."""
        total_weight = 0.0
        matches = []

        for pattern, weight in patterns.items():
            if pattern.search(text):
                total_weight += weight
                matches.append(pattern.pattern)

        # Use raw accumulated weight - a single strong match should be enough
        # Cap at 1.0 for display purposes
        return total_weight, matches

    def classify(self, query: str) -> ClassificationResult:
        """Classify query intent based on keyword patterns."""
        opt_score, opt_matches = self._score_patterns(query, self._opt_patterns)
        gen_score, gen_matches = self._score_patterns(query, self._gen_patterns)
        dbg_score, dbg_matches = self._score_patterns(query, self._dbg_patterns)

        # Check for code block presence (suggests optimization over generation)
        # But don't boost optimization if debugging patterns are strong
        has_code = bool(re.search(r'(LDA|STA|STZ|JSR|JMP|BNE|BEQ|CMP|AND|ORA|ASL|LSR)\s', query))
        if has_code and dbg_score < 0.8:
            opt_score += 0.5

        # Determine intent based on highest score above threshold
        scores = [
            (QueryIntent.OPTIMIZATION, opt_score, self.optimization_threshold, opt_matches),
            (QueryIntent.GENERATION, gen_score, self.generation_threshold, gen_matches),
            (QueryIntent.DEBUGGING, dbg_score, self.debugging_threshold, dbg_matches),
        ]

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        best_intent, best_score, threshold, matches = scores[0]

        # If any score is above threshold, use it
        if best_score >= threshold:
            # Normalize confidence to 0-1 range (cap at ~3.0 raw score)
            confidence = min(best_score / 3.0, 1.0)
            return ClassificationResult(
                intent=best_intent,
                confidence=confidence,
                matched_patterns=matches,
            )

        # Fallback to general
        max_score = max(opt_score, gen_score, dbg_score)
        return ClassificationResult(
            intent=QueryIntent.GENERAL,
            confidence=1.0 - min(max_score / 3.0, 0.9),
            matched_patterns=[],
        )
