"""Critic loop for iterative refinement.

Implements a generate-critique-refine loop where one expert
reviews another's output and provides feedback for improvement.
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class CriticVerdict(Enum):
    """Possible critic verdicts."""
    PASS = "pass"
    FAIL = "fail"
    NEEDS_IMPROVEMENT = "needs_improvement"


@dataclass
class CriticFeedback:
    """Feedback from critic on an output."""
    verdict: CriticVerdict
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    score: float = 0.0  # 0-1 quality score
    explanation: str = ""


@dataclass
class RefinementResult:
    """Result of a refinement iteration."""
    iteration: int
    original: str
    refined: str
    feedback: CriticFeedback
    improved: bool


@dataclass
class CriticConfig:
    """Configuration for critic loop."""
    max_iterations: int = 3
    min_score_threshold: float = 0.7
    auto_pass_threshold: float = 0.9
    critic_expert: str = "farore"  # Default critic


class CriticLoop:
    """Implements generate-critique-refine loop."""

    def __init__(
        self,
        generator: Callable[[str, str], Awaitable[str]],
        critic: Callable[[str, str], Awaitable[CriticFeedback]],
        config: CriticConfig | None = None,
    ):
        """
        Args:
            generator: Async function (prompt, feedback) -> output
            critic: Async function (prompt, output) -> CriticFeedback
            config: Loop configuration
        """
        self.generator = generator
        self.critic = critic
        self.config = config or CriticConfig()

    async def run(
        self,
        prompt: str,
        initial_output: str | None = None,
    ) -> tuple[str, list[RefinementResult]]:
        """Run the critic loop until pass or max iterations.

        Returns:
            (final_output, list of refinement results)
        """
        history = []

        # Generate initial output if not provided
        if initial_output is None:
            current = await self.generator(prompt, "")
        else:
            current = initial_output

        for i in range(self.config.max_iterations):
            # Get critic feedback
            feedback = await self.critic(prompt, current)

            # Check if we can stop
            if feedback.verdict == CriticVerdict.PASS:
                history.append(RefinementResult(
                    iteration=i,
                    original=current,
                    refined=current,
                    feedback=feedback,
                    improved=False,
                ))
                logger.info(f"Critic passed on iteration {i}")
                break

            if feedback.score >= self.config.auto_pass_threshold:
                history.append(RefinementResult(
                    iteration=i,
                    original=current,
                    refined=current,
                    feedback=feedback,
                    improved=False,
                ))
                logger.info(f"Auto-pass on iteration {i} (score: {feedback.score})")
                break

            # Generate refined output
            feedback_prompt = self._format_feedback(feedback)
            refined = await self.generator(prompt, feedback_prompt)

            history.append(RefinementResult(
                iteration=i,
                original=current,
                refined=refined,
                feedback=feedback,
                improved=refined != current,
            ))

            current = refined
            logger.info(f"Iteration {i}: score={feedback.score:.2f}, refined={refined != current}")

        return current, history

    def _format_feedback(self, feedback: CriticFeedback) -> str:
        """Format feedback for the generator."""
        parts = ["Previous output had issues:"]

        if feedback.issues:
            parts.append("\nIssues:")
            for issue in feedback.issues:
                parts.append(f"- {issue}")

        if feedback.suggestions:
            parts.append("\nSuggestions:")
            for suggestion in feedback.suggestions:
                parts.append(f"- {suggestion}")

        if feedback.explanation:
            parts.append(f"\nExplanation: {feedback.explanation}")

        return "\n".join(parts)


class FaroreCritic:
    """Farore-based critic for assembly code review."""

    def __init__(self, farore_handler: Callable):
        self.farore = farore_handler

    async def critique(self, prompt: str, output: str) -> CriticFeedback:
        """Critique assembly code output using Farore."""
        critique_prompt = f"""Review this 65816 assembly code for correctness and quality.

Original task: {prompt}

Generated code:
{output}

Analyze for:
1. Syntax errors
2. Logic bugs
3. Register/mode mismatches
4. Inefficiencies

Respond with:
- VERDICT: PASS, FAIL, or NEEDS_IMPROVEMENT
- SCORE: 0.0-1.0
- ISSUES: (list any issues found)
- SUGGESTIONS: (list improvements)
"""

        response = await self.farore(critique_prompt, "")

        return self._parse_critique(response)

    def _parse_critique(self, response: str) -> CriticFeedback:
        """Parse Farore's critique response."""
        import re

        # Extract verdict
        verdict = CriticVerdict.NEEDS_IMPROVEMENT
        if "VERDICT: PASS" in response.upper():
            verdict = CriticVerdict.PASS
        elif "VERDICT: FAIL" in response.upper():
            verdict = CriticVerdict.FAIL

        # Extract score
        score = 0.5
        score_match = re.search(r"SCORE:\s*([\d.]+)", response)
        if score_match:
            score = float(score_match.group(1))

        # Extract issues
        issues = []
        issues_section = re.search(r"ISSUES:(.*?)(?:SUGGESTIONS:|$)", response, re.DOTALL)
        if issues_section:
            for line in issues_section.group(1).split("\n"):
                line = line.strip().lstrip("-").strip()
                if line:
                    issues.append(line)

        # Extract suggestions
        suggestions = []
        sugg_section = re.search(r"SUGGESTIONS:(.*?)$", response, re.DOTALL)
        if sugg_section:
            for line in sugg_section.group(1).split("\n"):
                line = line.strip().lstrip("-").strip()
                if line:
                    suggestions.append(line)

        return CriticFeedback(
            verdict=verdict,
            issues=issues,
            suggestions=suggestions,
            score=score,
            explanation=response,
        )


async def create_din_farore_loop(
    din_handler: Callable,
    farore_handler: Callable,
    config: CriticConfig | None = None,
) -> CriticLoop:
    """Create a Din (optimizer) + Farore (critic) loop."""

    async def generator(prompt: str, feedback: str) -> str:
        full_prompt = prompt
        if feedback:
            full_prompt = f"{prompt}\n\n{feedback}"
        return await din_handler(full_prompt, "")

    critic = FaroreCritic(farore_handler)

    return CriticLoop(
        generator=generator,
        critic=critic.critique,
        config=config,
    )
