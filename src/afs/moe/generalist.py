"""Generalist model with expert invocation tokens.

Provides a unified model interface that can route to specialized experts
using special tokens like <INVOKE_EXPERT:din>.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ExpertToken(Enum):
    """Special tokens for expert invocation."""
    INVOKE_DIN = "<INVOKE_EXPERT:din>"
    INVOKE_NAYRU = "<INVOKE_EXPERT:nayru>"
    INVOKE_FARORE = "<INVOKE_EXPERT:farore>"
    INVOKE_VERAN = "<INVOKE_EXPERT:veran>"
    EXPERT_RESPONSE = "<EXPERT_RESPONSE>"
    END_EXPERT = "</EXPERT_RESPONSE>"


# Token patterns for parsing
INVOKE_PATTERN = re.compile(r"<INVOKE_EXPERT:(\w+)>")
RESPONSE_PATTERN = re.compile(r"<EXPERT_RESPONSE>(.*?)</EXPERT_RESPONSE>", re.DOTALL)


@dataclass
class ExpertInvocation:
    """Represents a parsed expert invocation."""
    expert_name: str
    context: str  # Text before invocation
    response: str | None = None


@dataclass
class GeneralistConfig:
    """Configuration for generalist model."""
    base_model: str = "qwen2.5:7b"
    temperature: float = 0.7
    max_tokens: int = 4096
    expert_timeout: float = 30.0


class GeneralistModel:
    """Generalist model that can invoke experts."""

    def __init__(
        self,
        config: GeneralistConfig | None = None,
        expert_handlers: dict[str, Callable] | None = None,
    ):
        self.config = config or GeneralistConfig()
        self.expert_handlers = expert_handlers or {}
        self._client = None

    def register_expert(self, name: str, handler: Callable) -> None:
        """Register an expert handler.

        Handler signature: async def handler(query: str, context: str) -> str
        """
        self.expert_handlers[name.lower()] = handler

    def parse_invocations(self, text: str) -> list[ExpertInvocation]:
        """Parse expert invocations from generated text."""
        invocations = []

        # Find all invocation tokens
        for match in INVOKE_PATTERN.finditer(text):
            expert_name = match.group(1).lower()
            context = text[:match.start()]
            invocations.append(ExpertInvocation(
                expert_name=expert_name,
                context=context,
            ))

        return invocations

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        execute_experts: bool = True,
    ) -> str:
        """Generate response, optionally executing expert invocations."""
        # Get base generation
        response = await self._generate_base(prompt, system_prompt)

        if not execute_experts:
            return response

        # Parse and execute expert invocations
        invocations = self.parse_invocations(response)

        if not invocations:
            return response

        # Execute each expert
        for inv in invocations:
            if inv.expert_name not in self.expert_handlers:
                logger.warning(f"No handler for expert: {inv.expert_name}")
                continue

            handler = self.expert_handlers[inv.expert_name]
            try:
                expert_response = await handler(prompt, inv.context)
                inv.response = expert_response
            except Exception as e:
                logger.error(f"Expert {inv.expert_name} failed: {e}")
                inv.response = f"[Expert error: {e}]"

        # Reconstruct response with expert outputs
        return self._reconstruct_response(response, invocations)

    async def _generate_base(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate from base model."""
        # Try ollama first
        try:
            import httpx

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": self.config.base_model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": self.config.temperature,
                            "num_predict": self.config.max_tokens,
                        },
                    },
                    timeout=60.0,
                )
                response.raise_for_status()
                return response.json()["message"]["content"]

        except Exception as e:
            logger.warning(f"Ollama failed: {e}, returning empty")
            return ""

    def _reconstruct_response(
        self,
        original: str,
        invocations: list[ExpertInvocation],
    ) -> str:
        """Reconstruct response with expert outputs inserted."""
        result = original

        for inv in invocations:
            if inv.response:
                # Find the invocation token and insert response after it
                token = f"<INVOKE_EXPERT:{inv.expert_name}>"
                replacement = f"{token}\n{ExpertToken.EXPERT_RESPONSE.value}\n{inv.response}\n{ExpertToken.END_EXPERT.value}"
                result = result.replace(token, replacement, 1)

        return result

    def get_system_prompt(self) -> str:
        """Get system prompt that teaches expert invocation."""
        expert_list = ", ".join(self.expert_handlers.keys()) or "din, nayru, farore, veran"

        return f"""You are a generalist AI assistant with access to specialized experts.

Available experts: {expert_list}

When you need specialized help, use the invocation token:
<INVOKE_EXPERT:expert_name>

The expert will respond between:
<EXPERT_RESPONSE>
[expert's response]
</EXPERT_RESPONSE>

Expert capabilities:
- din: Code optimization, reducing cycles and bytes
- nayru: Code generation, writing new assembly
- farore: Debugging, finding and fixing bugs
- veran: Code explanation, teaching concepts

Only invoke experts when their specialization is needed. For general questions, respond directly."""


class InvocationDataGenerator:
    """Generates training data for expert invocation."""

    def __init__(self, generalist: GeneralistModel):
        self.generalist = generalist

    async def generate_invocation_trace(
        self,
        query: str,
        expected_expert: str,
        expert_response: str,
    ) -> dict:
        """Generate a training example showing proper expert invocation."""
        return {
            "instruction": query,
            "input": "",
            "output": f"""This requires {expected_expert} expertise.

<INVOKE_EXPERT:{expected_expert}>
<EXPERT_RESPONSE>
{expert_response}
</EXPERT_RESPONSE>

Based on the expert's response, here is the complete answer...""",
            "metadata": {
                "expert": expected_expert,
                "type": "invocation_trace",
            }
        }

    async def generate_batch(
        self,
        examples: list[tuple[str, str, str]],  # (query, expert, response)
    ) -> list[dict]:
        """Generate batch of training examples."""
        return [
            await self.generate_invocation_trace(q, e, r)
            for q, e, r in examples
        ]
