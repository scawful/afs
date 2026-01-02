"""LLM API clients for CoT generation."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .prompts import ASM_COT_SYSTEM_PROMPT


class LLMClient(ABC):
    """Abstract base class for LLM API clients."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Generated text response
        """
        pass

    def generate_batch(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> list[str]:
        """Generate responses for multiple prompts.

        Default implementation calls generate() sequentially.
        Subclasses may override for batch API support.
        """
        return [
            self.generate(prompt, system_prompt, temperature, max_tokens)
            for prompt in prompts
        ]


class GeminiClient(LLMClient):
    """Google Gemini API client using google.genai SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3-flash-preview",
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Set environment variable or pass api_key."
            )
        self.model = model
        self._client = None

    def _get_client(self) -> Any:
        """Lazy initialization of Gemini client."""
        if self._client is None:
            try:
                from google import genai

                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "google-genai not installed. "
                    "Install with: pip install google-genai"
                )
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Generate response using Gemini API."""
        client = self._get_client()

        system = system_prompt or ASM_COT_SYSTEM_PROMPT

        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "system_instruction": system,
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )

        return response.text


class ClaudeClient(LLMClient):
    """Anthropic Claude API client."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-5-sonnet-20241022",
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Set environment variable or pass api_key."
            )
        self.model = model
        self._client = None

    def _get_client(self) -> Any:
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic not installed. Install with: pip install anthropic"
                )
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Generate response using Claude API."""
        client = self._get_client()

        message = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt or ASM_COT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        return message.content[0].text


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Set environment variable or pass api_key."
            )
        self.model = model
        self._client = None

    def _get_client(self) -> Any:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai not installed. Install with: pip install openai"
                )
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Generate response using OpenAI API."""
        client = self._get_client()

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt or ASM_COT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content


def get_client(provider: str, **kwargs) -> LLMClient:
    """Factory function to get LLM client by provider name.

    Args:
        provider: Provider name (gemini, claude, openai)
        **kwargs: Additional arguments for client constructor

    Returns:
        LLMClient instance
    """
    clients = {
        "gemini": GeminiClient,
        "claude": ClaudeClient,
        "openai": OpenAIClient,
    }

    if provider not in clients:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported: {', '.join(clients.keys())}"
        )

    return clients[provider](**kwargs)
