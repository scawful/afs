"""Multi-provider teacher model wrappers for distillation.

Provides unified interface to OpenAI, Google, and Anthropic models for
generating high-quality training data.

## Latest Models (January 2026)

### OpenAI
- gpt-5.2: Latest GPT-5 model (uses max_completion_tokens, not max_tokens)
- gpt-5-mini: Smaller GPT-5 variant
- gpt-4.1: Previous generation

### Google Gemini (use google-genai package, not deprecated google-generativeai)
- gemini-3-flash-preview: Latest Gemini 3 Flash (fast, frontier-class)
- gemini-3-pro-preview: Latest Gemini 3 Pro (most capable)
- gemini-2.5-flash: Legacy Gemini 2.5 Flash (avoid for new work)
- gemini-2.5-pro: Legacy Gemini 2.5 Pro (avoid for new work)

### Anthropic
- claude-opus-4-5-20251101: Claude Opus 4.5 (most capable)
- claude-sonnet-4-20250514: Claude Sonnet 4

## Environment Variables
- OPENAI_API_KEY: OpenAI API key
- GEMINI_API_KEY: Google Gemini API key
- CLAUDE_API_KEY: Anthropic Claude API key (ANTHROPIC_API_KEY also works)

## API Quirks
- OpenAI GPT-5 models use `max_completion_tokens` parameter (not `max_tokens`)
- Google genai: Use `google-genai` package, not deprecated `google-generativeai`
- Anthropic: SDK automatically falls back to ANTHROPIC_API_KEY if CLAUDE_API_KEY not set
"""

import asyncio
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, AsyncIterator

import logging

logger = logging.getLogger(__name__)


class Provider(Enum):
    """Supported API providers."""
    OPENAI = "openai"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"


@dataclass
class ProviderConfig:
    """Configuration for a provider."""
    provider: Provider
    model: str
    api_key: Optional[str] = None  # Uses env var if None
    requests_per_minute: int = 10
    max_tokens: int = 4096
    temperature: float = 0.7

    # Provider-specific defaults
    @classmethod
    def openai_default(cls) -> "ProviderConfig":
        return cls(
            provider=Provider.OPENAI,
            model="gpt-5.2",
            requests_per_minute=60,
            max_tokens=4096,
        )

    @classmethod
    def google_default(cls) -> "ProviderConfig":
        return cls(
            provider=Provider.GOOGLE,
            model="gemini-3-flash-preview",
            requests_per_minute=15,  # Conservative for free tier
            max_tokens=8192,
        )

    @classmethod
    def anthropic_default(cls) -> "ProviderConfig":
        return cls(
            provider=Provider.ANTHROPIC,
            model="claude-opus-4-5-20251101",
            requests_per_minute=50,
            max_tokens=4096,
        )


@dataclass
class TeacherResponse:
    """Response from a teacher model."""
    content: str
    provider: Provider
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and len(self.content) > 0


class TeacherModel(ABC):
    """Abstract base class for teacher models."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._last_request_time = 0.0
        self._min_interval = 60.0 / config.requests_per_minute

    async def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> TeacherResponse:
        """Generate a response from the teacher model."""
        pass

    @property
    def provider(self) -> Provider:
        return self.config.provider

    @property
    def model(self) -> str:
        return self.config.model


class OpenAITeacher(TeacherModel):
    """OpenAI teacher model wrapper.

    Available models (as of Jan 2026):
    - gpt-5.2: Latest GPT-5 model (uses max_completion_tokens, not max_tokens)
    - gpt-5-mini: Smaller GPT-5 variant
    - gpt-4.1: Previous generation
    """

    def __init__(self, config: Optional[ProviderConfig] = None):
        if config is None:
            config = ProviderConfig.openai_default()
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
                self._client = AsyncOpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai package required: pip install openai")
        return self._client

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> TeacherResponse:
        await self._rate_limit()

        start_time = time.time()
        try:
            client = self._get_client()

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = await client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_completion_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            content = response.choices[0].message.content or ""
            latency = (time.time() - start_time) * 1000

            return TeacherResponse(
                content=content,
                provider=self.provider,
                model=self.model,
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                latency_ms=latency,
            )
        except Exception as e:
            return TeacherResponse(
                content="",
                provider=self.provider,
                model=self.model,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )


class GoogleTeacher(TeacherModel):
    """Google Gemini teacher model wrapper.

    Uses the new google-genai package (not deprecated google-generativeai).

    Available models (as of Jan 2026):
    - gemini-3-flash-preview: Latest Gemini 3 Flash (fast, capable)
    - gemini-3-pro-preview: Latest Gemini 3 Pro (most capable)
    - gemini-2.5-flash: Legacy Gemini 2.5 Flash (avoid for new work)
    - gemini-2.5-pro: Legacy Gemini 2.5 Pro (avoid for new work)
    """

    def __init__(self, config: Optional[ProviderConfig] = None):
        if config is None:
            config = ProviderConfig.google_default()
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
                api_key = self.config.api_key or os.getenv("GEMINI_API_KEY")
                self._client = genai.Client(api_key=api_key)
            except ImportError:
                raise ImportError("google-genai package required: pip install google-genai")
        return self._client

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> TeacherResponse:
        await self._rate_limit()

        start_time = time.time()
        try:
            client = self._get_client()

            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            # Use async generate_content
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=self.config.model,
                    contents=full_prompt,
                    config={
                        "max_output_tokens": self.config.max_tokens,
                        "temperature": self.config.temperature,
                    }
                )
            )

            content = response.text if response.text else ""
            latency = (time.time() - start_time) * 1000

            return TeacherResponse(
                content=content,
                provider=self.provider,
                model=self.model,
                latency_ms=latency,
            )
        except Exception as e:
            return TeacherResponse(
                content="",
                provider=self.provider,
                model=self.model,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )


class AnthropicTeacher(TeacherModel):
    """Anthropic Claude teacher model wrapper.

    Available models (as of Jan 2026):
    - claude-opus-4-5-20251101: Claude Opus 4.5 (most capable)
    - claude-sonnet-4-20250514: Claude Sonnet 4

    Note: Uses CLAUDE_API_KEY env var (Anthropic SDK also accepts ANTHROPIC_API_KEY).
    """

    def __init__(self, config: Optional[ProviderConfig] = None):
        if config is None:
            config = ProviderConfig.anthropic_default()
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
                api_key = self.config.api_key or os.getenv("CLAUDE_API_KEY")
                self._client = AsyncAnthropic(api_key=api_key)
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")
        return self._client

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> TeacherResponse:
        await self._rate_limit()

        start_time = time.time()
        try:
            client = self._get_client()

            response = await client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
            )

            content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text

            latency = (time.time() - start_time) * 1000

            return TeacherResponse(
                content=content,
                provider=self.provider,
                model=self.model,
                prompt_tokens=response.usage.input_tokens if response.usage else 0,
                completion_tokens=response.usage.output_tokens if response.usage else 0,
                latency_ms=latency,
            )
        except Exception as e:
            return TeacherResponse(
                content="",
                provider=self.provider,
                model=self.model,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )


@dataclass
class TeacherEnsemble:
    """Ensemble of teacher models with load balancing and failover."""

    teachers: list[TeacherModel] = field(default_factory=list)
    _current_index: int = 0
    _failure_counts: dict = field(default_factory=dict)

    @classmethod
    def default_ensemble(cls) -> "TeacherEnsemble":
        """Create default ensemble with all three providers."""
        return cls(teachers=[
            OpenAITeacher(),
            GoogleTeacher(),
            AnthropicTeacher(),
        ])

    @classmethod
    def from_configs(cls, configs: list[ProviderConfig]) -> "TeacherEnsemble":
        """Create ensemble from provider configs."""
        teachers = []
        for config in configs:
            if config.provider == Provider.OPENAI:
                teachers.append(OpenAITeacher(config))
            elif config.provider == Provider.GOOGLE:
                teachers.append(GoogleTeacher(config))
            elif config.provider == Provider.ANTHROPIC:
                teachers.append(AnthropicTeacher(config))
        return cls(teachers=teachers)

    def _get_next_teacher(self) -> TeacherModel:
        """Round-robin teacher selection with failure awareness."""
        if not self.teachers:
            raise ValueError("No teachers in ensemble")

        # Skip teachers with too many failures
        max_failures = 5
        attempts = 0
        while attempts < len(self.teachers):
            teacher = self.teachers[self._current_index]
            self._current_index = (self._current_index + 1) % len(self.teachers)

            failures = self._failure_counts.get(teacher.provider, 0)
            if failures < max_failures:
                return teacher
            attempts += 1

        # Reset failure counts if all providers have failed
        self._failure_counts.clear()
        return self.teachers[0]

    def _record_failure(self, provider: Provider) -> None:
        """Record a failure for a provider."""
        self._failure_counts[provider] = self._failure_counts.get(provider, 0) + 1

    def _record_success(self, provider: Provider) -> None:
        """Reset failure count on success."""
        self._failure_counts[provider] = 0

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
    ) -> TeacherResponse:
        """Generate with automatic failover to other providers."""
        last_error = None

        for attempt in range(max_retries):
            teacher = self._get_next_teacher()

            response = await teacher.generate(prompt, system_prompt)

            if response.success:
                self._record_success(teacher.provider)
                return response

            # Record failure and try next provider
            self._record_failure(teacher.provider)
            last_error = response.error
            logger.warning(
                f"Teacher {teacher.provider.value} failed: {response.error}. "
                f"Attempt {attempt + 1}/{max_retries}"
            )

        return TeacherResponse(
            content="",
            provider=Provider.OPENAI,  # placeholder
            model="ensemble",
            error=f"All providers failed. Last error: {last_error}",
        )

    async def generate_parallel(
        self,
        prompts: list[str],
        system_prompt: Optional[str] = None,
        batch_size: int = 5,
    ) -> AsyncIterator[TeacherResponse]:
        """Generate responses in parallel with batch processing."""
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]

            tasks = [
                self.generate(prompt, system_prompt)
                for prompt in batch
            ]

            results = await asyncio.gather(*tasks)
            for result in results:
                yield result
