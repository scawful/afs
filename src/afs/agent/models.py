"""Model abstraction layer for unified access to local and cloud models.

Supports:
    - Ollama (local models)
    - LMStudio (local GGUF models via OpenAI-compatible API)
    - Google Gemini (cloud)
    - Anthropic Claude (cloud, optional)
    - OpenAI-compatible APIs (cloud or local gateways)
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..claude_defaults import claude_prompt_cache_enabled, claude_system_content
from ..gemini_defaults import validate_gemini_thinking_level

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Supported model providers."""

    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    LITELLM = "litellm"


@dataclass
class ModelConfig:
    """Configuration for a model.

    Examples:
        # Local Ollama model
        ModelConfig(provider=ModelProvider.OLLAMA, model_id="llama3.2")

        # Gemini
        ModelConfig(provider=ModelProvider.GEMINI, model_id="gemini-3.8-flash")

        # From string shorthand
        ModelConfig.from_string("ollama:llama3.2")
        ModelConfig.from_string("gemini-3.8-flash")  # Defaults to gemini provider
    """

    provider: ModelProvider
    model_id: str
    temperature: float = 0.5
    top_p: float = 0.85
    max_tokens: int = 4096
    system_prompt: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_string(cls, model_str: str) -> ModelConfig:
        """Parse model string like 'ollama:llama3.2' or 'gemini-3.8-flash'."""
        if ":" in model_str:
            parts = model_str.split(":", 1)
            provider_str = parts[0].lower()
            model_id = parts[1]

            try:
                provider = ModelProvider(provider_str)
            except ValueError:
                # Assume it's an Ollama model with : in the name
                provider = ModelProvider.OLLAMA
                model_id = model_str
        else:
            # Default inference based on model name
            if model_str.startswith("gemini"):
                provider = ModelProvider.GEMINI
            elif model_str.startswith("openrouter/") or model_str.startswith("openrouter-"):
                provider = ModelProvider.OPENROUTER
            elif model_str.startswith("claude"):
                provider = ModelProvider.ANTHROPIC
            elif model_str.startswith("gpt"):
                provider = ModelProvider.OPENAI
            elif model_str.startswith("litellm/"):
                provider = ModelProvider.LITELLM
            elif model_str.startswith("gguf/") or model_str.endswith(".gguf"):
                # GGUF models are typically served by LMStudio
                provider = ModelProvider.LMSTUDIO
            else:
                # Assume local Ollama model
                provider = ModelProvider.OLLAMA

            model_id = model_str

        return cls(provider=provider, model_id=model_id)

    # Compatibility presets now owned by afs_scawful
    @classmethod
    def din(cls) -> ModelConfig:
        """Compatibility preset for afs_scawful."""
        return _load_scawful_preset("din")

    @classmethod
    def nayru(cls) -> ModelConfig:
        """Compatibility preset for afs_scawful."""
        return _load_scawful_preset("nayru")

    @classmethod
    def farore(cls) -> ModelConfig:
        """Compatibility preset for afs_scawful."""
        return _load_scawful_preset("farore")

    @classmethod
    def veran(cls) -> ModelConfig:
        """Compatibility preset for afs_scawful."""
        return _load_scawful_preset("veran")

    # LMStudio compatibility presets now owned by afs_scawful
    @classmethod
    def din_lmstudio(cls) -> ModelConfig:
        """Compatibility preset for afs_scawful."""
        return _load_scawful_preset("din_lmstudio")

    @classmethod
    def farore_lmstudio(cls) -> ModelConfig:
        """Compatibility preset for afs_scawful."""
        return _load_scawful_preset("farore_lmstudio")

    @classmethod
    def veran_lmstudio(cls) -> ModelConfig:
        """Compatibility preset for afs_scawful."""
        return _load_scawful_preset("veran_lmstudio")

    @classmethod
    def majora_lmstudio(cls) -> ModelConfig:
        """Compatibility preset for afs_scawful."""
        return _load_scawful_preset("majora_lmstudio")


def _load_scawful_preset(name: str) -> ModelConfig:
    """Load an extension-owned preset without baking it into core AFS."""
    try:
        from afs_scawful.agent_model_presets import build_preset
    except Exception as exc:  # pragma: no cover - compatibility path
        raise RuntimeError(
            "Domain-specific model presets moved to the afs_scawful extension repo."
        ) from exc

    preset = build_preset(name)
    if not isinstance(preset, ModelConfig):
        raise TypeError(f"afs_scawful preset {name!r} did not return ModelConfig")
    return preset


@dataclass
class ToolCall:
    """A tool call requested by the model."""

    name: str
    arguments: dict[str, Any]
    id: str = ""  # Some providers return call IDs
    thought_signature: str = ""  # Base64 Gemini tool-turn continuity token


@dataclass
class GenerateResult:
    """Result from model generation."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    raw_response: Any = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@dataclass(frozen=True)
class GeminiCacheSettings:
    """Configuration for Gemini explicit cached-content usage."""

    mode: str = "off"
    ttl: str = "3600s"
    min_prefix_chars: int = 4000

    @property
    def enabled(self) -> bool:
        return self.mode in {"try", "required"}

    @property
    def strict(self) -> bool:
        return self.mode == "required"


@dataclass(frozen=True)
class GeminiThinkingSettings:
    """Optional Gemini 3 thinking-level override.

    A missing level delegates the choice to the selected model. This keeps AFS
    compatible with non-Gemini-3 models and lets host harnesses own cost/latency
    policy.
    """

    level: str | None = None


def _model_extra_value(config: ModelConfig, key: str) -> Any:
    nested = config.extra.get("gemini_cache")
    if isinstance(nested, dict) and nested.get(key) is not None:
        return nested.get(key)
    extra_key = f"gemini_cache_{key}"
    if config.extra.get(extra_key) is not None:
        return config.extra.get(extra_key)
    return None


def _coerce_int_setting(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


def resolve_gemini_cache_settings(config: ModelConfig) -> GeminiCacheSettings:
    """Resolve Gemini cache settings from ModelConfig.extra and env vars."""
    raw_mode = (
        str(_model_extra_value(config, "mode") or os.getenv("AFS_GEMINI_CACHE_MODE", "off"))
        .strip()
        .lower()
    )
    if raw_mode not in {"off", "try", "required"}:
        raw_mode = "off"

    raw_ttl = _model_extra_value(config, "ttl") or os.getenv("AFS_GEMINI_CACHE_TTL") or "3600s"
    ttl = str(raw_ttl).strip() or "3600s"

    raw_min_prefix_chars = (
        _model_extra_value(config, "min_chars") or os.getenv("AFS_GEMINI_CACHE_MIN_CHARS") or 4000
    )
    min_prefix_chars = _coerce_int_setting(raw_min_prefix_chars, 4000)

    return GeminiCacheSettings(
        mode=raw_mode,
        ttl=ttl,
        min_prefix_chars=min_prefix_chars,
    )


def resolve_gemini_thinking_settings(config: ModelConfig) -> GeminiThinkingSettings:
    """Resolve an optional thinking level from model config or the environment."""
    nested = config.extra.get("gemini_thinking")
    nested_level = nested.get("level") if isinstance(nested, dict) else None
    raw_level = (
        nested_level
        or config.extra.get("gemini_thinking_level")
        or os.getenv("AFS_GEMINI_THINKING_LEVEL")
        or ""
    )
    level = str(raw_level).strip().lower()
    if level in {"", "auto", "default"}:
        return GeminiThinkingSettings()
    try:
        level = validate_gemini_thinking_level(config.model_id, level)
    except ValueError as exc:
        raise ValueError(f"invalid Gemini thinking level: {exc}") from exc
    return GeminiThinkingSettings(level=level)


class ModelBackend(ABC):
    """Abstract base class for model backends."""

    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> GenerateResult:
        """Generate a response from the model.

        Args:
            messages: Conversation history in OpenAI format:
                [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            tools: Optional tool definitions in OpenAI format

        Returns:
            GenerateResult with content and/or tool calls
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        pass


class OllamaBackend(ModelBackend):
    """Ollama local model backend."""

    def __init__(
        self,
        config: ModelConfig,
        host: str = "http://localhost:11434",
    ):
        super().__init__(config)
        self.host = host
        # Optional provider SDKs are imported lazily, so the concrete client
        # type is deliberately not part of AFS core's type dependency graph.
        self._client: Any = None

    async def _ensure_client(self):
        """Lazily initialize the HTTP client."""
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=60.0)

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> GenerateResult:
        """Generate using Ollama API."""
        await self._ensure_client()

        # Convert messages to Ollama format
        ollama_messages = []
        for msg in messages:
            ollama_msg = {"role": msg["role"], "content": msg.get("content", "")}
            ollama_messages.append(ollama_msg)

        # Add system prompt if configured
        if self.config.system_prompt and (
            not ollama_messages or ollama_messages[0]["role"] != "system"
        ):
            ollama_messages.insert(0, {"role": "system", "content": self.config.system_prompt})

        # Build request
        payload = {
            "model": self.config.model_id,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.max_tokens,
            },
        }

        # Add tools if provided (Ollama supports tool calling for some models)
        if tools:
            payload["tools"] = self._convert_tools_to_ollama(tools)

        try:
            response = await self._client.post(
                f"{self.host}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            # Parse response
            message = data.get("message", {})
            content = message.get("content", "")

            # Check for tool calls
            tool_calls = []
            if "tool_calls" in message:
                for tc in message["tool_calls"]:
                    tool_calls.append(
                        ToolCall(
                            name=tc["function"]["name"],
                            arguments=tc["function"].get("arguments", {}),
                        )
                    )

            return GenerateResult(
                content=content,
                tool_calls=tool_calls,
                finish_reason="tool_calls" if tool_calls else "stop",
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                },
                raw_response=data,
            )

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    def _convert_tools_to_ollama(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI tool format to Ollama format."""
        ollama_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                ollama_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": func["name"],
                            "description": func.get("description", ""),
                            "parameters": func.get("parameters", {}),
                        },
                    }
                )
        return ollama_tools

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class LMStudioBackend(ModelBackend):
    """LMStudio local model backend using OpenAI-compatible API.

    LMStudio serves models at localhost:1234 with OpenAI-compatible endpoints.
    Supports both chat completions and text completions for models with
    template issues.
    """

    def __init__(
        self,
        config: ModelConfig,
        host: str = "http://localhost:1234",
    ):
        super().__init__(config)
        self.host = host
        # Optional provider SDKs are imported lazily, so the concrete client
        # type is deliberately not part of AFS core's type dependency graph.
        self._client: Any = None
        self._use_completions = False  # Fallback for template issues

    async def _ensure_client(self):
        """Lazily initialize the HTTP client."""
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=120.0)

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> GenerateResult:
        """Generate using LMStudio's OpenAI-compatible API."""
        await self._ensure_client()

        # Add system prompt if configured
        if self.config.system_prompt and (not messages or messages[0]["role"] != "system"):
            messages = [{"role": "system", "content": self.config.system_prompt}] + messages

        # Try chat completions first
        if not self._use_completions:
            try:
                return await self._generate_chat(messages, tools)
            except Exception as e:
                error_msg = str(e)
                if "jinja" in error_msg.lower() or "template" in error_msg.lower():
                    logger.warning(
                        f"Chat endpoint failed with template error, falling back to completions: {e}"
                    )
                    self._use_completions = True
                else:
                    raise

        # Fallback to completions endpoint
        return await self._generate_completions(messages)

    async def _generate_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> GenerateResult:
        """Generate using chat completions endpoint."""
        payload = {
            "model": self.config.model_id,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        if tools:
            payload["tools"] = tools

        response = await self._client.post(
            f"{self.host}/v1/chat/completions",
            json=payload,
        )

        data = response.json()

        # Check for error in response
        if "error" in data:
            raise RuntimeError(data["error"])

        response.raise_for_status()

        # Parse response
        choice = data["choices"][0]
        message = choice["message"]
        content = message.get("content", "")

        # Check for tool calls
        tool_calls = []
        if "tool_calls" in message:
            for tc in message["tool_calls"]:
                func = tc.get("function", {})
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tool_calls.append(
                    ToolCall(
                        name=func.get("name", ""),
                        arguments=args,
                        id=tc.get("id", ""),
                    )
                )

        return GenerateResult(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
            usage=data.get("usage", {}),
            raw_response=data,
        )

    async def _generate_completions(
        self,
        messages: list[dict[str, Any]],
    ) -> GenerateResult:
        """Generate using completions endpoint with ChatML format.

        Used as fallback when chat endpoint has template issues.
        """
        # Build ChatML-formatted prompt
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        prompt = "\n".join(prompt_parts)

        payload = {
            "model": self.config.model_id,
            "prompt": prompt,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stop": ["<|im_end|}"],
        }

        response = await self._client.post(
            f"{self.host}/v1/completions",
            json=payload,
        )

        data = response.json()

        if "error" in data:
            raise RuntimeError(data["error"])

        response.raise_for_status()

        # Parse response
        choice = data["choices"][0]
        content = choice.get("text", "").strip()

        return GenerateResult(
            content=content,
            tool_calls=[],
            finish_reason=choice.get("finish_reason", "stop"),
            usage=data.get("usage", {}),
            raw_response=data,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class OpenAIBackend(ModelBackend):
    """OpenAI-compatible backend (OpenAI, OpenRouter, LiteLLM, gateway)."""

    def __init__(
        self,
        config: ModelConfig,
        base_url: str,
        api_key: str | None = None,
        require_key: bool = True,
    ):
        super().__init__(config)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.require_key = require_key
        # Optional provider SDKs are imported lazily, so the concrete client
        # type is deliberately not part of AFS core's type dependency graph.
        self._client: Any = None

    async def _ensure_client(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=120.0)

    def _chat_url(self) -> str:
        if self.base_url.endswith("/v1") or self.base_url.endswith("/api/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> GenerateResult:
        await self._ensure_client()

        if self.require_key and not self.api_key:
            raise RuntimeError("Missing API key for OpenAI-compatible backend.")

        if self.config.system_prompt and (not messages or messages[0]["role"] != "system"):
            messages = [{"role": "system", "content": self.config.system_prompt}] + messages

        payload = {
            "model": self.config.model_id,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            payload["tools"] = tools

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = await self._client.post(
            self._chat_url(),
            json=payload,
            headers=headers,
        )

        data = response.json()
        if "error" in data:
            raise RuntimeError(data["error"])

        response.raise_for_status()

        choice = data["choices"][0]
        message = choice["message"]
        content = message.get("content", "")

        tool_calls = []
        if "tool_calls" in message:
            for tc in message["tool_calls"]:
                func = tc.get("function", {})
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tool_calls.append(
                    ToolCall(
                        name=func.get("name", ""),
                        arguments=args,
                        id=tc.get("id", ""),
                    )
                )

        return GenerateResult(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
            usage=data.get("usage", {}),
            raw_response=data,
        )

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


class AnthropicBackend(ModelBackend):
    """Native Anthropic Messages API backend with stable-prefix caching."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        super().__init__(config)
        self.api_key = api_key or ""
        self.base_url = base_url.rstrip("/") if base_url else None
        # Optional provider SDKs are imported lazily, so the concrete client
        # type is deliberately not part of AFS core's type dependency graph.
        self._client: Any = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "Anthropic backend requires the optional dependency: pip install 'afs[claude]'"
            ) from exc

        if not self.api_key:
            raise RuntimeError(
                "Missing Anthropic API key; set ANTHROPIC_API_KEY or AFS_ANTHROPIC_API_KEY."
            )
        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": 120.0,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)

    def _system_content(self, messages: list[dict[str, Any]]) -> str | list[dict[str, Any]] | None:
        # The harness already carries the effective system prompt as a system
        # message. Only fall back to ModelConfig when a caller omits one, or the
        # prompt would be duplicated and consume context twice.
        parts = [
            str(message.get("content", "")).strip()
            for message in messages
            if message.get("role") == "system" and str(message.get("content", "")).strip()
        ]
        if not parts and self.config.system_prompt.strip():
            parts.append(self.config.system_prompt.strip())
        if not parts:
            return None
        if not claude_prompt_cache_enabled(self.config.extra):
            return "\n\n".join(parts)

        # Cache only the stable leading system block. Later system blocks can
        # contain changing truncation/recovery notices and must not destabilize
        # the reusable prefix.
        cached_prefix = claude_system_content(parts[0], cache=True)
        assert isinstance(cached_prefix, list)
        return [*cached_prefix, *({"type": "text", "text": part} for part in parts[1:])]

    def _messages_to_anthropic(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", ""))
            if role == "system":
                continue
            content = message.get("content", "")
            if role == "assistant":
                blocks: list[dict[str, Any]] = []
                if content:
                    blocks.append({"type": "text", "text": str(content)})
                for index, call in enumerate(message.get("tool_calls", [])):
                    if not isinstance(call, dict):
                        continue
                    name = str(call.get("name", "")).strip()
                    if not name:
                        continue
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": str(call.get("id") or f"tool_{index}"),
                            "name": name,
                            "input": call.get("arguments") or {},
                        }
                    )
                if blocks:
                    converted.append({"role": "assistant", "content": blocks})
                continue
            if role == "tool":
                blocks = []
                results = message.get("results")
                if not isinstance(results, list):
                    results = [message]
                for index, result in enumerate(results):
                    if not isinstance(result, dict):
                        continue
                    blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": str(
                                result.get("tool_call_id") or result.get("id") or f"tool_{index}"
                            ),
                            "content": str(result.get("content", "")),
                        }
                    )
                if blocks:
                    converted.append({"role": "user", "content": blocks})
                continue
            if role == "user":
                converted.append({"role": "user", "content": content})
        return converted

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            function = tool.get("function", {})
            name = str(function.get("name", "")).strip()
            if not name:
                continue
            converted.append(
                {
                    "name": name,
                    "description": str(function.get("description", "")),
                    "input_schema": function.get("parameters")
                    or {"type": "object", "properties": {}},
                }
            )
        return converted

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> GenerateResult:
        self._ensure_client()
        kwargs: dict[str, Any] = {
            "model": self.config.model_id,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": self._messages_to_anthropic(messages),
        }
        system = self._system_content(messages)
        if system is not None:
            kwargs["system"] = system
        if tools:
            converted_tools = self._convert_tools(tools)
            if converted_tools:
                kwargs["tools"] = converted_tools

        response = await self._client.messages.create(**kwargs)
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            block_type = getattr(block, "type", "")
            if block_type == "text" or (not block_type and hasattr(block, "text")):
                text = getattr(block, "text", "")
                if text:
                    content_parts.append(str(text))
            elif block_type == "tool_use":
                arguments = getattr(block, "input", {})
                tool_calls.append(
                    ToolCall(
                        name=str(getattr(block, "name", "")),
                        arguments=dict(arguments) if isinstance(arguments, dict) else {},
                        id=str(getattr(block, "id", "")),
                    )
                )

        usage = getattr(response, "usage", None)
        return GenerateResult(
            content="".join(content_parts),
            tool_calls=tool_calls,
            finish_reason=str(getattr(response, "stop_reason", "stop") or "stop"),
            usage={
                "prompt_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "output_tokens", 0) or 0),
                "cache_creation_input_tokens": int(
                    getattr(usage, "cache_creation_input_tokens", 0) or 0
                ),
                "cache_read_input_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            },
            raw_response=response,
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


class GeminiBackend(ModelBackend):
    """Google Gemini model backend."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # Optional provider SDKs are imported lazily, so the concrete client
        # type is deliberately not part of AFS core's type dependency graph.
        self._client: Any = None
        self._cache_settings = resolve_gemini_cache_settings(config)
        self._thinking_settings = resolve_gemini_thinking_settings(config)
        self._cached_content_names: dict[str, str] = {}

    def _ensure_client(self):
        """Lazily initialize the Gemini client."""
        if self._client is None:
            from google import genai

            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if api_key:
                self._client = genai.Client(api_key=api_key)
            else:
                self._client = genai.Client()

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> GenerateResult:
        """Generate using Gemini API."""
        from google.genai import types

        self._ensure_client()

        # Convert messages to Gemini format
        contents = self._messages_to_gemini_contents(messages, types)

        # Convert tools to Gemini format
        gemini_tools = None
        if tools:
            gemini_tools = [
                self._convert_tool_to_gemini(t) for t in tools if t.get("type") == "function"
            ]

        cached_content_name, cache_key, request_contents = self._prepare_cached_request(
            messages=messages,
            contents=contents,
            types_module=types,
        )
        gen_config = self._build_generate_config(
            types,
            gemini_tools,
            cached_content_name=cached_content_name,
        )

        try:
            try:
                response = self._client.models.generate_content(
                    model=self.config.model_id,
                    contents=request_contents,
                    config=gen_config,
                )
            except Exception as exc:
                if not (cached_content_name and self._looks_like_cache_error(exc)):
                    raise
                if cache_key:
                    self._cached_content_names.pop(cache_key, None)
                if self._cache_settings.strict:
                    raise RuntimeError(f"Gemini cache required but failed: {exc}") from exc
                logger.warning("Gemini cached content failed, retrying uncached: %s", exc)
                response = self._client.models.generate_content(
                    model=self.config.model_id,
                    contents=contents,
                    config=self._build_generate_config(types, gemini_tools),
                )

            # Parse response
            content = ""
            tool_calls = []

            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        content += part.text
                    elif hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        tool_calls.append(
                            ToolCall(
                                name=fc.name,
                                arguments=dict(fc.args) if fc.args else {},
                                thought_signature=self._encode_thought_signature(
                                    getattr(part, "thought_signature", None)
                                ),
                            )
                        )

            return GenerateResult(
                content=content,
                tool_calls=tool_calls,
                finish_reason="tool_calls" if tool_calls else "stop",
                usage={
                    "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                    "completion_tokens": getattr(
                        response.usage_metadata, "candidates_token_count", 0
                    ),
                    "cached_content_tokens": getattr(
                        response.usage_metadata,
                        "cached_content_token_count",
                        0,
                    ),
                    "total_tokens": getattr(response.usage_metadata, "total_token_count", 0),
                },
                raw_response=response,
            )

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise

    def _messages_to_gemini_contents(
        self, messages: list[dict[str, Any]], types_module
    ) -> list[Any]:
        contents: list[Any] = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "user":
                contents.append(
                    types_module.Content(role="user", parts=[types_module.Part(text=content)])
                )
            elif role == "assistant":
                parts = []
                if content:
                    parts.append(types_module.Part(text=content))
                for tool_call in msg.get("tool_calls", []):
                    if not isinstance(tool_call, dict):
                        continue
                    name = str(tool_call.get("name", "")).strip()
                    if not name:
                        continue
                    part_kwargs: dict[str, Any] = {
                        "function_call": types_module.FunctionCall(
                            name=name,
                            args=tool_call.get("arguments") or {},
                        )
                    }
                    signature = self._decode_thought_signature(tool_call.get("thought_signature"))
                    if signature:
                        part_kwargs["thought_signature"] = signature
                    parts.append(types_module.Part(**part_kwargs))
                if parts:
                    contents.append(types_module.Content(role="model", parts=parts))
            elif role == "tool":
                results = msg.get("results", [])
                parts = []
                for result in results:
                    parts.append(
                        types_module.Part(
                            function_response=types_module.FunctionResponse(
                                name=result.get("name", "unknown"),
                                response={"result": result.get("content", "")},
                            )
                        )
                    )
                if parts:
                    contents.append(types_module.Content(role="user", parts=parts))
        return contents

    def _build_generate_config(
        self,
        types_module,
        gemini_tools: list[dict[str, Any]] | None,
        *,
        cached_content_name: str | None = None,
    ):
        config_kwargs: dict[str, Any] = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_output_tokens": self.config.max_tokens,
        }
        if self._thinking_settings.level:
            config_kwargs["thinking_config"] = types_module.ThinkingConfig(
                thinking_level=self._thinking_settings.level,
            )
        gen_config = types_module.GenerateContentConfig(
            **config_kwargs,
        )
        if cached_content_name:
            gen_config.cached_content = cached_content_name
        elif self.config.system_prompt:
            gen_config.system_instruction = self.config.system_prompt

        if gemini_tools:
            gen_config.tools = [types_module.Tool(function_declarations=gemini_tools)]
        return gen_config

    def _prepare_cached_request(
        self,
        *,
        messages: list[dict[str, Any]],
        contents: list[Any],
        types_module,
    ) -> tuple[str | None, str | None, list[Any]]:
        if not self._cache_settings.enabled:
            return None, None, contents
        if len(contents) < 2:
            return None, None, contents

        prefix_messages = [msg for msg in messages if msg.get("role") != "system"][:-1]
        cache_fingerprint = json.dumps(
            {
                "model": self.config.model_id,
                "system_prompt": self.config.system_prompt,
                "messages": prefix_messages,
            },
            sort_keys=True,
            ensure_ascii=True,
            default=str,
        )
        if len(cache_fingerprint) < self._cache_settings.min_prefix_chars:
            return None, None, contents

        cache_key = hashlib.sha256(cache_fingerprint.encode("utf-8")).hexdigest()
        cache_name = self._cached_content_names.get(cache_key)
        if cache_name is None:
            cache_name = self._create_cached_content(
                types_module=types_module,
                prefix_contents=contents[:-1],
                cache_key=cache_key,
            )
        if cache_name is None:
            return None, None, contents
        return cache_name, cache_key, contents[-1:]

    def _create_cached_content(
        self,
        *,
        types_module,
        prefix_contents: list[Any],
        cache_key: str,
    ) -> str | None:
        try:
            cache = self._client.caches.create(
                model=self.config.model_id,
                config=types_module.CreateCachedContentConfig(
                    contents=prefix_contents,
                    system_instruction=self.config.system_prompt or None,
                    ttl=self._cache_settings.ttl,
                    display_name=f"afs-{cache_key[:12]}",
                ),
            )
        except Exception as exc:
            if self._cache_settings.strict:
                raise RuntimeError(f"Gemini cache required but creation failed: {exc}") from exc
            logger.warning("Gemini cached content creation failed, continuing uncached: %s", exc)
            return None

        cache_name = str(getattr(cache, "name", "") or "").strip()
        if not cache_name:
            if self._cache_settings.strict:
                raise RuntimeError("Gemini cache required but create returned no cache name")
            return None
        self._cached_content_names[cache_key] = cache_name
        return cache_name

    def _looks_like_cache_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "cached" in message or "cache" in message

    @staticmethod
    def _encode_thought_signature(value: Any) -> str:
        if isinstance(value, bytes) and value:
            return base64.b64encode(value).decode("ascii")
        return ""

    @staticmethod
    def _decode_thought_signature(value: Any) -> bytes | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            return base64.b64decode(value, validate=True)
        except (ValueError, TypeError):
            return None

    def _convert_tool_to_gemini(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI tool format to Gemini FunctionDeclaration."""
        func = tool["function"]
        return {
            "name": func["name"],
            "description": func.get("description", ""),
            "parameters": func.get("parameters", {"type": "object", "properties": {}}),
        }

    async def close(self) -> None:
        """No cleanup needed for Gemini."""
        pass


def create_backend(config: ModelConfig | str) -> ModelBackend:
    """Create the appropriate backend for a model config.

    Args:
        config: ModelConfig or string shorthand like "ollama:llama3.2"
                or "lmstudio:gguf/qwen2.5-coder-7b-instruct.gguf"

    Returns:
        Appropriate ModelBackend instance
    """
    if isinstance(config, str):
        config = ModelConfig.from_string(config)

    if config.provider == ModelProvider.OLLAMA:
        return OllamaBackend(config)
    elif config.provider == ModelProvider.LMSTUDIO:
        return LMStudioBackend(config)
    elif config.provider == ModelProvider.GEMINI:
        return GeminiBackend(config)
    elif config.provider == ModelProvider.ANTHROPIC:
        transport = (
            str(
                config.extra.get("anthropic_transport")
                or os.getenv("AFS_ANTHROPIC_TRANSPORT")
                or ""
            )
            .strip()
            .lower()
        )
        has_native_config = bool(
            config.extra.get("anthropic_base_url")
            or os.getenv("AFS_ANTHROPIC_BASE_URL")
            or os.getenv("ANTHROPIC_BASE_URL")
            or os.getenv("AFS_ANTHROPIC_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
        )
        has_openai_gateway = bool(os.getenv("LITELLM_BASE_URL") or os.getenv("OPENROUTER_BASE_URL"))
        use_openai_gateway = transport in {"openai", "litellm", "openrouter"} or (
            not transport and not has_native_config and has_openai_gateway
        )
        if not use_openai_gateway:
            return AnthropicBackend(
                config,
                api_key=(
                    str(config.extra.get("anthropic_api_key") or "").strip()
                    or os.getenv("AFS_ANTHROPIC_API_KEY")
                    or os.getenv("ANTHROPIC_API_KEY")
                ),
                base_url=(
                    str(config.extra.get("anthropic_base_url") or "").strip()
                    or os.getenv("AFS_ANTHROPIC_BASE_URL")
                    or os.getenv("ANTHROPIC_BASE_URL")
                ),
            )
        base_url = (
            os.getenv("LITELLM_BASE_URL")
            or os.getenv("OPENROUTER_BASE_URL")
            or "http://localhost:4000/v1"
        )
        api_key = (
            os.getenv("LITELLM_MASTER_KEY")
            or os.getenv("LITELLM_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
        )
        return OpenAIBackend(config, base_url=base_url, api_key=api_key, require_key=True)
    elif config.provider == ModelProvider.OPENAI:
        base_url = (
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE_URL")
            or "https://api.openai.com/v1"
        )
        api_key = os.getenv("OPENAI_API_KEY")
        return OpenAIBackend(config, base_url=base_url, api_key=api_key, require_key=True)
    elif config.provider == ModelProvider.OPENROUTER:
        base_url = os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        api_key = os.getenv("OPENROUTER_API_KEY")
        return OpenAIBackend(config, base_url=base_url, api_key=api_key, require_key=True)
    elif config.provider == ModelProvider.LITELLM:
        base_url = os.getenv("LITELLM_BASE_URL") or "http://localhost:4000/v1"
        api_key = os.getenv("LITELLM_MASTER_KEY") or os.getenv("LITELLM_API_KEY")
        return OpenAIBackend(config, base_url=base_url, api_key=api_key, require_key=False)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")
