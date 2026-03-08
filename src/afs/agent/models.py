"""Model abstraction layer for unified access to local and cloud models.

Supports:
    - Ollama (local models)
    - LMStudio (local GGUF models via OpenAI-compatible API)
    - Google Gemini (cloud)
    - Anthropic Claude (cloud, optional)
    - OpenAI-compatible APIs (cloud or local gateways)
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

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
        ModelConfig(provider=ModelProvider.GEMINI, model_id="gemini-3-flash-preview")

        # From string shorthand
        ModelConfig.from_string("ollama:llama3.2")
        ModelConfig.from_string("gemini-3-flash-preview")  # Defaults to gemini provider
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
        """Parse model string like 'ollama:llama3.2' or 'gemini-3-flash-preview'."""
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

    # Compatibility presets now owned by afs-scawful
    @classmethod
    def din(cls) -> ModelConfig:
        """Compatibility preset for afs-scawful."""
        return _load_scawful_preset("din")

    @classmethod
    def nayru(cls) -> ModelConfig:
        """Compatibility preset for afs-scawful."""
        return _load_scawful_preset("nayru")

    @classmethod
    def farore(cls) -> ModelConfig:
        """Compatibility preset for afs-scawful."""
        return _load_scawful_preset("farore")

    @classmethod
    def veran(cls) -> ModelConfig:
        """Compatibility preset for afs-scawful."""
        return _load_scawful_preset("veran")

    # LMStudio compatibility presets now owned by afs-scawful
    @classmethod
    def din_lmstudio(cls) -> ModelConfig:
        """Compatibility preset for afs-scawful."""
        return _load_scawful_preset("din_lmstudio")

    @classmethod
    def farore_lmstudio(cls) -> ModelConfig:
        """Compatibility preset for afs-scawful."""
        return _load_scawful_preset("farore_lmstudio")

    @classmethod
    def veran_lmstudio(cls) -> ModelConfig:
        """Compatibility preset for afs-scawful."""
        return _load_scawful_preset("veran_lmstudio")

    @classmethod
    def majora_lmstudio(cls) -> ModelConfig:
        """Compatibility preset for afs-scawful."""
        return _load_scawful_preset("majora_lmstudio")


def _load_scawful_preset(name: str) -> ModelConfig:
    """Load an extension-owned preset without baking it into core AFS."""
    try:
        from afs_scawful.agent_model_presets import build_preset
    except Exception as exc:  # pragma: no cover - compatibility path
        raise RuntimeError(
            "Domain-specific model presets moved to the afs-scawful extension."
        ) from exc

    return build_preset(name)


@dataclass
class ToolCall:
    """A tool call requested by the model."""

    name: str
    arguments: dict[str, Any]
    id: str = ""  # Some providers return call IDs


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
        self._client = None

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
            ollama_messages.insert(
                0, {"role": "system", "content": self.config.system_prompt}
            )

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

    def _convert_tools_to_ollama(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
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
        self._client = None
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
        if self.config.system_prompt and (
            not messages or messages[0]["role"] != "system"
        ):
            messages = [{"role": "system", "content": self.config.system_prompt}] + messages

        # Try chat completions first
        if not self._use_completions:
            try:
                return await self._generate_chat(messages, tools)
            except Exception as e:
                error_msg = str(e)
                if "jinja" in error_msg.lower() or "template" in error_msg.lower():
                    logger.warning(f"Chat endpoint failed with template error, falling back to completions: {e}")
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
        self._client = None

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

        if self.config.system_prompt and (
            not messages or messages[0]["role"] != "system"
        ):
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


class GeminiBackend(ModelBackend):
    """Google Gemini model backend."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = None

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
        contents = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            # Map roles
            if role == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=content)]))
            elif role == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part(text=content)]))
            elif role == "tool":
                # Tool results need special handling
                results = msg.get("results", [])
                parts = []
                for result in results:
                    parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=result.get("name", "unknown"),
                                response={"result": result.get("content", "")},
                            )
                        )
                    )
                if parts:
                    contents.append(types.Content(role="user", parts=parts))
            # Skip system messages - handled via system_instruction

        # Convert tools to Gemini format
        gemini_tools = None
        if tools:
            gemini_tools = [self._convert_tool_to_gemini(t) for t in tools if t.get("type") == "function"]

        # Build config
        gen_config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_output_tokens=self.config.max_tokens,
        )

        if self.config.system_prompt:
            gen_config.system_instruction = self.config.system_prompt

        if gemini_tools:
            gen_config.tools = [types.Tool(function_declarations=gemini_tools)]

        try:
            response = self._client.models.generate_content(
                model=self.config.model_id,
                contents=contents,
                config=gen_config,
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
                            )
                        )

            return GenerateResult(
                content=content,
                tool_calls=tool_calls,
                finish_reason="tool_calls" if tool_calls else "stop",
                usage={
                    "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                    "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                },
                raw_response=response,
            )

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise

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
        base_url = (
            os.getenv("AFS_ANTHROPIC_BASE_URL")
            or os.getenv("LITELLM_BASE_URL")
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
