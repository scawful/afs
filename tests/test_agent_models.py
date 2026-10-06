from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from afs.agent.models import (
    AnthropicBackend,
    GeminiBackend,
    ModelConfig,
    ModelProvider,
    OpenAIBackend,
    ToolCall,
    create_backend,
    resolve_gemini_cache_settings,
    resolve_gemini_thinking_settings,
)


class FakePart:
    def __init__(
        self,
        text=None,
        function_response=None,
        function_call=None,
        thought_signature=None,
    ):
        self.text = text
        self.function_response = function_response
        self.function_call = function_call
        self.thought_signature = thought_signature


class FakeContent:
    def __init__(self, role, parts):
        self.role = role
        self.parts = parts


class FakeFunctionResponse:
    def __init__(self, name, response):
        self.name = name
        self.response = response


class FakeFunctionCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class FakeTool:
    def __init__(self, function_declarations):
        self.function_declarations = function_declarations


class FakeGenerateContentConfig:
    def __init__(self, **kwargs):
        self.temperature = kwargs.get("temperature")
        self.top_p = kwargs.get("top_p")
        self.max_output_tokens = kwargs.get("max_output_tokens")
        self.system_instruction = kwargs.get("system_instruction")
        self.tools = kwargs.get("tools")
        self.cached_content = kwargs.get("cached_content")
        self.thinking_config = kwargs.get("thinking_config")


class FakeThinkingConfig:
    def __init__(self, **kwargs):
        self.thinking_level = kwargs.get("thinking_level")


class FakeCreateCachedContentConfig:
    def __init__(self, **kwargs):
        self.contents = kwargs.get("contents")
        self.system_instruction = kwargs.get("system_instruction")
        self.ttl = kwargs.get("ttl")
        self.display_name = kwargs.get("display_name")


class FakeResponse:
    def __init__(self, text: str, *, cached_content_tokens: int = 0):
        self.candidates = [
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text=text, function_call=None)])
            )
        ]
        self.usage_metadata = SimpleNamespace(
            prompt_token_count=11,
            candidates_token_count=7,
            cached_content_token_count=cached_content_tokens,
            total_token_count=18 + cached_content_tokens,
        )


class FakeCaches:
    def __init__(self, *, fail_create: bool = False):
        self.fail_create = fail_create
        self.create_calls: list[dict[str, object]] = []

    def create(self, *, model, config):
        self.create_calls.append({"model": model, "config": config})
        if self.fail_create:
            raise RuntimeError("cache create failed")
        return SimpleNamespace(name=f"cached/{len(self.create_calls)}")


class FakeModels:
    def __init__(self, *, fail_on_cached: bool = False):
        self.fail_on_cached = fail_on_cached
        self.calls: list[dict[str, object]] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.fail_on_cached and getattr(config, "cached_content", None):
            self.fail_on_cached = False
            raise RuntimeError("cached content expired")
        cached_tokens = 24 if getattr(config, "cached_content", None) else 0
        return FakeResponse("ok", cached_content_tokens=cached_tokens)


class FakeClient:
    def __init__(self, *, fail_create: bool = False, fail_on_cached: bool = False):
        self.caches = FakeCaches(fail_create=fail_create)
        self.models = FakeModels(fail_on_cached=fail_on_cached)


def _install_fake_gemini(monkeypatch, client: FakeClient) -> None:
    fake_types = SimpleNamespace(
        Content=FakeContent,
        Part=FakePart,
        FunctionResponse=FakeFunctionResponse,
        FunctionCall=FakeFunctionCall,
        Tool=FakeTool,
        GenerateContentConfig=FakeGenerateContentConfig,
        ThinkingConfig=FakeThinkingConfig,
        CreateCachedContentConfig=FakeCreateCachedContentConfig,
    )
    fake_genai = ModuleType("google.genai")
    fake_genai.Client = lambda *args, **kwargs: client
    fake_genai.types = fake_types
    fake_google = ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)


def _gemini_messages() -> list[dict[str, object]]:
    return [
        {"role": "user", "content": "Large repeated context " * 50},
        {"role": "user", "content": "What changed?"},
    ]


def test_anthropic_backend_is_native_and_caches_stable_system(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("AFS_ANTHROPIC_TRANSPORT", raising=False)
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="reviewed")],
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=3,
            cache_creation_input_tokens=8,
            cache_read_input_tokens=0,
        ),
    )
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)
    fake_anthropic = ModuleType("anthropic")
    fake_anthropic.AsyncAnthropic = MagicMock(return_value=client)
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    backend = create_backend(
        ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_id="claude-sonnet-5",
            system_prompt="stable policy",
        )
    )
    result = asyncio.run(backend.generate([{"role": "user", "content": "inspect"}]))

    assert isinstance(backend, AnthropicBackend)
    assert result.content == "reviewed"
    assert result.usage["cache_creation_input_tokens"] == 8
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["system"] == [
        {
            "type": "text",
            "text": "stable policy",
            "cache_control": {"type": "ephemeral"},
        }
    ]


def test_anthropic_backend_does_not_duplicate_configured_system_prompt() -> None:
    backend = AnthropicBackend(
        ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_id="claude-sonnet-5",
            system_prompt="stable policy",
        ),
        api_key="test-key",
    )

    system = backend._system_content(
        [
            {"role": "system", "content": "stable policy"},
            {"role": "system", "content": "context was truncated"},
            {"role": "user", "content": "inspect"},
        ]
    )

    assert system == [
        {
            "type": "text",
            "text": "stable policy",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": "context was truncated"},
    ]


def test_anthropic_backend_keeps_explicit_openai_gateway(monkeypatch) -> None:
    monkeypatch.setenv("AFS_ANTHROPIC_TRANSPORT", "openai")
    monkeypatch.setenv("LITELLM_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("LITELLM_API_KEY", "gateway-key")

    backend = create_backend("anthropic:claude-sonnet-5")

    assert isinstance(backend, OpenAIBackend)
    assert backend.base_url == "https://gateway.example/v1"


def test_anthropic_backend_converts_tools_and_parses_tool_calls() -> None:
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="checking"),
            SimpleNamespace(
                type="tool_use",
                id="call-1",
                name="context_status",
                input={"project_path": "."},
            ),
        ],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=9, output_tokens=4),
    )
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)
    backend = AnthropicBackend(
        ModelConfig(provider=ModelProvider.ANTHROPIC, model_id="claude-sonnet-5"),
        api_key="test-key",
    )
    backend._client = client

    result = asyncio.run(
        backend.generate(
            [{"role": "user", "content": "inspect"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "context_status",
                        "description": "Read context health",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )
    )

    assert result.content == "checking"
    assert result.tool_calls == [
        ToolCall(
            name="context_status",
            arguments={"project_path": "."},
            id="call-1",
        )
    ]
    assert client.messages.create.call_args.kwargs["tools"] == [
        {
            "name": "context_status",
            "description": "Read context health",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]


def test_resolve_gemini_cache_settings_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("AFS_GEMINI_CACHE_MODE", "try")
    monkeypatch.setenv("AFS_GEMINI_CACHE_TTL", "90s")
    monkeypatch.setenv("AFS_GEMINI_CACHE_MIN_CHARS", "123")

    settings = resolve_gemini_cache_settings(
        ModelConfig(provider=ModelProvider.GEMINI, model_id="gemini-1.5-flash-001")
    )

    assert settings.mode == "try"
    assert settings.ttl == "90s"
    assert settings.min_prefix_chars == 123


def test_resolve_gemini_thinking_settings_is_optional_and_configurable(monkeypatch) -> None:
    config = ModelConfig(provider=ModelProvider.GEMINI, model_id="gemini-3.7-flash")
    assert resolve_gemini_thinking_settings(config).level is None

    monkeypatch.setenv("AFS_GEMINI_THINKING_LEVEL", "medium")
    assert resolve_gemini_thinking_settings(config).level == "medium"

    config.extra["gemini_thinking"] = {"level": "high"}
    assert resolve_gemini_thinking_settings(config).level == "high"


def test_resolve_gemini_thinking_settings_rejects_unknown_level() -> None:
    config = ModelConfig(
        provider=ModelProvider.GEMINI,
        model_id="gemini-3.7-flash",
        extra={"gemini_thinking_level": "minimal"},
    )
    with pytest.raises(ValueError, match="invalid Gemini thinking level"):
        resolve_gemini_thinking_settings(config)


def test_gemini_backend_passes_thinking_level(monkeypatch) -> None:
    client = FakeClient()
    _install_fake_gemini(monkeypatch, client)
    backend = GeminiBackend(
        ModelConfig(
            provider=ModelProvider.GEMINI,
            model_id="gemini-3.7-flash",
            extra={"gemini_thinking_level": "medium"},
        )
    )

    asyncio.run(backend.generate([{"role": "user", "content": "review this"}]))

    thinking = client.models.calls[0]["config"].thinking_config
    assert thinking.thinking_level == "medium"


def test_gemini_backend_preserves_tool_call_thought_signature(monkeypatch) -> None:
    client = FakeClient()
    _install_fake_gemini(monkeypatch, client)
    backend = GeminiBackend(ModelConfig(provider=ModelProvider.GEMINI, model_id="gemini-3.7-flash"))
    signature = backend._encode_thought_signature(b"provider-state")

    contents = backend._messages_to_gemini_contents(
        [
            {"role": "user", "content": "inspect"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "name": "context_status",
                        "arguments": {"path": "."},
                        "thought_signature": signature,
                    }
                ],
            },
        ],
        sys.modules["google.genai"].types,
    )

    tool_part = contents[1].parts[0]
    assert tool_part.function_call.name == "context_status"
    assert tool_part.function_call.args == {"path": "."}
    assert tool_part.thought_signature == b"provider-state"


def test_gemini_backend_uses_configurable_cached_content(monkeypatch) -> None:
    client = FakeClient()
    _install_fake_gemini(monkeypatch, client)
    backend = GeminiBackend(
        ModelConfig(
            provider=ModelProvider.GEMINI,
            model_id="gemini-1.5-flash-001",
            system_prompt="System prompt",
            extra={"gemini_cache": {"mode": "try", "ttl": "120s", "min_chars": 1}},
        )
    )

    result = asyncio.run(backend.generate(_gemini_messages()))

    assert len(client.caches.create_calls) == 1
    cache_config = client.caches.create_calls[0]["config"]
    assert cache_config.ttl == "120s"
    assert cache_config.system_instruction == "System prompt"
    generate_call = client.models.calls[0]
    assert generate_call["config"].cached_content == "cached/1"
    assert len(generate_call["contents"]) == 1
    assert result.usage["cached_content_tokens"] == 24


def test_gemini_backend_falls_back_uncached_when_cache_lookup_fails(monkeypatch) -> None:
    client = FakeClient(fail_on_cached=True)
    _install_fake_gemini(monkeypatch, client)
    backend = GeminiBackend(
        ModelConfig(
            provider=ModelProvider.GEMINI,
            model_id="gemini-1.5-flash-001",
            system_prompt="System prompt",
            extra={"gemini_cache_mode": "try", "gemini_cache_min_chars": 1},
        )
    )

    asyncio.run(backend.generate(_gemini_messages()))

    assert len(client.models.calls) == 2
    assert client.models.calls[0]["config"].cached_content == "cached/1"
    assert client.models.calls[1]["config"].cached_content is None
    assert len(client.models.calls[1]["contents"]) == 2


def test_gemini_backend_required_cache_raises_on_create_failure(monkeypatch) -> None:
    client = FakeClient(fail_create=True)
    _install_fake_gemini(monkeypatch, client)
    backend = GeminiBackend(
        ModelConfig(
            provider=ModelProvider.GEMINI,
            model_id="gemini-1.5-flash-001",
            extra={"gemini_cache": {"mode": "required", "min_chars": 1}},
        )
    )

    with pytest.raises(RuntimeError, match="Gemini cache required"):
        asyncio.run(backend.generate(_gemini_messages()))
