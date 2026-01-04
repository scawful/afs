"""OpenAI-compatible API models for AFS Gateway."""

from __future__ import annotations

import time
import uuid
from typing import Literal
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat message."""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str
    messages: list[Message]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int | None = None
    stream: bool = False
    stop: list[str] | str | None = None


class ChatChoice(BaseModel):
    """Chat completion choice."""
    index: int = 0
    message: Message
    finish_reason: Literal["stop", "length", "error"] = "stop"


class Usage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatChoice]
    usage: Usage = Field(default_factory=Usage)


class DeltaMessage(BaseModel):
    """Streaming delta message."""
    role: str | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    """Streaming choice."""
    index: int = 0
    delta: DeltaMessage
    finish_reason: Literal["stop", "length", "error"] | None = None


class StreamResponse(BaseModel):
    """Streaming chunk response."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]


class Model(BaseModel):
    """Model info for /v1/models endpoint."""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "afs"

    # AFS-specific metadata
    expert_name: str | None = None
    description: str | None = None
    intent: str | None = None
    backend: str = "local"


class ModelsResponse(BaseModel):
    """Response for /v1/models endpoint."""
    object: str = "list"
    data: list[Model]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    backends: dict[str, dict]
    active_backend: str
    version: str = "0.1.0"
