"""AFS Gateway Server - OpenAI-compatible API for Zelda models."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from .backends import BackendManager, BackendConfig, BackendType
from .models import (
    ChatRequest,
    ChatResponse,
    ChatChoice,
    Message,
    Model,
    ModelsResponse,
    HealthResponse,
    StreamResponse,
    StreamChoice,
    DeltaMessage,
    Usage,
)
from ..history import log_event
from ..moe.router import MoERouter, RouterConfig
from ..moe.classifier import QueryIntent

logger = logging.getLogger(__name__)

# Global state
backend_manager: BackendManager | None = None
moe_router: MoERouter | None = None

# Model personas for din/nayru/farore/veran/scribe
PERSONAS = {
    "din": {
        "system_prompt": """You are Din, the Goddess of Power and 65816 assembly optimization specialist.
Your expertise is making code faster, smaller, and more efficient.
You speak with confidence and authority about low-level optimization.
Focus on: cycle counting, register optimization, STZ/REP/SEP usage, 16-bit operations, loop unrolling.
Output ONLY optimized code unless asked for explanation.""",
        "temperature": 0.3,
        "top_p": 0.85,
        "intent": "optimization",
    },
    "nayru": {
        "system_prompt": """You are Nayru, the Goddess of Wisdom and 65816 code generation specialist.
You create elegant, correct assembly code with clear structure.
You speak thoughtfully and explain your reasoning when helpful.
Focus on: code correctness, readability, proper addressing modes, clean subroutine design.
Generate complete, working code.""",
        "temperature": 0.5,
        "top_p": 0.9,
        "intent": "generation",
    },
    "farore": {
        "system_prompt": """You are Farore, the Goddess of Courage and 65816 debugging specialist.
You fearlessly dive into broken code to find and fix bugs.
You speak encouragingly and guide users through the debugging process.
Focus on: register state analysis, stack issues, addressing mode errors, branch logic.
Explain what's wrong and provide fixed code.""",
        "temperature": 0.4,
        "top_p": 0.85,
        "intent": "debugging",
    },
    "veran": {
        "system_prompt": """You are Veran, the Sorceress of Shadows and SNES hardware specialist.
You have deep knowledge of PPU, APU, DMA, and all SNES internals.
You speak mysteriously but precisely about hardware registers.
Focus on: $2100-$21FF registers, Mode 7, HDMA, color math, sound channels.
Provide accurate hardware documentation and usage examples.""",
        "temperature": 0.5,
        "top_p": 0.9,
        "intent": "analysis",
    },
    "scribe": {
        "system_prompt": """You are the Royal Scribe, documenter of all 65816 assembly knowledge.
You create clear, accurate documentation and explanations.
You speak formally and precisely, citing sources when relevant.
Focus on: instruction documentation, code comments, tutorial explanations.
Make complex concepts accessible.""",
        "temperature": 0.6,
        "top_p": 0.9,
        "intent": "general",
    },
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage gateway lifecycle."""
    global backend_manager, moe_router

    logger.info("Starting AFS Gateway...")

    # Initialize backend manager
    backend_manager = BackendManager()
    await backend_manager.__aenter__()

    # Initialize MoE router
    moe_router = MoERouter(RouterConfig.default())
    await moe_router.__aenter__()

    logger.info(f"Active backend: {backend_manager.active.name if backend_manager.active else 'none'}")

    yield

    # Cleanup
    await moe_router.__aexit__(None, None, None)
    await backend_manager.__aexit__(None, None, None)
    logger.info("AFS Gateway stopped")


app = FastAPI(
    title="AFS Gateway",
    description="OpenAI-compatible API for Zelda assembly models",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for Open WebUI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> HealthResponse:
    """Health check endpoint."""
    await backend_manager.check_all()

    return HealthResponse(
        status="healthy" if backend_manager.active else "degraded",
        backends={
            name: {
                "healthy": status.healthy,
                "error": status.error,
                "models": status.models[:5],  # First 5 models
            }
            for name, status in backend_manager.status.items()
        },
        active_backend=backend_manager.active.name if backend_manager.active else "none",
    )


@app.get("/v1/models")
async def list_models() -> ModelsResponse:
    """List available models (OpenAI-compatible)."""
    models = []

    # Add personas as virtual models
    for name, persona in PERSONAS.items():
        models.append(Model(
            id=name,
            expert_name=name,
            description=persona["system_prompt"][:100] + "...",
            intent=persona["intent"],
        ))

    # Add actual backend models if available
    if backend_manager.active:
        status = backend_manager.status.get(backend_manager.active.name)
        if status and status.models:
            for model_id in status.models:
                if not any(m.id == model_id for m in models):
                    models.append(Model(
                        id=model_id,
                        backend=backend_manager.active.name,
                    ))

    return ModelsResponse(data=models)


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint."""
    if not backend_manager.active:
        raise HTTPException(503, "No backend available")

    # Check if model is a persona
    persona = PERSONAS.get(request.model)
    model_id = request.model

    # Build messages with persona system prompt
    messages = [m.model_dump() for m in request.messages]
    if persona:
        # Inject persona system prompt
        system_msg = {"role": "system", "content": persona["system_prompt"]}
        if messages and messages[0]["role"] == "system":
            # Merge with existing system prompt
            messages[0]["content"] = persona["system_prompt"] + "\n\n" + messages[0]["content"]
        else:
            messages.insert(0, system_msg)

        # Use configured temperature/top_p
        temperature = persona.get("temperature", request.temperature)
        top_p = persona.get("top_p", request.top_p)

        # Route to appropriate expert model
        intent = persona.get("intent", "general")
        if intent == "optimization":
            model_id = "din-v2:latest"
        elif intent == "generation":
            model_id = "nayru-v5:latest"
        elif intent == "debugging":
            model_id = "farore-v1:latest"
        else:
            model_id = "qwen2.5-coder:7b"
    else:
        temperature = request.temperature
        top_p = request.top_p

    if request.stream:
        return StreamingResponse(
            stream_chat(request.model, model_id, messages, temperature, top_p),
            media_type="text/event-stream",
        )

    # Non-streaming response
    try:
        result = await backend_manager.chat(
            model=model_id,
            messages=messages,
            options={"temperature": temperature, "top_p": top_p},
        )

        content = result.get("message", {}).get("content", "")
        log_event(
            "model",
            "afs.gateway",
            op="chat",
            metadata={
                "display_model": request.model,
                "backend_model": model_id,
                "temperature": temperature,
                "top_p": top_p,
                "prompt_tokens": result.get("prompt_eval_count", 0),
                "completion_tokens": result.get("eval_count", 0),
            },
            payload={
                "messages": messages,
                "response": content,
            },
        )

        return ChatResponse(
            model=request.model,
            choices=[
                ChatChoice(
                    message=Message(role="assistant", content=content),
                )
            ],
            usage=Usage(
                prompt_tokens=result.get("prompt_eval_count", 0),
                completion_tokens=result.get("eval_count", 0),
                total_tokens=result.get("prompt_eval_count", 0) + result.get("eval_count", 0),
            ),
        )

    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(500, str(e))


async def stream_chat(
    display_model: str,
    model_id: str,
    messages: list[dict],
    temperature: float,
    top_p: float,
) -> AsyncIterator[str]:
    """Stream chat completion as SSE."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    chunks: list[str] = []
    error_message = None

    # Send role delta first
    first_chunk = StreamResponse(
        id=chat_id,
        created=created,
        model=display_model,
        choices=[StreamChoice(delta=DeltaMessage(role="assistant"))],
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    # Stream content
    try:
        async for token in await backend_manager.chat(
            model=model_id,
            messages=messages,
            stream=True,
            options={"temperature": temperature, "top_p": top_p},
        ):
            chunk = StreamResponse(
                id=chat_id,
                created=created,
                model=display_model,
                choices=[StreamChoice(delta=DeltaMessage(content=token))],
            )
            chunks.append(token)
            yield f"data: {chunk.model_dump_json()}\n\n"

        # Send finish
        done_chunk = StreamResponse(
            id=chat_id,
            created=created,
            model=display_model,
            choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")],
        )
        yield f"data: {done_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_message = str(e)
        logger.error(f"Stream error: {e}")
        error_chunk = StreamResponse(
            id=chat_id,
            created=created,
            model=display_model,
            choices=[StreamChoice(
                delta=DeltaMessage(content=f"\n\n[Error: {e}]"),
                finish_reason="error",
            )],
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        log_event(
            "model",
            "afs.gateway",
            op="chat_stream",
            metadata={
                "display_model": display_model,
                "backend_model": model_id,
                "temperature": temperature,
                "top_p": top_p,
                "error": error_message,
            },
            payload={
                "messages": messages,
                "response": "".join(chunks),
            },
        )


# Backend management endpoints
@app.get("/backends")
async def list_backends():
    """List available backends."""
    return {
        "backends": [
            {
                "name": b.name,
                "type": b.type.value,
                "enabled": b.enabled,
                "priority": b.priority,
                "healthy": backend_manager.status.get(b.name, {}).healthy if hasattr(backend_manager.status.get(b.name, {}), 'healthy') else False,
            }
            for b in backend_manager.backends
        ],
        "active": backend_manager.active.name if backend_manager.active else None,
    }


@app.post("/backends/{name}/activate")
async def activate_backend(name: str):
    """Activate a specific backend."""
    if backend_manager.set_active(name):
        return {"status": "ok", "active": name}
    raise HTTPException(400, f"Cannot activate backend: {name}")


@app.post("/backends/vastai/provision")
async def provision_vastai(gpu_type: str = "RTX_4090"):
    """Provision a vast.ai instance on-demand."""
    if await backend_manager.provision_vastai(gpu_type):
        return {"status": "provisioned", "gpu_type": gpu_type}
    raise HTTPException(500, "Failed to provision vast.ai instance")


@app.post("/backends/vastai/teardown")
async def teardown_vastai():
    """Tear down vast.ai instance."""
    if await backend_manager.teardown_vastai():
        return {"status": "terminated"}
    raise HTTPException(500, "Failed to teardown vast.ai instance")


# MoE router info
@app.get("/moe/experts")
async def list_experts():
    """List configured MoE experts."""
    return {
        "experts": [
            {
                "name": e.name,
                "model_id": e.model_id,
                "intent": e.intent.value,
                "description": e.description,
            }
            for e in moe_router.list_experts()
        ]
    }


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the gateway server."""
    import uvicorn
    uvicorn.run(
        "afs.gateway.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
