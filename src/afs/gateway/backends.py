"""Backend management for AFS Gateway."""

from __future__ import annotations

import asyncio
import logging
import subprocess
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


class BackendType(str, Enum):
    """Available backend types."""
    LOCAL = "local"
    WINDOWS = "windows"
    VASTAI = "vastai"


@dataclass
class BackendConfig:
    """Configuration for a single backend."""
    name: str
    type: BackendType
    host: str
    port: int = 11434
    enabled: bool = True
    priority: int = 0  # Higher = preferred
    ssh_host: str | None = None  # For tunnel-based backends
    ssh_port: int = 22

    # vast.ai specific
    vastai_instance_id: str | None = None
    vastai_image: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0"
    vastai_gpu_type: str = "RTX_4090"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class BackendStatus:
    """Current status of a backend."""
    healthy: bool = False
    last_check: float = 0
    error: str | None = None
    models: list[str] = field(default_factory=list)


class BackendManager:
    """Manages multiple inference backends with failover."""

    DEFAULT_CONFIG_PATH = Path.home() / ".config" / "afs" / "backends.json"

    def __init__(self, backends: list[BackendConfig] | None = None):
        self.backends = backends or self._default_backends()
        self.status: dict[str, BackendStatus] = {}
        self._active_backend: str | None = None
        self._client: httpx.AsyncClient | None = None
        self._vast_process: subprocess.Popen | None = None

    def _default_backends(self) -> list[BackendConfig]:
        """Default backend configuration."""
        return [
            BackendConfig(
                name="local",
                type=BackendType.LOCAL,
                host="localhost",
                port=11434,
                priority=1,
            ),
            BackendConfig(
                name="windows",
                type=BackendType.WINDOWS,
                host="localhost",
                port=11435,  # SSH tunnel from Windows
                ssh_host="halext-nj",
                priority=2,
            ),
            BackendConfig(
                name="vastai",
                type=BackendType.VASTAI,
                host="localhost",
                port=11436,
                priority=0,  # On-demand only
                enabled=False,
            ),
        ]

    async def __aenter__(self) -> BackendManager:
        self._client = httpx.AsyncClient(timeout=30.0)
        await self.check_all()
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()
        if self._vast_process:
            self._vast_process.terminate()

    async def check_health(self, backend: BackendConfig) -> BackendStatus:
        """Check health of a single backend."""
        import time
        status = BackendStatus(last_check=time.time())

        try:
            resp = await self._client.get(f"{backend.base_url}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                status.healthy = True
                status.models = [m["name"] for m in data.get("models", [])]
            else:
                status.error = f"HTTP {resp.status_code}"
        except Exception as e:
            status.error = str(e)

        self.status[backend.name] = status
        return status

    async def check_all(self) -> dict[str, BackendStatus]:
        """Check health of all enabled backends."""
        tasks = []
        for backend in self.backends:
            if backend.enabled:
                tasks.append(self.check_health(backend))

        await asyncio.gather(*tasks, return_exceptions=True)

        # Select best available backend
        self._select_active()
        return self.status

    def _select_active(self) -> None:
        """Select the best available backend."""
        candidates = [
            (b.name, b.priority)
            for b in self.backends
            if b.enabled and self.status.get(b.name, BackendStatus()).healthy
        ]

        if candidates:
            # Sort by priority descending
            candidates.sort(key=lambda x: x[1], reverse=True)
            self._active_backend = candidates[0][0]
            logger.info(f"Active backend: {self._active_backend}")
        else:
            self._active_backend = None
            logger.warning("No healthy backends available")

    @property
    def active(self) -> BackendConfig | None:
        """Get currently active backend."""
        if not self._active_backend:
            return None
        return next(
            (b for b in self.backends if b.name == self._active_backend),
            None
        )

    def get_backend(self, name: str) -> BackendConfig | None:
        """Get backend by name."""
        return next((b for b in self.backends if b.name == name), None)

    def set_active(self, name: str) -> bool:
        """Manually set active backend."""
        backend = self.get_backend(name)
        if backend and backend.enabled:
            self._active_backend = name
            return True
        return False

    async def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        **kwargs,
    ) -> str | AsyncIterator[str]:
        """Generate completion using active backend."""
        backend = self.active
        if not backend:
            raise RuntimeError("No active backend available")

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            **kwargs,
        }

        if stream:
            return self._stream_generate(backend, payload)
        else:
            return await self._batch_generate(backend, payload)

    async def _batch_generate(
        self,
        backend: BackendConfig,
        payload: dict,
    ) -> str:
        """Non-streaming generation."""
        resp = await self._client.post(
            f"{backend.base_url}/api/generate",
            json=payload,
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    async def _stream_generate(
        self,
        backend: BackendConfig,
        payload: dict,
    ) -> AsyncIterator[str]:
        """Streaming generation."""
        import json
        async with self._client.stream(
            "POST",
            f"{backend.base_url}/api/generate",
            json=payload,
            timeout=120.0,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    data = json.loads(line)
                    if chunk := data.get("response"):
                        yield chunk

    async def chat(
        self,
        model: str,
        messages: list[dict],
        stream: bool = False,
        **kwargs,
    ) -> dict | AsyncIterator[str]:
        """Chat completion using active backend."""
        backend = self.active
        if not backend:
            raise RuntimeError("No active backend available")

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs,
        }

        if stream:
            return self._stream_chat(backend, payload)
        else:
            return await self._batch_chat(backend, payload)

    async def _batch_chat(
        self,
        backend: BackendConfig,
        payload: dict,
    ) -> dict:
        """Non-streaming chat."""
        resp = await self._client.post(
            f"{backend.base_url}/api/chat",
            json=payload,
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()

    async def _stream_chat(
        self,
        backend: BackendConfig,
        payload: dict,
    ) -> AsyncIterator[str]:
        """Streaming chat."""
        import json
        async with self._client.stream(
            "POST",
            f"{backend.base_url}/api/chat",
            json=payload,
            timeout=120.0,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    data = json.loads(line)
                    if chunk := data.get("message", {}).get("content"):
                        yield chunk

    # vast.ai management
    async def provision_vastai(self, gpu_type: str = "RTX_4090") -> bool:
        """Provision a vast.ai instance on-demand."""
        backend = self.get_backend("vastai")
        if not backend:
            logger.error("vast.ai backend not configured")
            return False

        logger.info(f"Provisioning vast.ai instance with {gpu_type}...")

        # This would call vastai CLI to spin up an instance
        # For now, placeholder
        try:
            result = subprocess.run(
                [
                    "vastai", "create", "instance",
                    "--image", backend.vastai_image,
                    "--gpu-type", gpu_type,
                    "--disk", "50",
                    "--onstart", "ollama serve",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                # Parse instance ID and set up SSH tunnel
                backend.enabled = True
                await self.check_health(backend)
                return self.status.get("vastai", BackendStatus()).healthy
            else:
                logger.error(f"vast.ai provision failed: {result.stderr}")
                return False

        except FileNotFoundError:
            logger.error("vastai CLI not installed. Run: pip install vastai")
            return False

    async def teardown_vastai(self) -> bool:
        """Tear down vast.ai instance."""
        backend = self.get_backend("vastai")
        if not backend or not backend.vastai_instance_id:
            return False

        try:
            subprocess.run(
                ["vastai", "destroy", "instance", backend.vastai_instance_id],
                check=True,
            )
            backend.enabled = False
            backend.vastai_instance_id = None
            self.status["vastai"] = BackendStatus()
            self._select_active()
            return True
        except Exception as e:
            logger.error(f"Failed to teardown vast.ai: {e}")
            return False
