"""Rate limiting utilities for API calls."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TypeVar, Callable, Any

T = TypeVar("T")


@dataclass
class RateLimiter:
    """Token bucket rate limiter for API requests."""

    requests_per_minute: int = 60
    _tokens: float = field(init=False)
    _last_update: float = field(init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        self._tokens = float(self.requests_per_minute)
        self._last_update = time.monotonic()

    @property
    def tokens_per_second(self) -> float:
        return self.requests_per_minute / 60.0

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(
            float(self.requests_per_minute),
            self._tokens + elapsed * self.tokens_per_second,
        )
        self._last_update = now

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            self._refill()
            if self._tokens < 1:
                wait_time = (1 - self._tokens) / self.tokens_per_second
                await asyncio.sleep(wait_time)
                self._refill()
            self._tokens -= 1

    def acquire_sync(self) -> None:
        """Synchronous version of acquire."""
        self._refill()
        if self._tokens < 1:
            wait_time = (1 - self._tokens) / self.tokens_per_second
            time.sleep(wait_time)
            self._refill()
        self._tokens -= 1


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0


async def retry_with_backoff(
    func: Callable[[], T],
    config: RetryConfig | None = None,
    on_retry: Callable[[Exception, int], None] | None = None,
) -> T:
    """Execute function with exponential backoff retry."""
    config = config or RetryConfig()
    last_exception: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func()
            return func()
        except Exception as e:
            last_exception = e
            if attempt == config.max_retries:
                raise

            delay = min(
                config.base_delay * (config.exponential_base ** attempt),
                config.max_delay,
            )

            if on_retry:
                on_retry(e, attempt + 1)

            await asyncio.sleep(delay)

    raise last_exception  # type: ignore


def batch_items(items: list[T], batch_size: int) -> list[list[T]]:
    """Split items into batches."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
