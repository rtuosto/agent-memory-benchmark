"""Thin facade around an :class:`LLMProvider` for judge-style invocations.

The judge layer (PR-6 / PR-9 / PR-11) composes a ``JudgeClient`` around a
provider to share three concerns:

- **Retries.** Transient errors (HTTP 429 and 5xx from the underlying
  provider) are retried with exponential backoff and jitter, bounded by
  ``max_retries``. Non-transient errors bubble up unchanged.
- **Multi-run concurrency.** LOCOMO's majority-vote judging runs N
  copies of the same prompt at ``temperature > 0`` and takes the majority;
  :meth:`complete_runs` dispatches them with ``asyncio.gather``.
- **Uniform invocation shape.** Judge prompts are plain user-message text;
  ``complete`` accepts the prompt and forwards to the provider.

This facade is deliberately stateless with respect to the benchmark's
cache — cache keys live in ``agent_memory_benchmark.cache`` and sit above
this layer, not inside it.
"""

from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING

import httpx

from . import LLMProvider

if TYPE_CHECKING:
    from .openai import OpenAIError  # noqa: F401 — for doc cross-reference only


class JudgeClient:
    """Async judge caller with retries and concurrent multi-run support.

    ``provider`` is any object that matches the :class:`LLMProvider`
    Protocol. ``temperature`` and ``max_tokens`` are defaults applied when
    the caller does not override per-call. ``max_retries`` counts retries
    beyond the first attempt (so ``max_retries=2`` means up to 3 attempts
    total). ``base_backoff`` is seconds for the first retry; subsequent
    retries double with ±25% jitter, capped at 30s.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        max_retries: int = 2,
        base_backoff: float = 1.0,
    ) -> None:
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.base_backoff = base_backoff

    @property
    def model(self) -> str:
        return self.provider.model

    @property
    def spec(self) -> str:
        return self.provider.spec

    async def complete(
        self,
        user: str,
        *,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """Run the judge prompt once, with retries on transient failures.

        Returns the raw assistant text (callers parse their own label).
        """

        effective_temperature = self.temperature if temperature is None else temperature
        effective_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                result = await self.provider.chat(
                    system=system,
                    user=user,
                    temperature=effective_temperature,
                    max_tokens=effective_max_tokens,
                    json_mode=json_mode,
                )
                return result.text
            except Exception as exc:  # noqa: BLE001 — intentional: classify below
                if not _is_transient(exc) or attempt == self.max_retries:
                    raise
                last_exc = exc
                await asyncio.sleep(self._backoff(attempt))
        # Unreachable: loop either returns or raises.
        assert last_exc is not None  # pragma: no cover
        raise last_exc  # pragma: no cover

    async def complete_runs(
        self,
        user: str,
        *,
        runs: int,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> list[str]:
        """Run the same prompt ``runs`` times concurrently and return all texts."""

        if runs <= 0:
            raise ValueError(f"runs must be positive, got {runs!r}")
        coros = [
            self.complete(
                user,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
            for _ in range(runs)
        ]
        return list(await asyncio.gather(*coros))

    def _backoff(self, attempt: int) -> float:
        base: float = self.base_backoff * (2**attempt)
        jitter: float = base * 0.25 * (random.random() * 2 - 1)
        clamped: float = min(max(base + jitter, 0.0), 30.0)
        return clamped


def _is_transient(exc: BaseException) -> bool:
    """Return True for errors worth retrying (429, 5xx, connect/read errors).

    Looks for an explicit ``status_code`` attribute (provider wrappers set
    this), then falls back to ``httpx`` exception types.
    """

    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status == 429 or 500 <= status < 600):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        http_status = exc.response.status_code
        return http_status == 429 or 500 <= http_status < 600
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


__all__ = ["JudgeClient"]
