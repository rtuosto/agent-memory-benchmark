"""Tests for :class:`agent_memory_benchmark.llm.judge_client.JudgeClient`.

Uses a lightweight fake provider to exercise retry logic and multi-run
concurrency without depending on a real HTTP backend.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from agent_memory_benchmark.llm import ChatResult
from agent_memory_benchmark.llm.judge_client import JudgeClient


class _FakeProvider:
    """Provider stub that emits scripted results in order."""

    spec = "fake:model"
    model = "model"

    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    async def chat(
        self,
        *,
        system: str = "",
        user: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_mode": json_mode,
            }
        )
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        assert isinstance(item, str)
        return ChatResult(text=item, model=self.model)

    async def resolve_spec(self) -> str:
        return self.spec

    async def close(self) -> None:
        return None


class _StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero out backoff sleeps so retry tests run instantly."""

    async def _instant(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _instant)


async def test_complete_returns_text_and_forwards_args() -> None:
    provider = _FakeProvider(["answer"])
    judge = JudgeClient(provider, temperature=0.3, max_tokens=64)
    out = await judge.complete("prompt", system="sys", json_mode=True)
    assert out == "answer"
    call = provider.calls[0]
    assert call["user"] == "prompt"
    assert call["system"] == "sys"
    assert call["temperature"] == 0.3
    assert call["max_tokens"] == 64
    assert call["json_mode"] is True


async def test_complete_retries_on_transient_status() -> None:
    provider = _FakeProvider([_StatusError(429), _StatusError(503), "ok"])
    judge = JudgeClient(provider, max_retries=2, base_backoff=0.01)
    assert await judge.complete("q") == "ok"
    assert len(provider.calls) == 3


async def test_complete_gives_up_after_max_retries() -> None:
    provider = _FakeProvider([_StatusError(500), _StatusError(500), _StatusError(500)])
    judge = JudgeClient(provider, max_retries=2, base_backoff=0.01)
    with pytest.raises(_StatusError) as exc_info:
        await judge.complete("q")
    assert exc_info.value.status_code == 500
    assert len(provider.calls) == 3  # initial + 2 retries


async def test_complete_does_not_retry_non_transient() -> None:
    provider = _FakeProvider([_StatusError(400), "unused"])
    judge = JudgeClient(provider, max_retries=3, base_backoff=0.01)
    with pytest.raises(_StatusError):
        await judge.complete("q")
    assert len(provider.calls) == 1


async def test_complete_retries_on_httpx_timeout() -> None:
    provider = _FakeProvider([httpx.ConnectTimeout("slow"), "ok"])
    judge = JudgeClient(provider, max_retries=1, base_backoff=0.01)
    assert await judge.complete("q") == "ok"


async def test_complete_runs_dispatches_concurrently() -> None:
    provider = _FakeProvider(["a", "b", "c"])
    judge = JudgeClient(provider)
    out = await judge.complete_runs("q", runs=3, temperature=0.7)
    assert out == ["a", "b", "c"]
    assert len(provider.calls) == 3
    for call in provider.calls:
        assert call["temperature"] == 0.7


async def test_complete_runs_rejects_non_positive() -> None:
    provider = _FakeProvider([])
    judge = JudgeClient(provider)
    with pytest.raises(ValueError):
        await judge.complete_runs("q", runs=0)


async def test_model_and_spec_delegate_to_provider() -> None:
    provider = _FakeProvider([])
    judge = JudgeClient(provider)
    assert judge.model == "model"
    assert judge.spec == "fake:model"
