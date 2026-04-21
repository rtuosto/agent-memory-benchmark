"""Tests for :class:`agent_memory_benchmark.llm.openai.OpenAIProvider`.

Uses ``respx`` to intercept the HTTP requests the OpenAI SDK makes, so we
exercise the actual request/response plumbing without touching the network.
"""

from __future__ import annotations

import json

import pytest
import respx

pytest.importorskip(
    "openai",
    reason="OpenAIProvider tests require the optional [openai] extra.",
)

from agent_memory_benchmark.llm.openai import (  # noqa: E402 — after importorskip
    OpenAIError,
    OpenAIProvider,
)

_BASE_URL = "https://api.openai.test/v1"


@pytest.fixture
def provider() -> OpenAIProvider:
    return OpenAIProvider(
        model="gpt-4o-mini-2024-07-18",
        api_key="test-key",
        base_url=_BASE_URL,
    )


def _completion_payload(
    content: str, *, prompt_tokens: int = 5, completion_tokens: int = 7
) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o-mini-2024-07-18",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def test_chat_returns_message_content_and_usage(provider: OpenAIProvider) -> None:
    with respx.mock(assert_all_called=True) as router:
        route = router.post(f"{_BASE_URL}/chat/completions").respond(
            json=_completion_payload("howdy")
        )
        result = await provider.chat(system="s", user="u")
    assert result.text == "howdy"
    assert result.model == "gpt-4o-mini-2024-07-18"
    assert result.prompt_tokens == 5
    assert result.completion_tokens == 7

    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "gpt-4o-mini-2024-07-18"
    assert body["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    assert body["temperature"] == 0.0
    assert "response_format" not in body


async def test_chat_omits_blank_system_message(provider: OpenAIProvider) -> None:
    with respx.mock(assert_all_called=True) as router:
        route = router.post(f"{_BASE_URL}/chat/completions").respond(json=_completion_payload("ok"))
        await provider.chat(user="u")
    body = json.loads(route.calls.last.request.content)
    assert body["messages"] == [{"role": "user", "content": "u"}]


async def test_chat_forwards_json_mode_and_max_tokens(provider: OpenAIProvider) -> None:
    with respx.mock(assert_all_called=True) as router:
        route = router.post(f"{_BASE_URL}/chat/completions").respond(
            json=_completion_payload('{"label":"CORRECT"}')
        )
        await provider.chat(user="u", json_mode=True, max_tokens=32, temperature=0.5)
    body = json.loads(route.calls.last.request.content)
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_tokens"] == 32
    assert body["temperature"] == 0.5


async def test_resolve_spec_echoes(provider: OpenAIProvider) -> None:
    assert await provider.resolve_spec() == "openai:gpt-4o-mini-2024-07-18"


async def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(OpenAIError, match="OPENAI_API_KEY"):
        OpenAIProvider(model="gpt-4o-mini-2024-07-18")


async def test_close_is_safe() -> None:
    provider = OpenAIProvider(
        model="gpt-4o-mini-2024-07-18", api_key="test-key", base_url=_BASE_URL
    )
    await provider.close()
