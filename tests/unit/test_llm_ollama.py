"""Tests for :class:`agent_memory_benchmark.llm.ollama.OllamaProvider`.

Uses ``respx`` to mock the Ollama HTTP API so tests are hermetic and do
not require a running Ollama daemon.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from agent_memory_benchmark.llm.ollama import OllamaError, OllamaProvider


@pytest.fixture
def base_url() -> str:
    return "http://ollama.test:11434"


@pytest.fixture
def provider(base_url: str) -> OllamaProvider:
    return OllamaProvider(model="llama3.1:8b", base_url=base_url, timeout=10.0)


async def test_chat_posts_to_api_chat_and_returns_content(
    provider: OllamaProvider, base_url: str
) -> None:
    with respx.mock(assert_all_called=True) as router:
        route = router.post(f"{base_url}/api/chat").respond(
            json={
                "model": "llama3.1:8b",
                "message": {"role": "assistant", "content": "hello world"},
                "prompt_eval_count": 11,
                "eval_count": 2,
            }
        )
        result = await provider.chat(system="be nice", user="hi")
    assert result.text == "hello world"
    assert result.model == "llama3.1:8b"
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 2

    request = route.calls.last.request
    assert request.url == httpx.URL(f"{base_url}/api/chat")
    import json as _json

    body = _json.loads(request.content)
    assert body["model"] == "llama3.1:8b"
    assert body["stream"] is False
    assert body["messages"] == [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hi"},
    ]
    assert body["options"]["temperature"] == 0.0
    assert body["options"]["num_ctx"] == 8192
    assert "format" not in body


async def test_chat_omits_system_when_blank(provider: OllamaProvider, base_url: str) -> None:
    with respx.mock(assert_all_called=True) as router:
        route = router.post(f"{base_url}/api/chat").respond(json={"message": {"content": "ok"}})
        await provider.chat(user="hi")
    import json as _json

    body = _json.loads(route.calls.last.request.content)
    assert body["messages"] == [{"role": "user", "content": "hi"}]


async def test_chat_sets_json_mode_and_max_tokens(provider: OllamaProvider, base_url: str) -> None:
    with respx.mock(assert_all_called=True) as router:
        route = router.post(f"{base_url}/api/chat").respond(
            json={"message": {"content": '{"ok": true}'}}
        )
        await provider.chat(user="q", json_mode=True, max_tokens=42, temperature=0.7)
    import json as _json

    body = _json.loads(route.calls.last.request.content)
    assert body["format"] == "json"
    assert body["options"]["temperature"] == 0.7
    assert body["options"]["num_predict"] == 42


async def test_chat_raises_on_http_error(provider: OllamaProvider, base_url: str) -> None:
    with respx.mock() as router:
        router.post(f"{base_url}/api/chat").respond(500, text="boom")
        with pytest.raises(OllamaError) as exc_info:
            await provider.chat(user="q")
    assert exc_info.value.status_code == 500


async def test_resolve_spec_appends_digest(provider: OllamaProvider, base_url: str) -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{base_url}/api/tags").respond(
            json={
                "models": [
                    {
                        "name": "mistral:7b",
                        "digest": "sha256:deadbeef",
                    },
                    {
                        "name": "llama3.1:8b",
                        "digest": "sha256:cafef00d",
                    },
                ]
            }
        )
        resolved = await provider.resolve_spec()
    assert resolved == "ollama:llama3.1:8b@sha256:cafef00d"


async def test_resolve_spec_handles_bare_digest(provider: OllamaProvider, base_url: str) -> None:
    """Some Ollama versions return the digest without the ``sha256:`` prefix."""
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{base_url}/api/tags").respond(
            json={"models": [{"name": "llama3.1:8b", "digest": "cafef00d"}]}
        )
        resolved = await provider.resolve_spec()
    assert resolved == "ollama:llama3.1:8b@sha256:cafef00d"


async def test_resolve_spec_falls_back_when_model_not_listed(
    provider: OllamaProvider, base_url: str
) -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{base_url}/api/tags").respond(
            json={"models": [{"name": "mistral:7b", "digest": "sha256:deadbeef"}]}
        )
        resolved = await provider.resolve_spec()
    assert resolved == "ollama:llama3.1:8b"


async def test_close_is_idempotent_for_owned_client(base_url: str) -> None:
    provider = OllamaProvider(model="llama3.1:8b", base_url=base_url)
    await provider.close()
    # Calling close a second time on an already-closed client must not raise.
    await provider.close()


async def test_does_not_close_injected_client(base_url: str) -> None:
    client = httpx.AsyncClient()
    try:
        provider = OllamaProvider(model="llama3.1:8b", base_url=base_url, client=client)
        await provider.close()
        assert not client.is_closed
    finally:
        await client.aclose()
