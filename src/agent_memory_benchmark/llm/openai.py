"""OpenAI provider — uses the ``openai`` Python SDK.

Guarded behind the ``[openai]`` extra. If the SDK is not installed,
constructing the provider raises :class:`ProviderUnavailableError` with a
clear hint. OpenAI model IDs are expected to be dated (e.g.
``gpt-4o-mini-2024-07-18``) so they are immutable — :meth:`resolve_spec`
simply echoes the configured spec.

``base_url`` support lets this provider also talk to OpenAI-compatible
gateways (Azure, vLLM, Ollama's ``/v1`` compat shim); the SDK's own types
flow through unchanged.
"""

from __future__ import annotations

import os
from types import TracebackType
from typing import TYPE_CHECKING, Any

from . import ChatResult, ProviderError, ProviderUnavailableError

if TYPE_CHECKING:  # pragma: no cover — import only for type checking
    from openai import AsyncOpenAI


class OpenAIError(ProviderError):
    """Raised when the OpenAI client returns an unexpected payload."""


_SDK_HINT = (
    "The openai package is not installed. Install the optional extra: "
    "`pip install 'agent-memory-benchmark[openai]'`."
)


def _load_async_openai() -> type[AsyncOpenAI]:
    try:
        from openai import AsyncOpenAI as _AsyncOpenAI
    except ModuleNotFoundError as exc:  # pragma: no cover — exercised in tests via monkeypatch
        raise ProviderUnavailableError(_SDK_HINT) from exc
    return _AsyncOpenAI


class OpenAIProvider:
    """Chat against the OpenAI API (or an OpenAI-compatible gateway).

    ``api_key`` defaults to ``$OPENAI_API_KEY``. ``base_url`` defaults to
    ``$OPENAI_BASE_URL`` if set, else the SDK's default.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 600.0,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.model = model
        self.spec = f"openai:{model}"
        if client is None:
            async_openai = _load_async_openai()
            resolved_key = api_key or os.getenv("OPENAI_API_KEY")
            if not resolved_key:
                raise OpenAIError("OPENAI_API_KEY is not set; cannot construct OpenAIProvider.")
            resolved_base = base_url or os.getenv("OPENAI_BASE_URL")
            kwargs: dict[str, Any] = {"api_key": resolved_key, "timeout": timeout}
            if resolved_base:
                kwargs["base_url"] = resolved_base
            client = async_openai(**kwargs)
        self._client: AsyncOpenAI = client

    async def chat(
        self,
        *,
        system: str = "",
        user: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        messages: list[dict[str, str]] = []
        if system.strip():
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = await self._client.chat.completions.create(**kwargs)
        if not resp.choices:
            raise OpenAIError("OpenAI response had no choices.")
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        return ChatResult(
            text=text,
            model=self.model,
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
        )

    async def resolve_spec(self) -> str:
        return self.spec

    async def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result

    async def __aenter__(self) -> OpenAIProvider:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()


__all__ = ["OpenAIError", "OpenAIProvider"]
