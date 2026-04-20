"""LLM provider abstractions.

The benchmark talks to answer- and judge-generation backends through a small
``Protocol`` plus a factory. Two concrete providers ship in-tree:

- :class:`OllamaProvider` (``llm.ollama``) — native ``/api/chat`` over httpx.
  No third-party dependency beyond ``httpx``.
- :class:`OpenAIProvider` (``llm.openai``) — uses the ``openai`` package,
  guarded behind the ``[openai]`` extra. Importing the provider without the
  extra installed raises :class:`ProviderUnavailableError` with a clear hint.

A spec string has the form ``"<provider>:<model>"``. Examples:

- ``"ollama:llama3.1:8b"`` → provider=``ollama``, model=``llama3.1:8b``
- ``"openai:gpt-4o-mini-2024-07-18"`` → provider=``openai``, model=``gpt-4o-mini-2024-07-18``

Ollama model names legitimately contain a colon (``llama3.1:8b``), so
:func:`parse_spec` splits on the *first* colon only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class ProviderError(Exception):
    """Base class for LLM-provider errors raised by this package."""


class ProviderUnavailableError(ProviderError):
    """An optional provider was requested but its dependency is not installed."""


class InvalidSpecError(ProviderError, ValueError):
    """Spec string did not match ``<provider>:<model>``."""


@dataclass(frozen=True)
class ChatResult:
    """Outcome of a single chat completion.

    The runner measures wall time independently; ``prompt_tokens`` /
    ``completion_tokens`` are opportunistic and may be ``None`` when the
    backend does not report them (Ollama native ``/api/chat`` does, OpenAI
    always does — but we do not rely on them for KPIs).
    """

    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@runtime_checkable
class LLMProvider(Protocol):
    """Async chat-style LLM backend.

    ``spec`` is the original ``provider:model`` string the caller passed in.
    ``resolve_spec`` returns a digest-pinned variant safe to hash into cache
    keys — for OpenAI the dated model IDs are already immutable so it echoes
    the spec; for Ollama it queries ``/api/tags`` and appends
    ``@sha256:<digest>``.
    """

    spec: str
    model: str

    async def chat(
        self,
        *,
        system: str = "",
        user: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult: ...

    async def resolve_spec(self) -> str: ...

    async def close(self) -> None: ...


def parse_spec(spec: str) -> tuple[str, str]:
    """Split ``"<provider>:<model>"`` into ``(provider, model)``.

    Splits on the first colon only; Ollama model names legitimately contain
    further colons (``llama3.1:8b``). Provider is lowercased; model is
    preserved verbatim (whitespace stripped).
    """

    if not isinstance(spec, str) or ":" not in spec:
        raise InvalidSpecError(
            f"Invalid LLM spec {spec!r}; expected '<provider>:<model>' "
            "(e.g. 'ollama:llama3.1:8b' or 'openai:gpt-4o-mini-2024-07-18')."
        )
    provider, _, model = spec.partition(":")
    provider = provider.strip().lower()
    model = model.strip()
    if not provider or not model:
        raise InvalidSpecError(
            f"Invalid LLM spec {spec!r}; provider and model must both be non-empty."
        )
    return provider, model


def build_provider(
    spec: str,
    *,
    ollama_base_url: str | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    timeout: float = 600.0,
) -> LLMProvider:
    """Build a provider from a spec string.

    Unknown providers raise :class:`InvalidSpecError`. The ``openai`` kind
    raises :class:`ProviderUnavailableError` if the optional ``[openai]``
    extra is not installed.
    """

    provider, model = parse_spec(spec)
    if provider == "ollama":
        from .ollama import OllamaProvider

        return OllamaProvider(model=model, base_url=ollama_base_url, timeout=timeout)
    if provider == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider(
            model=model,
            api_key=openai_api_key,
            base_url=openai_base_url,
            timeout=timeout,
        )
    raise InvalidSpecError(
        f"Unknown provider {provider!r} in spec {spec!r}; supported providers: 'ollama', 'openai'."
    )


__all__ = [
    "ChatResult",
    "InvalidSpecError",
    "LLMProvider",
    "ProviderError",
    "ProviderUnavailableError",
    "build_provider",
    "parse_spec",
]
