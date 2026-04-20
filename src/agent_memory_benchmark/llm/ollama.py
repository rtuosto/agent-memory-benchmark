"""Ollama provider — native ``/api/chat`` over ``httpx``.

No third-party dependency beyond ``httpx`` (already a required runtime
dependency). Uses Ollama's native chat endpoint (``/api/chat``) rather than
the OpenAI-compat shim at ``/v1/chat/completions`` so that options like
``num_ctx`` and ``format: "json"`` are first-class.

Digest resolution for cache-key pinning uses ``/api/tags`` (a stable listing
endpoint) — it reports each locally installed model with ``digest`` and
``modified_at``. ``/api/show`` is not used because its digest field is not
uniformly available across Ollama versions.
"""

from __future__ import annotations

import os
from types import TracebackType
from typing import Any

import httpx

from . import ChatResult, ProviderError


class OllamaError(ProviderError):
    """Raised when the Ollama server returns an error or unexpected payload."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


_DEFAULT_BASE_URL = "http://127.0.0.1:11434"


class OllamaProvider:
    """Chat against a local or networked Ollama daemon.

    ``base_url`` defaults to ``$OLLAMA_BASE_URL`` if set, else
    ``http://127.0.0.1:11434``. ``model`` is the raw model tag (e.g.
    ``llama3.1:8b``); the full spec is ``f"ollama:{model}"``.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        timeout: float = 600.0,
        num_ctx: int = 8192,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.spec = f"ollama:{model}"
        self._base = (base_url or os.getenv("OLLAMA_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")
        self._num_ctx = num_ctx
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)

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
        options: dict[str, Any] = {"temperature": temperature, "num_ctx": self._num_ctx}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if json_mode:
            payload["format"] = "json"

        resp = await self._client.post(f"{self._base}/api/chat", json=payload)
        if resp.status_code >= 400:
            raise OllamaError(
                f"Ollama /api/chat returned HTTP {resp.status_code}: {resp.text[:500]}",
                status_code=resp.status_code,
            )
        data = resp.json()
        message = data.get("message") or {}
        text = message.get("content") or ""
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        return ChatResult(
            text=text,
            model=self.model,
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
        )

    async def resolve_spec(self) -> str:
        """Return ``ollama:<model>@sha256:<digest>`` using ``/api/tags``.

        If the model is not present locally (Ollama lazily pulls, so this is
        possible before the first call), returns the unpinned spec unchanged.
        Callers that require a pinned spec for publication should surface
        that via the M3 guard, not by failing here.
        """

        resp = await self._client.get(f"{self._base}/api/tags")
        if resp.status_code >= 400:
            raise OllamaError(
                f"Ollama /api/tags returned HTTP {resp.status_code}: {resp.text[:500]}",
                status_code=resp.status_code,
            )
        data = resp.json()
        models = data.get("models") or []
        for entry in models:
            name = entry.get("name") or entry.get("model")
            if name == self.model:
                digest = entry.get("digest")
                if isinstance(digest, str) and digest:
                    digest = digest if digest.startswith("sha256:") else f"sha256:{digest}"
                    return f"{self.spec}@{digest}"
                break
        return self.spec

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> OllamaProvider:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()


__all__ = ["OllamaError", "OllamaProvider"]
