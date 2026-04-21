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

import logging
import os
from types import TracebackType
from typing import Any

import httpx

from . import ChatResult, ProviderError

_log = logging.getLogger(__name__)


class OllamaError(ProviderError):
    """Raised when the Ollama server returns an error or unexpected payload."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


_DEFAULT_BASE_URL = "http://127.0.0.1:11434"
_CHARS_PER_TOKEN_ESTIMATE = 4
"""Conservative chars-per-token ratio used as a pre-flight truncation heuristic.

English text averages ~4 characters per token for BPE/SentencePiece tokenizers.
We use this only to decide whether to warn before dispatch; the authoritative
check happens after the call by comparing ``prompt_eval_count`` to ``num_ctx``.
"""


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

        # Pre-flight heuristic: if the prompt is clearly larger than
        # ``num_ctx`` can hold, warn. Ollama silently truncates in this case
        # — making the full-context baseline meaningless — so surfacing it
        # early gives the user a chance to bump ``--num-ctx`` and re-run.
        prompt_chars = len(system) + len(user)
        estimated_prompt_tokens = prompt_chars // _CHARS_PER_TOKEN_ESTIMATE
        if estimated_prompt_tokens > self._num_ctx:
            _log.warning(
                "Ollama prompt ~%d tokens exceeds num_ctx=%d for model %s; "
                "Ollama will silently truncate. Pass --num-ctx to allocate more.",
                estimated_prompt_tokens,
                self._num_ctx,
                self.model,
            )

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
        # Authoritative post-flight check using Ollama's own token count.
        # When prompt_eval_count >= num_ctx, Ollama has definitely truncated;
        # we emit a second-line warning so the log trail shows both the
        # heuristic guess (pre) and the actual count (post).
        if isinstance(prompt_tokens, int) and prompt_tokens >= self._num_ctx:
            _log.warning(
                "Ollama prompt_eval_count=%d >= num_ctx=%d for model %s "
                "(truncation is near-certain).",
                prompt_tokens,
                self._num_ctx,
                self.model,
            )
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
