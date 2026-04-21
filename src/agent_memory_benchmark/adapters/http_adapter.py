"""``HttpAdapter`` — reach a memory system over REST.

Maps the runner's three abstract methods (``ingest_session`` /
``answer_question`` / ``reset``) onto a small REST surface documented in
``openapi.yaml`` and ``docs/http-api.md``. Any service that implements that
contract is benchmarkable with zero Python-level coupling: the adapter
speaks JSON over HTTP, no target-class import required.

Endpoints (v1):

    GET  /v1/identity     -> {memory_system_id, memory_version, ...}
    POST /v1/ingest       -> ingest one Session for a case
    POST /v1/answer       -> produce one AnswerResult for a question
    POST /v1/reset        -> drop all state

Optional (persistence — probed at connect time via identity flags):

    GET  /v1/state        -> state bytes (opaque to the adapter)
    PUT  /v1/state        -> restore state

Identity is fetched at :meth:`open` so ``memory_system_id`` and
``memory_version`` flow into cache keys and the manifest before the
first request against the benchmark's state. The adapter refuses to
serve any call (including ``ingest_session``) before ``open`` — a
misconfigured service should fail loudly at assembly time, not halfway
through ingestion.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import TracebackType
from typing import Any

import httpx

from ..types import AnswerResult, RetrievedUnit, Session
from .base import MemoryAdapter

_log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 600.0
"""Matches the Ollama provider timeout; memory systems over HTTP often
do real retrieval work that can take tens of seconds on cold caches."""


class HttpAdapterError(Exception):
    """Raised when the remote service returns an error or unexpected payload.

    ``status_code`` is populated for HTTP-level errors (non-2xx). Transport
    errors (connection refused, timeout) surface with ``status_code=None``
    and the original ``httpx`` exception chained via ``__cause__``.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class HttpAdapter(MemoryAdapter):
    """Thin HTTP client that satisfies :class:`MemoryAdapter`.

    Construction does not make network calls; :meth:`open` hits
    ``GET /v1/identity`` to populate ``memory_system_id`` and
    ``memory_version``. Use :meth:`connect` as the async factory when you
    want a single-call open-and-return.
    """

    # Provisional placeholders — replaced on :meth:`open`. Users who try to
    # read them before ``open`` get a clear error via the property below.
    _identity_populated: bool = False

    def __init__(
        self,
        base_url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not base_url or not isinstance(base_url, str):
            raise ValueError(f"HttpAdapter base_url must be a non-empty string (got {base_url!r}).")
        self._base = base_url.rstrip("/")
        self._headers = dict(headers or {})
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout, headers=self._headers)
        self._supports_persistence_remote = False

    @classmethod
    async def connect(
        cls,
        base_url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ) -> HttpAdapter:
        """Construct and :meth:`open` in one call."""

        adapter = cls(base_url, headers=headers, timeout=timeout, client=client)
        await adapter.open()
        return adapter

    async def open(self) -> None:
        """Fetch ``/v1/identity`` and populate adapter-level fields.

        Idempotent — re-calling re-reads identity (useful if the remote
        service restarts between case groups).
        """

        payload = await self._request_json("GET", "/v1/identity")
        try:
            self.memory_system_id = _require_str(payload, "memory_system_id")
            self.memory_version = _require_str(payload, "memory_version")
        except KeyError as exc:
            raise HttpAdapterError(
                f"GET /v1/identity response is missing required field: {exc}. See docs/http-api.md."
            ) from exc
        self._supports_persistence_remote = bool(payload.get("supports_persistence", False))
        self._identity_populated = True

    async def ingest_session(self, session: Session, case_id: str) -> None:
        self._require_open()
        body = {"case_id": case_id, "session": _session_to_json(session)}
        await self._request_json("POST", "/v1/ingest", body=body)

    async def answer_question(self, question: str, case_id: str) -> AnswerResult:
        self._require_open()
        body = {"case_id": case_id, "question": question}
        payload = await self._request_json("POST", "/v1/answer", body=body)
        return _answer_from_json(payload)

    async def reset(self) -> None:
        self._require_open()
        await self._request_json("POST", "/v1/reset", body={})

    async def save_state(self, path: Path) -> None:
        self._require_open()
        if not self._supports_persistence_remote:
            raise NotImplementedError(
                f"Remote service at {self._base} reports supports_persistence=false."
            )
        # GET /v1/state returns opaque bytes; we persist them verbatim so a
        # future load_state can feed them back unchanged. The benchmark
        # never inspects the body — it's the service's own state format.
        resp = await self._client.get(f"{self._base}/v1/state", headers=self._headers)
        self._raise_for_status(resp, method="GET", path="/v1/state")
        path.mkdir(parents=True, exist_ok=True)
        (path / "state.bin").write_bytes(resp.content)

    async def load_state(self, path: Path) -> None:
        self._require_open()
        if not self._supports_persistence_remote:
            raise NotImplementedError(
                f"Remote service at {self._base} reports supports_persistence=false."
            )
        blob = (path / "state.bin").read_bytes()
        resp = await self._client.put(
            f"{self._base}/v1/state",
            content=blob,
            headers={**self._headers, "Content-Type": "application/octet-stream"},
        )
        self._raise_for_status(resp, method="PUT", path="/v1/state")

    @property
    def supports_persistence(self) -> bool:
        # Override the base-class reflection: HttpAdapter always defines
        # save_state/load_state, but whether they *work* depends on the
        # remote service's own capability flag.
        return self._identity_populated and self._supports_persistence_remote

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> HttpAdapter:
        if not self._identity_populated:
            await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    def _require_open(self) -> None:
        if not self._identity_populated:
            raise HttpAdapterError(
                "HttpAdapter must be opened (call `await adapter.open()` or use "
                "`HttpAdapter.connect(...)`) before any memory operation."
            )

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self._base}{path}"
        try:
            if method == "GET":
                resp = await self._client.get(url, headers=self._headers)
            elif method == "POST":
                resp = await self._client.post(url, json=body or {}, headers=self._headers)
            else:  # pragma: no cover — defensive; only GET/POST used here
                raise ValueError(f"Unsupported HTTP method {method!r}")
        except httpx.HTTPError as exc:
            raise HttpAdapterError(
                f"Transport error on {method} {url}: {exc}",
            ) from exc
        self._raise_for_status(resp, method=method, path=path)
        if resp.status_code == 204 or not resp.content:
            return {}
        try:
            return resp.json()  # type: ignore[no-any-return]
        except json.JSONDecodeError as exc:
            raise HttpAdapterError(
                f"{method} {path} returned non-JSON body: {resp.text[:500]!r}",
                status_code=resp.status_code,
            ) from exc

    @staticmethod
    def _raise_for_status(resp: httpx.Response, *, method: str, path: str) -> None:
        if resp.status_code >= 400:
            raise HttpAdapterError(
                f"{method} {path} returned HTTP {resp.status_code}: {resp.text[:500]}",
                status_code=resp.status_code,
            )


def _session_to_json(session: Session) -> dict[str, Any]:
    return {
        "session_index": session.session_index,
        "session_time": session.session_time,
        "session_id": session.session_id,
        "turns": [
            {
                "turn_id": t.turn_id,
                "speaker": t.speaker,
                "text": t.text,
                "timestamp": t.timestamp,
                "image_caption": t.image_caption,
            }
            for t in session.turns
        ],
    }


def _answer_from_json(payload: dict[str, Any]) -> AnswerResult:
    try:
        answer = payload["answer"]
    except KeyError as exc:
        raise HttpAdapterError(
            f"POST /v1/answer response missing 'answer' field: {payload!r}",
        ) from exc
    if not isinstance(answer, str):
        raise HttpAdapterError(
            f"POST /v1/answer 'answer' must be a string, got {type(answer).__name__}.",
        )
    retrieved_raw = payload.get("retrieved") or ()
    retrieved: tuple[RetrievedUnit, ...] = tuple(
        _retrieved_from_json(entry) for entry in retrieved_raw
    )
    return AnswerResult(
        answer=answer,
        retrieval_time_ms=float(payload.get("retrieval_time_ms", 0.0) or 0.0),
        generation_time_ms=float(payload.get("generation_time_ms", 0.0) or 0.0),
        units_retrieved=int(payload.get("units_retrieved", 0) or 0),
        tokens_retrieved=int(payload.get("tokens_retrieved", 0) or 0),
        retrieved=retrieved,
    )


def _retrieved_from_json(entry: dict[str, Any]) -> RetrievedUnit:
    source = entry.get("source_turn_ids") or ()
    return RetrievedUnit(
        unit_id=str(entry.get("unit_id") or ""),
        text=str(entry.get("text") or ""),
        score=_as_optional_float(entry.get("score")),
        source_turn_ids=tuple(str(s) for s in source),
    )


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise KeyError(key)
    return value


__all__ = ["HttpAdapter", "HttpAdapterError"]
