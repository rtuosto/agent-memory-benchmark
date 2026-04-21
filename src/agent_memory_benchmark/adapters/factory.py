"""Adapter grammar resolver.

Maps a CLI ``--memory`` string to one of the concrete adapters. The
grammar is intentionally small so adapter selection can be reasoned about
directly from the command line::

    full-context                 -> FullContextAdapter
    engram                       -> EngramAdapter
    python:pkg.module:ClassName  -> PythonAdapter
    http(s)://host:port          -> HttpAdapter

The factory is separated from ``__init__`` so importing an adapter does
not force the entire package to load.
"""

from __future__ import annotations

from typing import Any

from ..llm import LLMProvider
from .base import MemoryAdapter
from .engram_adapter import EngramAdapter
from .full_context import FullContextAdapter
from .http_adapter import HttpAdapter
from .python_adapter import PythonAdapter, ResultMapper, SessionMapper


class AdapterSpecError(ValueError):
    """Raised when ``--memory`` cannot be resolved to an adapter."""


def resolve_adapter(
    spec: str,
    *,
    config: dict[str, Any] | None = None,
    answer_provider: LLMProvider | None = None,
    session_mapper: SessionMapper | None = None,
    result_mapper: ResultMapper | None = None,
    http_headers: dict[str, str] | None = None,
) -> MemoryAdapter:
    """Resolve a ``--memory`` spec string to a concrete :class:`MemoryAdapter`.

    ``config`` is forwarded to :meth:`PythonAdapter.from_spec` as target
    constructor kwargs; for ``full-context`` it is currently unused.
    ``answer_provider`` is required for the full-context baseline (it
    does its own generation).

    ``session_mapper`` / ``result_mapper`` forward to
    :class:`PythonAdapter` for boundary type translation; both are
    no-ops for the other adapter kinds (passing them is an error if
    ``kind != "python"`` so misconfigurations surface early).

    ``http_headers`` forward to :class:`HttpAdapter`; passing it with a
    non-``http*`` spec is an error for the same symmetry reason as the
    mapper kwargs above. The returned :class:`HttpAdapter` has NOT yet
    called :meth:`HttpAdapter.open`; the runner must do so before the
    first ingest call so identity flows into cache keys.
    """

    if not isinstance(spec, str) or not spec.strip():
        raise AdapterSpecError("Adapter spec is empty.")
    spec = spec.strip()

    if spec.startswith("http://") or spec.startswith("https://"):
        if config:
            raise AdapterSpecError(
                "--memory-config is only valid for python adapters; the HTTP "
                "adapter configures the remote service via headers, not ctor kwargs."
            )
        if session_mapper is not None or result_mapper is not None:
            raise AdapterSpecError(
                "session_mapper / result_mapper are only valid for python adapters."
            )
        return HttpAdapter(spec, headers=http_headers)

    kind, _, target = spec.partition(":")
    kind = kind.strip().lower()

    if kind == "full-context" or spec == "full-context":
        if answer_provider is None:
            raise AdapterSpecError(
                "full-context requires an answer_provider (it generates its own answers)."
            )
        if session_mapper is not None or result_mapper is not None:
            raise AdapterSpecError(
                "session_mapper / result_mapper are only valid for python adapters; "
                "full-context has no target type to translate."
            )
        if http_headers is not None:
            raise AdapterSpecError("--memory-header is only valid for http adapters.")
        return FullContextAdapter(answer_provider)

    if kind == "engram" or spec == "engram":
        if answer_provider is None:
            raise AdapterSpecError(
                "engram requires an answer_provider (engram itself never calls an LLM; "
                "the adapter generates the answer from the RecallResult)."
            )
        if session_mapper is not None or result_mapper is not None:
            raise AdapterSpecError(
                "session_mapper / result_mapper are only valid for python adapters."
            )
        if http_headers is not None:
            raise AdapterSpecError("--memory-header is only valid for http adapters.")
        return EngramAdapter(answer_provider, **(config or {}))

    if kind == "python":
        if not target:
            raise AdapterSpecError("python adapter needs a target: python:pkg.module:ClassName")
        if http_headers is not None:
            raise AdapterSpecError("--memory-header is only valid for http adapters.")
        return PythonAdapter.from_spec(
            target,
            config=config,
            session_mapper=session_mapper,
            result_mapper=result_mapper,
        )

    raise AdapterSpecError(
        f"Unknown adapter kind {kind!r} in spec {spec!r}; "
        "supported: 'full-context', 'engram', 'python:...', 'http(s)://...'."
    )


__all__ = ["AdapterSpecError", "resolve_adapter"]
