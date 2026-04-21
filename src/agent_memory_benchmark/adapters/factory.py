"""Adapter grammar resolver.

Maps a CLI ``--memory`` string to one of the concrete adapters. The
grammar is intentionally small so adapter selection can be reasoned about
directly from the command line::

    full-context                 -> FullContextAdapter
    python:pkg.module:ClassName  -> PythonAdapter
    http://host:port             -> HttpAdapter (PR-10)

The factory is separated from ``__init__`` so importing an adapter does
not force the entire package to load (e.g., importing
:class:`FullContextAdapter` on its own does not pull the ``httpx``-backed
HTTP adapter once that lands in PR-10).
"""

from __future__ import annotations

from typing import Any

from ..llm import LLMProvider
from .base import MemoryAdapter
from .full_context import FullContextAdapter
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
    """

    if not isinstance(spec, str) or not spec.strip():
        raise AdapterSpecError("Adapter spec is empty.")
    spec = spec.strip()

    if spec.startswith("http://") or spec.startswith("https://"):
        raise AdapterSpecError(
            "HTTP adapter lands in PR-10. Use 'full-context' or 'python:...' for now."
        )

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
        return FullContextAdapter(answer_provider)

    if kind == "python":
        if not target:
            raise AdapterSpecError("python adapter needs a target: python:pkg.module:ClassName")
        return PythonAdapter.from_spec(
            target,
            config=config,
            session_mapper=session_mapper,
            result_mapper=result_mapper,
        )

    raise AdapterSpecError(
        f"Unknown adapter kind {kind!r} in spec {spec!r}; supported: 'full-context', 'python:...'."
    )


__all__ = ["AdapterSpecError", "resolve_adapter"]
