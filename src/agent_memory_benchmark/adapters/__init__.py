"""Adapter layer — transport-neutral abstraction the runner drives.

The benchmark runner holds exactly one :class:`MemoryAdapter` and calls
``ingest_session`` / ``answer_question`` / ``reset`` on it. Each adapter
subclass knows how to reach a specific transport:

- :class:`PythonAdapter` — imports an in-process class that matches
  ``agent_memory_benchmark.compat.MemorySystemShape``.
- :class:`FullContextAdapter` — reference "null memory" baseline that
  concatenates every ingested turn into the prompt.
- :class:`EngramAdapter` — wraps :class:`engram.EngramGraphMemorySystem`;
  ingests per-turn :class:`engram.Memory` objects and generates answers
  from the :class:`engram.RecallResult` via a supplied LLM provider.
- :class:`HttpAdapter` — speaks the REST contract documented in
  ``openapi.yaml`` / ``docs/http-api.md``.

The :func:`resolve_adapter` factory maps a CLI ``--memory`` string to a
concrete adapter. Grammar::

    full-context                             -> FullContextAdapter
    engram                                   -> EngramAdapter
    python:pkg.module:ClassName              -> PythonAdapter
    http(s)://host:port                      -> HttpAdapter
"""

from __future__ import annotations

from .base import MemoryAdapter
from .engram_adapter import EngramAdapter
from .factory import AdapterSpecError, resolve_adapter
from .full_context import FullContextAdapter
from .http_adapter import HttpAdapter, HttpAdapterError
from .python_adapter import PythonAdapter

__all__ = [
    "AdapterSpecError",
    "EngramAdapter",
    "FullContextAdapter",
    "HttpAdapter",
    "HttpAdapterError",
    "MemoryAdapter",
    "PythonAdapter",
    "resolve_adapter",
]
