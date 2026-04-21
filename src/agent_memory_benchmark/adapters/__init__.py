"""Adapter layer — transport-neutral abstraction the runner drives.

The benchmark runner holds exactly one :class:`MemoryAdapter` and calls
``ingest_session`` / ``answer_question`` / ``reset`` on it. Each adapter
subclass knows how to reach a specific transport:

- :class:`PythonAdapter` — imports an in-process class that matches
  ``agent_memory_benchmark.compat.MemorySystemShape``.
- :class:`FullContextAdapter` — reference "null memory" baseline that
  concatenates every ingested turn into the prompt.
- *HttpAdapter* — lands in PR-10.

The :func:`resolve_adapter` factory maps a CLI ``--memory`` string to a
concrete adapter. Grammar::

    full-context                             -> FullContextAdapter
    python:pkg.module:ClassName              -> PythonAdapter
    http://host:port                         -> HttpAdapter (PR-10)
"""

from __future__ import annotations

from .base import MemoryAdapter
from .factory import AdapterSpecError, resolve_adapter
from .full_context import FullContextAdapter
from .python_adapter import PythonAdapter

__all__ = [
    "AdapterSpecError",
    "FullContextAdapter",
    "MemoryAdapter",
    "PythonAdapter",
    "resolve_adapter",
]
