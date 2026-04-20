"""Memory-system compatibility spec.

This module publishes a ``typing.Protocol`` that describes the shape the
``PythonAdapter`` expects of an in-process memory system. Memory systems
that want zero-glue benchmarking can design their class to match the shape
below — they DO NOT need to import this module or inherit from anything in
this package.

The Protocol is ``runtime_checkable`` so ``PythonAdapter`` can use
``isinstance(instance, MemorySystemShape)`` for an early friendly error
before the first call. Structural checks cannot verify argument types or
async-ness, so adapters still perform targeted attribute checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from .types import AnswerResult, Session


@runtime_checkable
class MemorySystemShape(Protocol):
    """Structural protocol for Python-adapter-compatible memory systems.

    Required identity attributes (class- or instance-level):

    - ``memory_system_id`` — short stable slug, e.g. ``"engram"``.
    - ``memory_version`` — bumped whenever state layout or retrieval/answer
      semantics change. Hashed into cache keys; stale caches are invalidated
      automatically on version bump.

    Required async methods:

    - ``ingest_session(session, case_id)`` — deterministic; called in
      chronological order per case_id.
    - ``answer_question(question, case_id)`` — returns an ``AnswerResult``
      with the generated answer and (optionally) a retrieval trace used for
      evidence-keyed KPIs.
    - ``reset()`` — drops all ingested state before the next case.

    Optional async methods — raising ``NotImplementedError`` (or omitting
    them entirely) disables the ingestion cache for this system; the runner
    falls back to re-ingesting each run:

    - ``save_state(path)`` — serialize to a directory.
    - ``load_state(path)`` — restore from a serialized directory.
    """

    memory_system_id: str
    memory_version: str

    async def ingest_session(self, session: Session, case_id: str) -> None: ...

    async def answer_question(self, question: str, case_id: str) -> AnswerResult: ...

    async def reset(self) -> None: ...


@runtime_checkable
class PersistableMemorySystemShape(MemorySystemShape, Protocol):
    """Opt-in extension of ``MemorySystemShape`` for persistence support.

    Implementing this shape lets the runner cache ingestion state between
    runs (keyed by memory_system_id, memory_version, dataset descriptor,
    case_id). Systems that cannot serialize their state should implement
    only ``MemorySystemShape`` and will re-ingest each run.
    """

    async def save_state(self, path: Path) -> None: ...

    async def load_state(self, path: Path) -> None: ...


__all__ = ["MemorySystemShape", "PersistableMemorySystemShape"]
