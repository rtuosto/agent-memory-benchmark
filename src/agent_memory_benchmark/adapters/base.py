"""``MemoryAdapter`` — the internal abstraction the runner drives.

This ABC is distinct from ``compat.MemorySystemShape``:

- :class:`MemorySystemShape` is a *structural* ``Protocol`` that memory
  systems can match without importing anything from this package. It's a
  spec.
- :class:`MemoryAdapter` is an internal ABC the runner depends on. It
  carries the runner's expectations about identity, state lifecycle, and
  optional persistence. Adapters translate between the runner's view and
  the target system's actual API.

Keeping these separate means that memory-system authors are never forced
to subclass anything from this package; they design a class that matches
the structural shape, and one of our built-in adapters bridges the gap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType

from ..types import AnswerResult, Session


class MemoryAdapter(ABC):
    """Internal contract between the runner and a memory-system backend.

    Subclasses MUST set ``memory_system_id`` and ``memory_version`` either
    as class attributes or during ``__init__`` before the first call.
    These values are hashed into cache keys and recorded in the run
    manifest, so they must be stable for any given build of the adapted
    memory system.
    """

    memory_system_id: str
    memory_version: str

    @abstractmethod
    async def ingest_session(self, session: Session, case_id: str) -> None:
        """Ingest one session's turns, preserving ``case_id`` isolation.

        Called in chronological order of ``session.session_index``. The
        runner does not interleave cases, so implementations may assume
        all sessions for a given ``case_id`` arrive back-to-back before
        any ``answer_question(case_id=...)`` call.
        """

    @abstractmethod
    async def answer_question(self, question: str, case_id: str) -> AnswerResult:
        """Produce one answer for the given case's QA turn.

        Returns an :class:`AnswerResult` with the generated text and
        (optionally) a retrieval trace. The runner independently measures
        wall time; ``retrieval_time_ms`` / ``generation_time_ms`` are
        informational.
        """

    @abstractmethod
    async def reset(self) -> None:
        """Drop all state so the adapter is ready for the next case group."""

    async def save_state(self, path: Path) -> None:
        """Serialize internal state to ``path`` (a directory).

        Default: raise :class:`NotImplementedError`. Adapters that can
        persist state override this to enable the ingestion cache — the
        runner will cache ``state.json`` under the ingestion subdir and
        skip re-ingestion on cache hit.
        """

        raise NotImplementedError(f"{type(self).__name__} does not implement save_state.")

    async def load_state(self, path: Path) -> None:
        """Restore state previously written by :meth:`save_state`."""

        raise NotImplementedError(f"{type(self).__name__} does not implement load_state.")

    @property
    def supports_persistence(self) -> bool:
        """True iff this adapter has real ``save_state``/``load_state`` impls.

        Default detection checks whether either method is still this
        base-class implementation. Subclasses that override persistence
        automatically flip this to True.
        """

        base_save = MemoryAdapter.save_state
        base_load = MemoryAdapter.load_state
        return type(self).save_state is not base_save or type(self).load_state is not base_load

    async def close(self) -> None:
        """Release any transport-level resources. Default: no-op."""

    async def __aenter__(self) -> MemoryAdapter:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()


__all__ = ["MemoryAdapter"]
