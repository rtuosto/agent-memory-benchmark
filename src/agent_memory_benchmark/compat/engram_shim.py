"""Wrapper shim for engram's ``memory.system.MultiLayerMemory``.

The invariant for this repo is that memory systems have zero knowledge of
the benchmark. Engram's ``MultiLayerMemory`` diverges from
:class:`~agent_memory_benchmark.compat.MemorySystemShape` in two ways:

1. **Identity attributes** — engram stores its version as a module-level
   ``MULTI_LAYER_MEMORY_VERSION`` constant and has no ``memory_system_id``.
2. **Type shapes** — engram's ``ingest_session`` expects a
   ``benchmark.datasets.locomo.Session`` with ``DialogueTurn`` elements
   (fields ``speaker``, ``dia_id``, ``text``, ``blip_caption``), and its
   ``answer_question`` returns an engram-internal ``AnswerResult`` without
   the ``retrieved`` tuple our :class:`~agent_memory_benchmark.types.AnswerResult`
   carries.

This module bridges both entirely on the benchmark side — no engram patches.
The wrapper class :class:`EngramShim` declares the required identity attrs
itself, holds an inner ``MultiLayerMemory``, and translates types at the
boundary. Engram-side population of ``RetrievedUnit.source_turn_ids`` is a
separate engram-side improvement that unblocks evidence-keyed KPIs; without
it, quality / latency / footprint / throughput KPIs still work.

**Duck-typed sessions on ingest.** The shim tries to import engram's real
``benchmark.datasets.locomo.Session`` / ``DialogueTurn`` first and falls
back to lightweight namespace objects when those aren't importable (e.g.
a different engram version that exposes types under a different path).
Engram's code paths read session.turns via attribute access, so duck typing
suffices for normal operation; if engram ever `isinstance`-checks the
session type this would need tightening.

**Verification:** the mappers reflect the survey in commit history at the
time PR-7.5 was written. On first end-to-end run against a real engram
install, verify that ``ingest_session`` succeeds on the first session and
``answer_question`` returns the expected fields — any drift in engram's
type shape surfaces as an :class:`AttributeError` here, not silent data
corruption.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..types import AnswerResult, Session


class EngramShim:
    """Wrap ``memory.system.MultiLayerMemory`` for benchmark use.

    Instantiation fails loudly (:class:`ImportError`) when engram isn't
    importable — the module itself stays importable so the benchmark
    package and test suite don't require engram to be installed.
    """

    memory_system_id: ClassVar[str] = "engram"
    memory_version: str  # set in __init__ from engram's module constant

    def __init__(self, **kwargs: Any) -> None:
        try:
            from memory.system import (  # type: ignore[import-not-found]
                MULTI_LAYER_MEMORY_VERSION,
                MultiLayerMemory,
            )
        except ImportError as exc:
            raise ImportError(
                "EngramShim requires engram (the 'memory' package) to be importable. "
                "Install engram or add its source directory to PYTHONPATH. "
                "If you only want to test the benchmark against the full-context baseline, "
                "use '--memory full-context' instead."
            ) from exc

        self.memory_version = str(MULTI_LAYER_MEMORY_VERSION)
        self._inner = MultiLayerMemory(**kwargs)

    async def ingest_session(self, session: Session, case_id: str) -> None:
        await self._inner.ingest_session(_to_engram_session(session), case_id)

    async def answer_question(self, question: str, case_id: str) -> AnswerResult:
        raw = await self._inner.answer_question(question, case_id)
        return _from_engram_answer(raw)

    async def reset(self) -> None:
        await self._inner.reset()

    async def close(self) -> None:
        """Forward close() if the inner exposes one; otherwise no-op."""

        close = getattr(self._inner, "close", None)
        if close is None:
            return
        result = close()
        if hasattr(result, "__await__"):
            await result


def _to_engram_session(session: Session) -> Any:
    """Translate a benchmark :class:`Session` into engram's session type.

    Tries engram's real types first; falls back to duck-typed namespaces
    when they aren't importable. Field-name mapping:

    - ``session.session_index`` → engram ``session_index``
    - ``session.session_time``  → engram ``date_time``
    - ``session.turns[i].turn_id``       → engram ``dia_id``
    - ``session.turns[i].speaker``       → engram ``speaker``
    - ``session.turns[i].text``          → engram ``text``
    - ``session.turns[i].image_caption`` → engram ``blip_caption``
    """

    real_session, real_turn = _resolve_engram_session_types()
    turns = [
        real_turn(
            speaker=t.speaker,
            dia_id=t.turn_id,
            text=t.text,
            blip_caption=t.image_caption,
        )
        for t in session.turns
    ]
    return real_session(
        session_index=session.session_index,
        date_time=session.session_time,
        turns=turns,
    )


def _resolve_engram_session_types() -> tuple[Any, Any]:
    """Return ``(SessionType, TurnType)`` from engram or a duck-typed fallback.

    Cached on first call via module-level attribute so the import probe
    only happens once per process.
    """

    global _CACHED_ENGRAM_TYPES  # noqa: PLW0603  # intentional module cache
    if _CACHED_ENGRAM_TYPES is not None:
        return _CACHED_ENGRAM_TYPES
    try:
        from benchmark.datasets.locomo import (  # type: ignore[import-not-found]
            DialogueTurn,
        )
        from benchmark.datasets.locomo import (
            Session as EngramSession,
        )

        _CACHED_ENGRAM_TYPES = (EngramSession, DialogueTurn)
    except ImportError:
        _CACHED_ENGRAM_TYPES = (_DuckSession, _DuckTurn)
    return _CACHED_ENGRAM_TYPES


_CACHED_ENGRAM_TYPES: tuple[Any, Any] | None = None


class _DuckTurn:
    """Fallback turn object exposing engram's expected attribute names."""

    __slots__ = ("blip_caption", "dia_id", "speaker", "text")

    def __init__(self, *, speaker: str, dia_id: str, text: str, blip_caption: str | None) -> None:
        self.speaker = speaker
        self.dia_id = dia_id
        self.text = text
        self.blip_caption = blip_caption


class _DuckSession:
    """Fallback session object exposing engram's expected attribute names."""

    __slots__ = ("date_time", "session_index", "turns")

    def __init__(
        self,
        *,
        session_index: int,
        date_time: str | None,
        turns: list[_DuckTurn],
    ) -> None:
        self.session_index = session_index
        self.date_time = date_time
        self.turns = turns


def _from_engram_answer(raw: Any) -> AnswerResult:
    """Translate engram's answer-return into a benchmark :class:`AnswerResult`.

    Reads attributes permissively so the mapper survives minor engram
    schema shifts (extra fields are ignored; missing optional fields
    default to zero). If engram starts returning a ``retrieved`` tuple
    with ``source_turn_ids``, it passes through unchanged — this mapper
    makes no assumption about retrieved being absent forever.
    """

    retrieved = getattr(raw, "retrieved", ()) or ()
    return AnswerResult(
        answer=str(getattr(raw, "answer", "")),
        retrieval_time_ms=float(getattr(raw, "retrieval_time_ms", 0.0) or 0.0),
        generation_time_ms=float(getattr(raw, "generation_time_ms", 0.0) or 0.0),
        units_retrieved=int(getattr(raw, "units_retrieved", 0) or 0),
        tokens_retrieved=int(getattr(raw, "tokens_retrieved", 0) or 0),
        retrieved=tuple(retrieved),
    )


__all__ = ["EngramShim"]
