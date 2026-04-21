"""Tests for ``compat.engram_shim`` that do NOT require engram installed.

Engram isn't a dev dependency of the benchmark; these tests verify the
shim stays importable without engram, that instantiation fails with a
clear message, and that the private mappers round-trip correctly against
the duck-typed fallback types.

An integration test against a real engram install belongs in
``tests/integration/`` and will land with PR-13 when the recorded HTTP
fixtures + real-engram smoke path come online.
"""

from __future__ import annotations

import pytest

from agent_memory_benchmark.compat import engram_shim
from agent_memory_benchmark.compat.engram_shim import (
    EngramShim,
    _DuckSession,
    _DuckTurn,
    _from_engram_answer,
    _to_engram_session,
)
from agent_memory_benchmark.types import AnswerResult, RetrievedUnit, Session, Turn


def _sample_session() -> Session:
    return Session(
        session_index=3,
        turns=(
            Turn(
                turn_id="t1",
                speaker="user",
                text="hi",
                timestamp=None,
                image_caption="a cat on a mat",
            ),
            Turn(turn_id="t2", speaker="assistant", text="hello", timestamp=None),
        ),
        session_time="2026-04-20T10:00:00Z",
        session_id="sess_1",
    )


def test_shim_has_static_memory_system_id() -> None:
    """The identity attribute is declared on the class, not on engram."""

    assert EngramShim.memory_system_id == "engram"


def test_instantiation_without_engram_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Users see a targeted message, not an opaque ImportError."""

    real_import = __import__

    def _block_memory(name: str, *args: object, **kwargs: object) -> object:
        if name == "memory.system" or name.startswith("memory.system."):
            raise ImportError("mock: engram not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _block_memory)
    with pytest.raises(ImportError, match="EngramShim requires engram"):
        EngramShim()


def test_to_engram_session_uses_duck_fallback_when_real_types_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without benchmark.datasets.locomo, the fallback types are used."""

    # Reset the module-level cache so the probe runs again this test.
    monkeypatch.setattr(engram_shim, "_CACHED_ENGRAM_TYPES", None)

    real_import = __import__

    def _block_predecessor(name: str, *args: object, **kwargs: object) -> object:
        if name == "benchmark.datasets.locomo" or name.startswith("benchmark.datasets.locomo."):
            raise ImportError("mock: predecessor benchmark not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _block_predecessor)

    result = _to_engram_session(_sample_session())
    assert isinstance(result, _DuckSession)
    assert result.session_index == 3
    assert result.date_time == "2026-04-20T10:00:00Z"
    assert len(result.turns) == 2
    first = result.turns[0]
    assert isinstance(first, _DuckTurn)
    assert first.dia_id == "t1"
    assert first.speaker == "user"
    assert first.text == "hi"
    assert first.blip_caption == "a cat on a mat"


def test_to_engram_session_preserves_turn_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engram_shim, "_CACHED_ENGRAM_TYPES", (_DuckSession, _DuckTurn))
    session = _sample_session()
    result = _to_engram_session(session)
    assert [t.dia_id for t in result.turns] == ["t1", "t2"]


def test_to_engram_session_handles_none_image_caption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engram_shim, "_CACHED_ENGRAM_TYPES", (_DuckSession, _DuckTurn))
    session = Session(
        session_index=1,
        turns=(Turn(turn_id="t1", speaker="user", text="hi", image_caption=None),),
        session_time=None,
    )
    result = _to_engram_session(session)
    assert result.turns[0].blip_caption is None
    assert result.date_time is None


def test_from_engram_answer_builds_our_answer_result() -> None:
    """Engram's answer object has attributes, not fields — duck-typed."""

    class _EngramAnswer:
        answer = "a shell necklace"
        retrieval_time_ms = 12.5
        generation_time_ms = 487.2
        units_retrieved = 4
        tokens_retrieved = 1234

    result = _from_engram_answer(_EngramAnswer())
    assert isinstance(result, AnswerResult)
    assert result.answer == "a shell necklace"
    assert result.retrieval_time_ms == pytest.approx(12.5)
    assert result.generation_time_ms == pytest.approx(487.2)
    assert result.units_retrieved == 4
    assert result.tokens_retrieved == 1234
    assert result.retrieved == ()


def test_from_engram_answer_preserves_retrieved_when_present() -> None:
    """If engram starts populating retrieved, we pass it through."""

    retrieved_unit = RetrievedUnit(
        unit_id="u1", text="shell necklace", score=0.9, source_turn_ids=("t5",)
    )

    class _EngramAnswer:
        answer = "shell necklace"
        retrieval_time_ms = 5.0
        generation_time_ms = 100.0
        units_retrieved = 1
        tokens_retrieved = 100
        retrieved = (retrieved_unit,)

    result = _from_engram_answer(_EngramAnswer())
    assert result.retrieved == (retrieved_unit,)


def test_from_engram_answer_tolerates_missing_optional_fields() -> None:
    """Missing numeric fields default to zero, not AttributeError."""

    class _Minimal:
        answer = "hi"

    result = _from_engram_answer(_Minimal())
    assert result.answer == "hi"
    assert result.retrieval_time_ms == 0.0
    assert result.generation_time_ms == 0.0
    assert result.units_retrieved == 0
    assert result.tokens_retrieved == 0


def test_from_engram_answer_coerces_null_timings() -> None:
    """Engram may legitimately report ``None`` for timings we still want as 0."""

    class _NullTimings:
        answer = "x"
        retrieval_time_ms = None
        generation_time_ms = None
        units_retrieved = None
        tokens_retrieved = None

    result = _from_engram_answer(_NullTimings())
    assert result.retrieval_time_ms == 0.0
    assert result.tokens_retrieved == 0
