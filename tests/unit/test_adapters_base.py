"""Tests for :class:`agent_memory_benchmark.adapters.base.MemoryAdapter`."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_memory_benchmark.adapters.base import MemoryAdapter
from agent_memory_benchmark.types import AnswerResult, Session


class _MinimalAdapter(MemoryAdapter):
    """Just enough to satisfy the ABC's abstract methods."""

    memory_system_id = "minimal"
    memory_version = "0.0.1"

    def __init__(self) -> None:
        self.closed = False

    async def ingest_session(self, session: Session, case_id: str) -> None:
        return None

    async def answer_question(self, question: str, case_id: str) -> AnswerResult:
        return AnswerResult(answer="", retrieval_time_ms=0.0, generation_time_ms=0.0)

    async def reset(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


class _PersistentAdapter(_MinimalAdapter):
    memory_system_id = "persistent"

    async def save_state(self, path: Path) -> None:
        (path / "state.json").write_text("{}", encoding="utf-8")

    async def load_state(self, path: Path) -> None:
        (path / "state.json").read_text(encoding="utf-8")


def test_cannot_instantiate_abstract_base() -> None:
    with pytest.raises(TypeError):
        MemoryAdapter()  # type: ignore[abstract]


async def test_default_save_state_raises_not_implemented(tmp_path: Path) -> None:
    adapter = _MinimalAdapter()
    with pytest.raises(NotImplementedError, match="save_state"):
        await adapter.save_state(tmp_path)
    with pytest.raises(NotImplementedError, match="load_state"):
        await adapter.load_state(tmp_path)


def test_supports_persistence_default_false() -> None:
    assert _MinimalAdapter().supports_persistence is False


def test_supports_persistence_true_when_overridden() -> None:
    assert _PersistentAdapter().supports_persistence is True


async def test_context_manager_invokes_close() -> None:
    adapter = _MinimalAdapter()
    async with adapter as entered:
        assert entered is adapter
        assert adapter.closed is False
    assert adapter.closed is True


async def test_default_close_is_noop() -> None:
    """A subclass that doesn't override close() should still work under `async with`."""

    class _NoClose(MemoryAdapter):
        memory_system_id = "nc"
        memory_version = "0"

        async def ingest_session(self, session: Session, case_id: str) -> None:
            return None

        async def answer_question(self, question: str, case_id: str) -> AnswerResult:
            return AnswerResult(answer="", retrieval_time_ms=0.0, generation_time_ms=0.0)

        async def reset(self) -> None:
            return None

    async with _NoClose():
        pass
