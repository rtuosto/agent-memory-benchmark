"""Tests for ``MemorySystemShape`` structural typing.

The Protocol is ``runtime_checkable``, so ``isinstance`` works against any
class that exposes the right methods and attributes. This test pins that
behavior so ``PythonAdapter`` can rely on it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_memory_benchmark import (
    AnswerResult,
    MemorySystemShape,
    PersistableMemorySystemShape,
    Session,
)


class _Compatible:
    memory_system_id = "compatible"
    memory_version = "0.1.0"

    async def ingest_session(self, session: Session, case_id: str) -> None:  # noqa: ARG002
        return None

    async def answer_question(self, question: str, case_id: str) -> AnswerResult:  # noqa: ARG002
        return AnswerResult(answer="", retrieval_time_ms=0.0, generation_time_ms=0.0)

    async def reset(self) -> None:
        return None


class _Persistable(_Compatible):
    memory_system_id = "persistable"

    async def save_state(self, path: Path) -> None:  # noqa: ARG002
        return None

    async def load_state(self, path: Path) -> None:  # noqa: ARG002
        return None


class _MissingMethod:
    memory_system_id = "missing"
    memory_version = "0.1.0"

    async def ingest_session(self, session: Session, case_id: str) -> None:  # noqa: ARG002
        return None

    # answer_question missing


class _MissingAttr:
    memory_version = "0.1.0"

    async def ingest_session(self, session: Session, case_id: str) -> None:  # noqa: ARG002
        return None

    async def answer_question(self, question: str, case_id: str) -> AnswerResult:  # noqa: ARG002
        return AnswerResult(answer="", retrieval_time_ms=0.0, generation_time_ms=0.0)

    async def reset(self) -> None:
        return None

    # memory_system_id missing


def test_compatible_matches_shape() -> None:
    instance = _Compatible()
    assert isinstance(instance, MemorySystemShape)


def test_missing_method_fails_shape() -> None:
    instance = _MissingMethod()
    assert not isinstance(instance, MemorySystemShape)


def test_missing_attribute_fails_shape() -> None:
    instance = _MissingAttr()
    assert not isinstance(instance, MemorySystemShape)


def test_persistable_matches_both_shapes() -> None:
    instance = _Persistable()
    assert isinstance(instance, MemorySystemShape)
    assert isinstance(instance, PersistableMemorySystemShape)


def test_compatible_not_persistable() -> None:
    instance = _Compatible()
    assert isinstance(instance, MemorySystemShape)
    assert not isinstance(instance, PersistableMemorySystemShape)


@pytest.mark.parametrize("klass", [_Compatible, _Persistable])
def test_class_level_runtime_check(klass: type) -> None:
    # runtime_checkable Protocol isinstance works on instances, not classes
    # but we can at least sanity-check that attribute access works as designed.
    assert hasattr(klass, "memory_system_id")
    assert hasattr(klass, "ingest_session")
    assert hasattr(klass, "answer_question")
    assert hasattr(klass, "reset")
